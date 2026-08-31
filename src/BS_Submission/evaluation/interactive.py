"""The challenge's interactive correction protocol.

A case is predicted repeatedly. Round 0 is unguided; each later round is given one more
simulated scribble, drawn on the model's own largest remaining error, as
``autoPETV/interactive/interactive_loop.py`` does it. Scoring is the official
:class:`MetricEvaluator`, not a reimplementation.

THE SCRIBBLE RULE (interactive_loop.py, reproduced exactly)::

    overseg  = (pred == 1) & (gt == 0)      # false positives -> BACKGROUND scribble
    underseg = (pred == 0) & (gt == 1)      # false negatives -> FOREGROUND scribble
    _, _, fp = simulate_scribble_from_label(overseg,  strategy)
    _, _, fn = simulate_scribble_from_label(underseg, strategy)
    if fp <= fn: tumor      += scribbles_fg
    else:        background += scribbles_bg

``fp`` and ``fn`` are each scribble's returned ``size`` -- the **scribble length**, not the
error volume. The rule is "correct whichever error drew the longer centreline, ties to
foreground". The simulator is imported from the vendored challenge code, never reimplemented.

Everything is scored on each case's **original** grid: the probability is reverse-resampled
before thresholding and compared against ``labels/<case>_orig.npy``, which is where the
annotation lives.

DECLARED DEVIATIONS FROM interactive_loop.py

1. ``rounds=5`` gives **6** inference passes (r0 plus 5 scribbles). The repo's ``max_iters=5``
   yields only 4 scribbles, because its first iteration adds none.
2. Inference runs in-process through :func:`predict_case` -- the same function ``bs-predict``
   calls -- rather than by invoking a container through ``bash test.sh``. The arithmetic is
   identical; only the process boundary and the ``.mha`` front end are skipped.
3. Cases whose ground truth has no lesions have an undefined Dice and F1 under the official
   scorer, which returns NaN. Those cases are excluded from the averages, as the official
   evaluation does, rather than counted as perfect or as zero.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np

from ..inference.predict import Prompts, predict_case
from .upstream.metrics import MetricEvaluator
from .upstream.simulate_scribbles import simulate_scribble_from_label

OVERLAP_THRESHOLD = 0.1       # official lesion-match IoU
CONNECTIVITY = 18             # official cc3d connectivity
STRATEGIES = ("centerline", "random", "boundary")
DEFAULT_ROUNDS = 5


def case_id(case: str) -> str:
    """Stable short identifier for a case name.

    ``hash()`` is salted per process for strings, so using it here would hand the same case a
    different id on every run -- resume would never match and saved masks would be orphaned.
    """
    return hashlib.sha1(case.encode()).hexdigest()[:12]


# --- scribbles ---------------------------------------------------------------------------


def safe_simulate(mask, strategy: str, seed: int) -> tuple[list, int]:
    """``simulate_scribble_from_label``, guarded.

    The vendored function returns three values normally but only two on its fallback path
    (``if best_component is None: return [], False``), so unpacking it directly raises on an
    empty error mask. Empty error masks are routine -- a lesion-free case has no
    under-segmentation at all -- so this has to be handled rather than crash the run. The
    vendored file stays byte-identical to upstream, so the shape is absorbed here.
    """
    if mask is None or not mask.any():
        return [], 0
    try:
        result = simulate_scribble_from_label(mask.astype(np.uint8), strategy, seed)
    except Exception:
        return [], 0
    if not isinstance(result, (tuple, list)) or len(result) < 3:
        return [], 0
    coords, _, size = result[0], result[1], result[2]
    return (list(coords) if coords else []), int(size)


def next_scribble(prediction, ground_truth, strategy: str, seed: int):
    """One scribble on the biggest remaining error. Returns ``(kind, coords, length)``.

    ``kind`` is None when neither error can be drawn on, which is the signal to stop: there is
    nothing left for a user to correct. The fallbacks matter -- the longer error can still fail
    to yield coordinates, and silently drawing nothing would waste the round.
    """
    over_segmented = (prediction > 0) & (ground_truth == 0)
    under_segmented = (prediction == 0) & (ground_truth > 0)
    background, fp = safe_simulate(over_segmented, strategy, seed)
    foreground, fn = safe_simulate(under_segmented, strategy, seed)
    if fp == 0 and fn == 0:
        return None, [], 0
    if fp <= fn:
        return ("tumor", foreground, fn) if foreground else ("background", background, fp)
    return ("background", background, fp) if background else ("tumor", foreground, fn)


# --- scoring -----------------------------------------------------------------------------


def score(prediction, ground_truth, case: str, spacing=None) -> dict:
    """Every quantity needed to recompute any metric later, without re-running inference.

    Three groups are stored. The official evaluator's own fields (``dsc``, ``f1``, lesion
    counts, unmatched volumes in ml), with undefined values stored as null so those cases stay
    droppable. ``dice_loop``, the upstream loop's volumetric Dice, which scores 1.0 where the
    official scorer is undefined -- its AUC is what the loop integrates, so keeping it means
    that number can be produced from the JSON alone. And the voxel primitives, from which
    Dice, IoU, precision, recall and any confusion-matrix cell follow by arithmetic.
    """
    prediction = np.asarray(prediction).astype(np.uint8)
    ground_truth = np.asarray(ground_truth).astype(np.uint8)
    metrics = MetricEvaluator(overlap_threshold=OVERLAP_THRESHOLD, connectivity=CONNECTIVITY)(
        prediction, ground_truth, case, spacing=spacing
    )

    def defined(value):
        if value is None:
            return None
        value = float(value)
        return None if np.isnan(value) else value

    predicted = int(prediction.sum())
    annotated = int(ground_truth.sum())
    intersection = int(np.count_nonzero((prediction > 0) & (ground_truth > 0)))
    total = int(prediction.size)
    denominator = predicted + annotated

    return {
        "dsc": defined(metrics.get("dsc")),
        "f1": defined(metrics.get("f1")),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "fpv": defined(metrics.get("fpv")),
        "fnv": defined(metrics.get("fnv")),
        "dice_loop": 1.0 if denominator == 0 else 2.0 * intersection / denominator,
        "vox_inter": intersection,
        "vox_pred": predicted,
        "vox_gt": annotated,
        "vox_total": total,
        "vox_tn": total - predicted - annotated + intersection,
        "spacing": [float(x) for x in np.asarray(spacing).ravel()] if spacing is not None else None,
    }


# --- masks -------------------------------------------------------------------------------


def save_mask(path: str, mask, spacing) -> None:
    """Bit-pack the mask and compress it.

    A whole-body lesion mask is over 99.7% zeros, so this lands near 60 kB against 48 MB raw,
    and keeping every round's mask is what makes a later metric change free rather than another
    pass over the GPU.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        bits=np.packbits(np.asarray(mask, np.uint8).ravel()),
        shape=np.asarray(mask.shape, np.int64),
        spacing=np.asarray(spacing, np.float64),
    )


def load_mask(path: str):
    with np.load(path) as handle:
        shape = tuple(int(v) for v in handle["shape"])
        count = int(np.prod(shape))
        mask = np.unpackbits(handle["bits"])[:count].reshape(shape).astype(np.uint8)
        return mask, handle["spacing"]


# --- one case ----------------------------------------------------------------------------


def run_case(
    model,
    ct,
    pet,
    ground_truth,
    case: str,
    spacing,
    rounds: int = DEFAULT_ROUNDS,
    strategy: str = "centerline",
    sigma: float = 0.0,
    rng=None,
    device: str = "cuda",
    sw_batch_size: int = 1,
    mask_dir: str | None = None,
) -> list:
    """Run the loop for one case. ``ground_truth`` and ``spacing`` are the original grid.

    Returns one record per round. When no error can be drawn on, the last record is repeated to
    the end rather than truncated, so every case contributes the same number of rounds.
    """
    import random as _random

    rng = rng or _random.Random(0)
    prompts = Prompts()
    records = []

    for round_index in range(rounds + 1):
        prediction = predict_case(
            model,
            ct,
            pet,
            original_shape=ground_truth.shape,
            original_spacing=spacing,
            prompts=prompts,
            device=device,
            sw_batch_size=sw_batch_size,
            prompt_sigma=sigma,
        )
        mask = prediction.mask

        if mask_dir:
            save_mask(os.path.join(mask_dir, f"{case_id(case)}_r{round_index}.npz"), mask, spacing)

        record = score(mask, ground_truth, case, spacing)
        record.update(
            round=round_index,
            n_fg=len(prompts.foreground),
            n_bg=len(prompts.background),
        )
        records.append(record)
        if round_index == rounds:
            break

        drawn = rng.choice(STRATEGIES) if strategy == "random" else strategy
        kind, coords, length = next_scribble(mask, ground_truth, drawn, rng.randrange(2**31 - 1))
        record["next_strategy"], record["next_kind"], record["next_len"] = drawn, kind, length
        if kind is None or not coords:
            while len(records) <= rounds:  # nothing left to correct: hold the last score
                records.append(dict(records[-1], round=len(records)))
            break
        prompts.add(kind, coords)

    return records


# --- cohort ------------------------------------------------------------------------------


def aggregate(out_dir: str, rounds: int) -> dict:
    """Mean scores per round over every case JSON in ``out_dir``."""
    cases = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "case_*.json"))):
        with open(path) as handle:
            record = json.load(handle)
        cases[record["case"]] = record
    if not cases:
        return {}

    table = []
    for round_index in range(rounds + 1):
        rows = [
            next((r for r in c["rounds"] if r["round"] == round_index), None) for c in cases.values()
        ]
        rows = [r for r in rows if r]
        if not rows:
            continue
        dsc = [r["dsc"] for r in rows if r["dsc"] is not None]
        f1 = [r["f1"] for r in rows if r["f1"] is not None]
        fpv = [r["fpv"] for r in rows if r.get("fpv") is not None]
        fnv = [r["fnv"] for r in rows if r.get("fnv") is not None]
        table.append(
            {
                "round": round_index,
                "dsc": float(np.mean(dsc)) if dsc else float("nan"),
                "f1": float(np.mean(f1)) if f1 else float("nan"),
                "dice_loop": float(np.mean([r["dice_loop"] for r in rows])),
                "tp": sum(r["tp"] for r in rows),
                "fp": sum(r["fp"] for r in rows),
                "fn": sum(r["fn"] for r in rows),
                "fpv": float(np.mean(fpv)) if fpv else float("nan"),
                "fnv": float(np.mean(fnv)) if fnv else float("nan"),
                "scored_cases": len(dsc),
                "scribbles": float(np.mean([r["n_fg"] + r["n_bg"] for r in rows])),
            }
        )
    return {"cases": len(cases), "rounds": table}


def print_table(summary: dict, out_dir: str) -> None:
    if not summary:
        print("no results in", out_dir)
        return
    rows = summary["rounds"]
    print(f"\n{'=' * 96}")
    print(f"INTERACTIVE LOOP  --  {summary['cases']} cases  ({out_dir})")
    print(f"{'=' * 96}")
    print(
        f"{'round':<8}{'Dice':>10}{'dDice':>9}{'F1':>10}{'dF1':>9}"
        f"{'TP':>8}{'FP':>7}{'FN':>7}{'FPV ml':>9}{'FNV ml':>9}{'n':>6}{'scribbles':>11}"
    )
    print("-" * 96)
    base_dsc, base_f1 = rows[0]["dsc"], rows[0]["f1"]
    for row in rows:
        print(
            f"r{row['round']:<7}{row['dsc']:>10.4f}{row['dsc'] - base_dsc:>+9.4f}"
            f"{row['f1']:>10.4f}{row['f1'] - base_f1:>+9.4f}"
            f"{row['tp']:>8}{row['fp']:>7}{row['fn']:>7}"
            f"{row['fpv']:>9.2f}{row['fnv']:>9.2f}{row['scored_cases']:>6}"
            f"{row['scribbles']:>11.1f}"
        )
    print("-" * 96)
    print(f"  r0 is unguided; each later round adds one scribble")
    print("  n = cases with a defined Dice; lesion-free cases are excluded, as officially")


def rescore(out_dir: str, mask_dir: str, labels_dir: str, rounds: int) -> dict:
    """Recompute every metric from saved masks -- no model, no GPU, no inference.

    This is why the masks are kept: a changed metric definition costs a few CPU minutes here
    rather than another full pass over the cohort.
    """
    done = 0
    for path in sorted(glob.glob(os.path.join(out_dir, "case_*.json"))):
        with open(path) as handle:
            record = json.load(handle)
        identifier = os.path.basename(path)[len("case_") : -len(".json")]
        ground_truth = np.load(os.path.join(labels_dir, f"{record['case']}_orig.npy")).astype(np.uint8)
        for round_record in record["rounds"]:
            mask_path = os.path.join(mask_dir, f"{identifier}_r{round_record['round']}.npz")
            if not os.path.exists(mask_path):
                continue
            mask, spacing = load_mask(mask_path)
            round_record.update(score(mask, ground_truth, record["case"], spacing))
        with open(path, "w") as handle:
            json.dump(record, handle, indent=1)
        done += 1
        if done % 25 == 0:
            print(f"  rescored {done}", flush=True)
    print(f"rescored {done} cases from saved masks")
    return aggregate(out_dir, rounds)
