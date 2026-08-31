"""Command-line entry point for prediction.

``--fold`` picks both halves of an evaluation at once: the fold's validation split *and* the
model trained on that fold's training split. Pairing them by hand is how a model ends up
evaluated on cases it was trained on, so the default makes the correct pairing the easy one.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from ..constants import SEGMENTATION_THRESHOLD
from ..model import build_model
from .predict import Prompts, predict_case

N_FOLDS = 5
CHECKPOINT_TEMPLATE = "fold{fold}.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bs-predict",
        description="Predict lesion masks for a fold's validation split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="example: bs-predict --fold 0 --data-root /data/prepared --out-dir runs/pred0",
    )
    sel = parser.add_argument_group("what to run")
    sel.add_argument(
        "--fold",
        type=int,
        choices=range(N_FOLDS),
        metavar=f"{{0..{N_FOLDS - 1}}}",
        help="fold to evaluate: selects splits/fold_<n>_val.csv and the model trained on that "
        "fold, so the model never sees its own training cases",
    )
    sel.add_argument("--data-root", required=True, help="prepared arrays: ct/ pet/ meta.json")
    sel.add_argument("--splits-dir", default="splits", help="directory holding the split CSVs")
    sel.add_argument("--val-csv", default="", help="validation split; overrides --fold")
    sel.add_argument("--cases", nargs="+", default=None, help="predict these cases only")
    sel.add_argument("--limit", type=int, default=None, help="first N cases only")

    mdl = parser.add_argument_group("model")
    mdl.add_argument("--model-dir", default="assets/model", help="directory of per-fold weights")
    mdl.add_argument("--checkpoint", default="", help="weights to use; overrides --fold")
    mdl.add_argument("--arch-path", default="", help="dotted path to the architecture class")

    out = parser.add_argument_group("output")
    out.add_argument("--out-dir", required=True, help="destination for predicted masks")
    out.add_argument("--save-probability", action="store_true", help="also write the probability")

    run = parser.add_argument_group("run")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--threshold", type=float, default=SEGMENTATION_THRESHOLD)
    run.add_argument("--sw-batch", type=int, default=1, help="patches per forward pass")
    return parser


def resolve(args: argparse.Namespace) -> tuple[str, str]:
    """Fill the validation split and the checkpoint in from ``--fold`` where not given."""
    val_csv, checkpoint = args.val_csv, args.checkpoint
    if args.fold is not None:
        val_csv = val_csv or os.path.join(args.splits_dir, f"fold_{args.fold}_val.csv")
        checkpoint = checkpoint or os.path.join(args.model_dir, CHECKPOINT_TEMPLATE.format(fold=args.fold))
    if not val_csv:
        raise SystemExit("no validation split: pass --fold or --val-csv")
    if not checkpoint:
        raise SystemExit("no weights: pass --fold or --checkpoint")
    for what, path in (("validation split", val_csv), ("weights", checkpoint)):
        if not os.path.exists(path):
            extra = ""
            if args.fold is not None and what == "weights":
                extra = (
                    f"\n  Per-fold weights are resolved as <--model-dir>/"
                    f"{CHECKPOINT_TEMPLATE}; see assets/model/README.md."
                )
            raise SystemExit(f"{what} not found: {path}{extra}")
    return val_csv, checkpoint


def cases_from(val_csv: str) -> list[str]:
    """Case identifiers in a split, taken from the mask column's basename."""
    with open(val_csv, newline="") as handle:
        return [os.path.basename(row["mask"]).replace(".npy", "") for row in csv.DictReader(handle)]


def main() -> None:
    args = build_parser().parse_args()
    val_csv, checkpoint = resolve(args)

    with open(os.path.join(args.data_root, "meta.json")) as handle:
        meta = json.load(handle)

    cases = args.cases or cases_from(val_csv)
    cases = [c for c in cases if c in meta]
    if args.limit:
        cases = cases[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"fold {args.fold} | split {val_csv} | weights {checkpoint}")
    print(f"{len(cases)} cases | device {args.device} | threshold {args.threshold}", flush=True)

    from ..config import Config

    cfg = Config(checkpoint_path=checkpoint, compile_model=False)
    if args.arch_path:
        cfg = Config(checkpoint_path=checkpoint, compile_model=False, arch_path=args.arch_path)
    model = build_model(cfg).to(args.device).eval()

    for i, case in enumerate(cases, 1):
        ct = np.asarray(np.load(os.path.join(args.data_root, "ct", f"{case}_0000.npy")), np.float32)
        pet = np.asarray(np.load(os.path.join(args.data_root, "pet", f"{case}_0001.npy")), np.float32)
        result = predict_case(
            model,
            ct,
            pet,
            original_shape=meta[case]["shape"],
            original_spacing=meta[case]["spacing"],
            prompts=Prompts(),
            device=args.device,
            threshold=args.threshold,
            sw_batch_size=args.sw_batch,
        )
        np.save(os.path.join(args.out_dir, f"{case}.npy"), result.mask)
        if args.save_probability:
            np.save(os.path.join(args.out_dir, f"{case}_prob.npy"), result.probability.astype("float16"))
        print(f"[{i}/{len(cases)}] {case[:46]:<48} lesion voxels {int(result.mask.sum())}", flush=True)


if __name__ == "__main__":
    main()
