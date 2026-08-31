"""Score a fold under the challenge's interactive correction protocol."""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np

from ..inference.cli import CHECKPOINT_TEMPLATE, N_FOLDS, cases_from, resolve
from .interactive import (
    DEFAULT_ROUNDS,
    STRATEGIES,
    aggregate,
    case_id,
    print_table,
    rescore,
    run_case,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bs-evaluate",
        description="Score a fold under the challenge's interactive correction protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="example: bs-evaluate --fold 0 --data-root /data/prepared --out-dir runs/eval0",
    )
    sel = parser.add_argument_group("what to run")
    sel.add_argument(
        "--fold",
        type=int,
        choices=range(N_FOLDS),
        metavar=f"{{0..{N_FOLDS - 1}}}",
        help="fold to evaluate: selects splits/fold_<n>_val.csv and the model trained on that "
        "fold, so the model never scores its own training cases",
    )
    sel.add_argument("--data-root", required=True, help="prepared arrays: ct/ pet/ labels/ meta.json")
    sel.add_argument("--splits-dir", default="splits", help="directory holding the split CSVs")
    sel.add_argument("--val-csv", default="", help="validation split; overrides --fold")
    sel.add_argument("--cases", nargs="+", default=None, help="evaluate these cases only")
    sel.add_argument("--limit", type=int, default=0, help="first N cases only")

    mdl = parser.add_argument_group("model")
    mdl.add_argument("--model-dir", default="assets/model", help="directory of per-fold weights")
    mdl.add_argument("--checkpoint", default="", help="weights to use; overrides --fold")
    mdl.add_argument("--arch-path", default="", help="dotted path to the architecture class")

    pro = parser.add_argument_group("protocol")
    pro.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help="scribbles drawn after round 0, so passes = rounds + 1",
    )
    pro.add_argument(
        "--strategy",
        default="centerline",
        choices=list(STRATEGIES),
        help="how a scribble is drawn; 'random' draws one of the three per round",
    )
    pro.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        help="scribble blur. 0 is the encoding the weights were trained against; any other "
        "value puts prompts on a scale the model has not seen",
    )
    pro.add_argument("--seed", type=int, default=13)

    out = parser.add_argument_group("output")
    out.add_argument("--out-dir", required=True, help="per-case result JSONs")
    out.add_argument("--mask-dir", default="", help="saved masks (default: <out-dir>/masks)")
    out.add_argument(
        "--no-save-masks",
        action="store_true",
        help="score only, without persisting masks. Masks are ~60 kB per round and make a "
        "later metric change free, so keeping them is strongly preferred",
    )
    out.add_argument(
        "--rescore",
        action="store_true",
        help="recompute metrics from saved masks; needs no model and no GPU",
    )
    out.add_argument("--aggregate", action="store_true", help="print the table and exit")

    run = parser.add_argument_group("run")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--sw-batch", type=int, default=1, help="patches per forward pass")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    mask_dir = None if args.no_save_masks else (args.mask_dir or os.path.join(args.out_dir, "masks"))
    labels_dir = os.path.join(args.data_root, "labels")

    if args.aggregate:
        print_table(aggregate(args.out_dir, args.rounds), args.out_dir)
        return
    if args.rescore:
        summary = rescore(
            args.out_dir, mask_dir or os.path.join(args.out_dir, "masks"), labels_dir, args.rounds
        )
        print_table(summary, args.out_dir)
        return

    val_csv, checkpoint = resolve(args)
    with open(os.path.join(args.data_root, "meta.json")) as handle:
        meta = json.load(handle)

    cases = args.cases or cases_from(val_csv)
    cases = [c for c in cases if c in meta]
    if args.limit:
        cases = cases[: args.limit]
    if mask_dir:
        os.makedirs(mask_dir, exist_ok=True)

    print(f"[eval] ckpt   {checkpoint}")
    print(f"[eval] data   {args.data_root}")
    print(f"[eval] masks  {mask_dir or 'NOT SAVED'}")
    print(f"[eval] fold {args.fold} | {len(cases)} val cases | device {args.device}")
    print(
        f"[eval] rounds {args.rounds} (+r0) | strategy {args.strategy} | sigma {args.sigma}",
        flush=True,
    )

    import torch

    from ..config import Config
    from ..model import build_model

    cfg = Config(checkpoint_path=checkpoint, compile_model=False)
    if args.arch_path:
        cfg = Config(checkpoint_path=checkpoint, compile_model=False, arch_path=args.arch_path)
    model = build_model(cfg).to(args.device).eval()

    rng = random.Random(args.seed)
    started = time.perf_counter()
    done = 0

    with torch.no_grad():
        for index, case in enumerate(cases, 1):
            destination = os.path.join(args.out_dir, f"case_{case_id(case)}.json")
            if os.path.exists(destination):
                continue  # already scored: resume rather than repeat
            case_started = time.perf_counter()
            ct = np.asarray(np.load(os.path.join(args.data_root, "ct", f"{case}_0000.npy")), np.float32)
            pet = np.asarray(np.load(os.path.join(args.data_root, "pet", f"{case}_0001.npy")), np.float32)
            ground_truth = np.load(os.path.join(labels_dir, f"{case}_orig.npy")).astype(np.uint8)
            spacing = np.asarray(meta[case]["spacing"], np.float64)
            try:
                records = run_case(
                    model,
                    ct,
                    pet,
                    ground_truth,
                    case=case,
                    spacing=spacing,
                    rounds=args.rounds,
                    strategy=args.strategy,
                    sigma=args.sigma,
                    rng=rng,
                    device=args.device,
                    sw_batch_size=args.sw_batch,
                    mask_dir=mask_dir,
                )
            except Exception as error:  # one bad case must not end the run
                print(f"[{index}/{len(cases)}] ERROR {case[:40]}: {error!r:.120}", flush=True)
                continue
            with open(destination, "w") as handle:
                json.dump(
                    {
                        "case": case,
                        "fold": args.fold,
                        "checkpoint": checkpoint,
                        "strategy": args.strategy,
                        "sigma": args.sigma,
                        "rounds": records,
                    },
                    handle,
                    indent=1,
                )
            done += 1

            def shown(value):
                return float("nan") if value is None else value

            elapsed = time.perf_counter() - started
            eta = elapsed / max(done, 1) * (len(cases) - index) / 60
            print(
                f"[{index}/{len(cases)}] {case[:38]:40s} "
                f"r0 {shown(records[0]['dsc']):.4f} -> r{args.rounds} {shown(records[-1]['dsc']):.4f} "
                f"| {time.perf_counter() - case_started:.0f}s | eta {eta:.0f} min",
                flush=True,
            )

    print_table(aggregate(args.out_dir, args.rounds), args.out_dir)


if __name__ == "__main__":
    main()
