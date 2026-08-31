"""Command-line entry point for dataset preparation."""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from .constants import PET_MEAN, PET_STD, REF_ORDER
from .layout import Layout
from .stages import run_prep, run_reference, run_segmentation

STAGES = ("seg", "ref", "prep", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bs-preprocess",
        description="Prepare raw PET/CT and prompt heatmaps into model-ready arrays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "stages: seg (organ masks, GPU) -> ref (PET reference + cohort stats, CPU) -> "
            "prep (write channels, CPU). Supply cached organ masks to skip seg."
        ),
    )
    io = parser.add_argument_group("paths")
    io.add_argument("--dataset-dir", required=True, help="raw dataset with imagesTr/ and labelsTr/")
    io.add_argument("--heatmaps-dir", default="", help="published prompt heatmaps (needed by prep)")
    io.add_argument("--out-dir", required=True, help="destination for the prepared arrays")

    stage = parser.add_argument_group("stage")
    stage.add_argument("--stage", choices=STAGES, default="all")
    stage.add_argument(
        "--ref-organ",
        choices=sorted(REF_ORDER),
        default="aorta",
        help="primary PET reference region. The released dataset and weights use the aortic "
        "blood pool: it is stable across tracers, whereas liver uptake differs sharply "
        "between FDG and PSMA. Choosing liver changes the PET axis and invalidates the "
        "released checkpoints",
    )
    stage.add_argument(
        "--pet-only",
        action="store_true",
        help="prep writes only pet/, reusing CT, labels and heatmaps already prepared",
    )

    compute = parser.add_argument_group("compute")
    compute.add_argument("--device", default="gpu:0", help="device for the seg stage")
    compute.add_argument("--num-shards", type=int, default=1, help="split cases across processes")
    compute.add_argument("--shard-index", type=int, default=0)
    compute.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 8))
    compute.add_argument("--limit", type=int, default=None, help="first N cases, for a smoke test")

    stats = parser.add_argument_group("normalisation")
    stats.add_argument(
        "--ref-json",
        default="",
        help="per-case PET references to use instead of this run's own. Pass "
        "assets/preprocessing/ref.json to prepare the released cohort without re-running the "
        "ref stage, which needs the raw PET for every case",
    )
    stats.add_argument(
        "--pet-stats",
        default="",
        help="npz with mean/std overriding the committed cohort PET statistics",
    )
    return parser


def layout_from(args: argparse.Namespace) -> Layout:
    """Build the layout, carrying the path overrides through.

    Separated out so the wiring is testable: without the two ``_override`` arguments the flags
    parse and are then silently ignored, and ``--ref-json`` -- the documented way to prepare the
    released cohort without re-running ``ref`` -- fails looking for a ``ref.json`` in the output
    directory that nothing has written.
    """
    return Layout(
        dataset_dir=args.dataset_dir,
        heatmaps_dir=args.heatmaps_dir,
        out_dir=args.out_dir,
        ref_json_override=args.ref_json,
        stats_npz_override=args.pet_stats,
    )


def pet_statistics(args: argparse.Namespace, layout: Layout) -> tuple[float, float]:
    """Resolve the PET global-z statistics: an explicit override, this run's, or the committed pair.

    The committed values are the fallback rather than the last resort, because they are what the
    released checkpoints were trained against. Recomputing them for a different cohort is a
    deliberate act, so it has to be asked for.
    """
    source = args.pet_stats or (layout.stats_npz if os.path.exists(layout.stats_npz) else "")
    if source:
        stats = np.load(source)
        return float(stats["mean"]), float(stats["std"])
    return PET_MEAN, PET_STD


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    layout = layout_from(args)
    os.makedirs(layout.out_dir, exist_ok=True)

    cases = sorted(
        os.path.basename(p)[: -len("_0000.nii.gz")] for p in glob.glob(os.path.join(layout.images_dir, "*_0000.nii.gz"))
    )
    if not cases:
        raise SystemExit(f"no cases found in {layout.images_dir}")
    if args.limit:
        cases = cases[: args.limit]

    order = REF_ORDER[args.ref_organ]
    print(f"{len(cases)} cases | stage={args.stage} | reference={args.ref_organ} {order}")

    if args.stage in ("seg", "all"):
        shard = cases[args.shard_index :: args.num_shards] if args.num_shards > 1 else cases
        print(f"[seg] {len(shard)} cases on {args.device}", flush=True)
        run_segmentation(shard, layout, args.device)
    if args.stage in ("ref", "all"):
        run_reference(cases, layout, args.workers, order)
    if args.stage in ("prep", "all"):
        if not args.pet_only and not layout.heatmaps_dir:
            raise SystemExit("prep needs --heatmaps-dir (or --pet-only to skip the heatmaps)")
        mean, std = pet_statistics(args, layout)
        run_prep(cases, layout, args.workers, mean, std, args.pet_only)


if __name__ == "__main__":
    main()
