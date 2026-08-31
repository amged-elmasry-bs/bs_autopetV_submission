"""Score one predicted mask against its annotation, with the official evaluator."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .interactive import OVERLAP_THRESHOLD, CONNECTIVITY, score


def load_mask(path: str):
    """A mask as an (x, y, z) array, plus its voxel spacing where the format carries one.

    NIfTI and ``.npy`` are both accepted because they are what the two halves of this repository
    produce: ``bs-segment`` writes NIfTI carrying the scan's geometry, while the prepared cohort
    stores ``.npy`` with the geometry held separately in ``meta.json``.
    """
    if path.endswith(".npy"):
        return np.load(path), None
    import SimpleITK as sitk

    from ..preprocessing.geometry import to_xyz

    image = sitk.ReadImage(path)
    return to_xyz(image), [float(s) for s in image.GetSpacing()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bs-score",
        description="Score a predicted lesion mask against its annotation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="example: bs-score --prediction mask.nii.gz --reference labelsTr/case.nii.gz",
    )
    parser.add_argument("--prediction", required=True, help="predicted mask, NIfTI or .npy")
    parser.add_argument("--reference", required=True, help="annotation, NIfTI or .npy")
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="voxel spacing in mm. Taken from the NIfTI header when available; required with "
        "two .npy inputs, since the volume metrics are reported in ml and cannot be computed "
        "without it",
    )
    parser.add_argument("--case", default="", help="name recorded in the output")
    parser.add_argument("--json", default="", help="write the full record here")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name, path in (("prediction", args.prediction), ("reference", args.reference)):
        if not os.path.exists(path):
            raise SystemExit(f"{name} not found: {path}")

    prediction, pred_spacing = load_mask(args.prediction)
    reference, ref_spacing = load_mask(args.reference)
    if prediction.shape != reference.shape:
        raise SystemExit(
            f"shape mismatch: prediction {prediction.shape} vs reference {reference.shape}.\n"
            "  Both must be on the same grid. bs-segment writes on the CT's original grid, which "
            "is where the annotation lives; the prepared labels/<case>.npy are on the target grid "
            "instead -- compare against labels/<case>_orig.npy."
        )

    spacing = args.spacing or ref_spacing or pred_spacing
    if spacing is None:
        print("no spacing available: volume metrics (FPV/FNV) will be omitted", flush=True)

    case = args.case or os.path.basename(args.prediction).split(".")[0]
    record = score(prediction, reference, case, spacing)

    def shown(value, digits=4):
        return "undefined" if value is None else f"{value:.{digits}f}"

    print(f"case      {case}")
    print(f"shape     {tuple(int(s) for s in prediction.shape)}"
          + (f"   spacing {[round(s, 4) for s in spacing]} mm" if spacing else ""))
    print(f"voxels    predicted {record['vox_pred']}  annotated {record['vox_gt']}  "
          f"overlap {record['vox_inter']}")
    print()
    print(f"  Dice          {shown(record['dsc'])}")
    print(f"  F1            {shown(record['f1'])}   (lesion detection, IoU>{OVERLAP_THRESHOLD}, "
          f"connectivity {CONNECTIVITY})")
    print(f"  lesions       TP {record['tp']}  FP {record['fp']}  FN {record['fn']}")
    print(f"  unmatched     FPV {shown(record['fpv'], 2)} ml   FNV {shown(record['fnv'], 2)} ml")
    if record["dsc"] is None:
        print("\n  Dice and F1 are undefined because the annotation holds no lesions -- there is "
              "nothing\n  to detect, so the official evaluation excludes such a case rather than "
              "scoring it 0 or 1.")
        print(f"  The loop's own volumetric Dice, which does define it, is {record['dice_loop']:.4f}.")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump({"case": case, "prediction": args.prediction,
                       "reference": args.reference, **record}, handle, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
