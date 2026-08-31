"""Segment one case from its CT and PET, without a prepared dataset."""

from __future__ import annotations

import argparse
import json
import os

import SimpleITK as sitk

from ..constants import SEGMENTATION_THRESHOLD
from ..preprocessing.constants import REF_ORDER, SHIPPED_REF_ORGAN
from .cli import CHECKPOINT_TEMPLATE, N_FOLDS
from .predict import Prompts
from .standalone import segment_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bs-segment",
        description="Predict a lesion mask for one case, straight from its CT and PET.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="example: bs-segment --ct ct.nii.gz --pet pet.nii.gz --fold 0 --out mask.nii.gz",
    )
    scan = parser.add_argument_group("input")
    scan.add_argument("--ct", required=True, help="CT volume, NIfTI")
    scan.add_argument("--pet", required=True, help="PET volume in SUV, NIfTI, same case")
    scan.add_argument(
        "--clicks",
        default="",
        help="JSON of click coordinates on the CT grid. Both shapes are read: the challenge's "
        'lesion-clicks.json ({"points": [{"point": [x,y,z], "name": "tumor"}, ...]}) and the '
        'plain {"tumor": [[x,y,z], ...], "background": [...]}. Omitted means an unguided pass',
    )

    mdl = parser.add_argument_group("model")
    mdl.add_argument(
        "--fold",
        type=int,
        nargs="+",
        choices=range(N_FOLDS),
        metavar=f"{{0..{N_FOLDS - 1}}}",
        help="fold weights to load. Give several -- `--fold 0 1 2 3 4` -- to ensemble them, "
        "which is what the submitted container does by default",
    )
    mdl.add_argument("--model-dir", default="assets/model", help="directory of per-fold weights")
    mdl.add_argument(
        "--checkpoint", default=[], nargs="+", help="weights to use; overrides --fold"
    )
    mdl.add_argument("--arch-path", default="", help="dotted path to the architecture class")

    norm = parser.add_argument_group("PET normalisation")
    norm.add_argument(
        "--ref",
        type=float,
        default=None,
        help="the scan's reference SUV. Supplying it skips organ segmentation entirely; "
        "otherwise TotalSegmentator is run on the CT to find the reference organ",
    )
    norm.add_argument(
        "--ref-organ",
        default=SHIPPED_REF_ORGAN,
        choices=sorted(REF_ORDER),
        help="reference region. The released weights use the aortic blood pool; changing it "
        "moves the PET axis away from what they were trained on",
    )
    norm.add_argument(
        "--organ-dir",
        default="",
        help="where to keep the organ masks. Reused if already present, so a second run on the "
        "same case skips the segmentation",
    )

    out = parser.add_argument_group("output")
    out.add_argument("--out", required=True, help="destination for the mask, NIfTI")
    out.add_argument("--save-probability", default="", help="also write the probability map here")

    run = parser.add_argument_group("run")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--threshold", type=float, default=SEGMENTATION_THRESHOLD)
    run.add_argument("--sw-batch", type=int, default=1, help="patches per forward pass")
    return parser


def resolve_checkpoints(args: argparse.Namespace) -> list[str]:
    """Every checkpoint to load, in order. More than one means a per-patch ensemble."""
    paths = list(args.checkpoint)
    if not paths and args.fold:
        paths = [
            os.path.join(args.model_dir, CHECKPOINT_TEMPLATE.format(fold=f)) for f in args.fold
        ]
    if not paths:
        raise SystemExit("no weights: pass --fold or --checkpoint")
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("weights not found: " + ", ".join(missing))
    return paths


def load_clicks(path: str) -> Prompts:
    """Click coordinates from JSON, in either shape the challenge uses.

    The platform hands over a ``lesion-clicks.json`` listing every point with a name; the
    simulator and the interactive loop use a plain tumor/background mapping. Both are read here
    so a file from either source works without conversion.
    """
    prompts = Prompts()
    if not path:
        return prompts
    with open(path) as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "points" in data:  # Grand Challenge "Multiple points"
        for entry in data["points"]:
            name = entry.get("name")
            if name in ("tumor", "background") and entry.get("point") is not None:
                prompts.add(name, [entry["point"]])
        return prompts
    for key in ("tumor", "foreground"):
        prompts.add("tumor", data.get(key, []))
    prompts.add("background", data.get("background", []))
    return prompts


def main() -> None:
    args = build_parser().parse_args()
    checkpoints = resolve_checkpoints(args)
    for name, path in (("CT", args.ct), ("PET", args.pet)):
        if not os.path.exists(path):
            raise SystemExit(f"{name} not found: {path}")

    prompts = load_clicks(args.clicks)
    print(f"ct  {args.ct}\npet {args.pet}")
    for path in checkpoints:
        print(f"weights {path}")
    print(
        f"device {args.device} | threshold {args.threshold} | "
        f"{len(checkpoints)}-fold {'ensemble' if len(checkpoints) > 1 else 'single model'} | "
        f"clicks {len(prompts.foreground)} tumor / {len(prompts.background)} background",
        flush=True,
    )

    import hashlib

    from ..config import Config
    from ..model import build_model

    def build(path):
        extra = {"arch_path": args.arch_path} if args.arch_path else {}
        return build_model(Config(checkpoint_path=path, compile_model=False, **extra))

    models = [build(path).to(args.device).eval() for path in checkpoints]

    if len(models) > 1:
        # Checksum the loaded parameters, not the files: copying one checkpoint under five names
        # ensembles perfectly happily and returns exactly that one model, silently.
        digests = [
            hashlib.md5(
                b"".join(v.detach().cpu().numpy().tobytes() for v in m.state_dict().values())
            ).hexdigest()[:8]
            for m in models
        ]
        print("folds: " + ", ".join(digests), flush=True)
        if len(set(digests)) == 1:
            print(
                "WARNING: every fold has identical weights, so this is not an ensemble -- "
                "averaging N copies of one model returns that model.",
                flush=True,
            )

    model = models if len(models) > 1 else models[0]

    def announce(ref, source, cv, frac):
        note = "" if source in (args.ref_organ, "given") else "  <- fell back, not the trained axis"
        print(f"PET reference {ref:.4f} from {source}{note}", flush=True)

    mask, probability = segment_case(
        args.ct,
        args.pet,
        model,
        ref=args.ref,
        ref_organ=args.ref_organ,
        organ_dir=args.organ_dir,
        device=args.device,
        threshold=args.threshold,
        sw_batch_size=args.sw_batch,
        prompts=prompts,
        on_reference=announce,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sitk.WriteImage(mask, args.out, True)
    import numpy as np

    voxels = int(np.count_nonzero(sitk.GetArrayFromImage(mask)))
    print(f"wrote {args.out} | lesion voxels {voxels}", flush=True)
    if args.save_probability:
        sitk.WriteImage(probability, args.save_probability, True)
        print(f"wrote {args.save_probability}", flush=True)


if __name__ == "__main__":
    main()
