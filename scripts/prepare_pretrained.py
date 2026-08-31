#!/usr/bin/env python
"""Turn published nnU-Net fold checkpoints into per-fold initialisation weights.

The upstream autoPET III checkpoints are trained on two input channels (CT and PET). This
network takes four: CT, PET, and two prompt heatmaps. The conversion keeps every trained
weight and widens the stem convolution from two input channels to four, initialising the two
new channels to **zero** — so at step zero the network computes exactly what the pretrained
two-channel model computed, and the prompt channels start contributing nothing.

The stem convolution appears several times in the state dict under aliased names (the decoder
holds a reference to the encoder, and each conv is registered twice), so every alias is widened.

Usage:
    python scripts/prepare_pretrained.py <nnunet-model-dir> <out-dir> [--folds 0 1 2 3 4]

where <nnunet-model-dir> contains fold_0/checkpoint_final.pth ... fold_4/checkpoint_final.pth.
"""

from __future__ import annotations

import argparse
import pathlib

import torch

STEM_MARKER = "stem.convs.0"
WEIGHTS_KEY = "network_weights"


def widen_stem(state: dict[str, torch.Tensor], in_channels: int) -> tuple[dict, list[str]]:
    """Zero-pad the input-channel dimension of every stem convolution weight."""
    widened = []
    out = {}
    for key, tensor in state.items():
        is_stem_conv = STEM_MARKER in key and key.endswith("weight") and tensor.ndim == 5
        if is_stem_conv and tensor.shape[1] < in_channels:
            pad = torch.zeros(
                (tensor.shape[0], in_channels - tensor.shape[1], *tensor.shape[2:]),
                dtype=tensor.dtype,
            )
            out[key] = torch.cat([tensor, pad], dim=1)
            widened.append(key)
        else:
            out[key] = tensor
    return out, widened


def convert(src: pathlib.Path, dst: pathlib.Path, in_channels: int) -> None:
    """Read one fold checkpoint and write the weights-only initialisation file."""
    checkpoint = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint.get(WEIGHTS_KEY, checkpoint)
    out, widened = widen_stem(state, in_channels)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    size_mb = dst.stat().st_size / 1e6
    print(f"  {src.parent.name} -> {dst.name}  {len(out)} tensors, {size_mb:.0f} MB")
    print(f"      widened {len(widened)} stem alias(es) to {in_channels} input channels")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=pathlib.Path, help="directory of fold_<n> subdirectories")
    parser.add_argument("out_dir", type=pathlib.Path, help="where to write pretrained_fold<n>.pt")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--checkpoint-name", default="checkpoint_final.pth")
    args = parser.parse_args()

    for fold in args.folds:
        src = args.model_dir / f"fold_{fold}" / args.checkpoint_name
        if not src.exists():
            raise SystemExit(f"missing checkpoint: {src}")
        convert(src, args.out_dir / f"pretrained_fold{fold}.pt", args.in_channels)


if __name__ == "__main__":
    main()
