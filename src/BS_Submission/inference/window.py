"""Gaussian-weighted sliding-window prediction over a whole volume."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def gaussian_weight(patch_size, sigma_scale: float = 0.125) -> torch.Tensor:
    """Per-voxel weight for stitching overlapping patches.

    Centre-weighted, so a voxel's prediction is dominated by the patches that saw it away from
    their edges. Floored at 1e-3 rather than 0 so the accumulator can never divide by zero.
    """
    tmp = np.zeros(patch_size, dtype=np.float32)
    tmp[tuple(p // 2 for p in patch_size)] = 1.0
    g = gaussian_filter(tmp, [p * sigma_scale for p in patch_size], mode="constant")
    g = np.clip(g / g.max(), 1e-3, None)
    return torch.from_numpy(g.astype(np.float32))


@torch.no_grad()
def sliding_window_inference(
    model,
    image: torch.Tensor,
    patch_size,
    num_classes: int = 2,
    overlap: float = 0.5,
    device: str = "cuda",
    sw_batch_size: int = 1,
) -> np.ndarray:
    """Predict over a whole volume patch by patch. ``image`` is ``(C, A0, A1, A2)``.

    ``model`` is one module or a **list** of them. Given several, each patch is run through every
    member and their softmax outputs are averaged before stitching -- a per-patch ensemble, which
    is what the submitted container does. Averaging probabilities rather than logits matters: the
    softmax is not linear, so the two are not the same combination.

    A single model divides the accumulator by exactly 1.0, so one-model results are unchanged.

    Returns the foreground probability map as CPU numpy, on the input's grid. A volume smaller
    than the patch is zero-padded and cropped back afterwards.

    ``sw_batch_size`` defaults to 1 deliberately, and there is no OOM-and-retry-smaller path: a
    CUDA out-of-memory does not reliably raise ``torch.cuda.OutOfMemoryError`` (it can surface as
    a generic ``RuntimeError``) and can leave the context unusable, so the retry would fail too.
    One patch at a time never over-commits; raise it only on a GPU known to have the headroom.
    """
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    ps = np.array(patch_size, int)
    orig = np.array(image.shape[1:], int)
    pad = np.maximum(ps - orig, 0)
    if pad.any():
        image = F.pad(image, [0, int(pad[2]), 0, int(pad[1]), 0, int(pad[0])])
    spatial = np.array(image.shape[1:], int)
    step = np.maximum((ps * (1 - overlap)).astype(int), 1)

    starts = []
    for d in range(3):
        axis = list(range(0, max(spatial[d] - ps[d], 0) + 1, step[d]))
        if axis[-1] != spatial[d] - ps[d]:
            axis.append(spatial[d] - ps[d])  # always cover the far edge
        starts.append(axis)

    coords = [(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]]
    weight = gaussian_weight(ps).to(device)
    total = torch.zeros((num_classes, *spatial), device=device)
    count = torch.zeros((1, *spatial), device=device)
    image = image.to(device)
    use_amp = str(device).startswith("cuda")

    for i in range(0, len(coords), sw_batch_size):
        chunk = coords[i : i + sw_batch_size]
        patches = torch.stack([image[:, z : z + ps[0], y : y + ps[1], x : x + ps[2]] for (z, y, x) in chunk], 0)
        with torch.autocast("cuda", enabled=use_amp):
            accumulated = None
            for member in models:
                logits = member(patches)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]  # deep supervision on: keep the full-resolution head
                member_probs = torch.softmax(logits, dim=1)
                accumulated = member_probs if accumulated is None else accumulated + member_probs
            probs = accumulated / len(models)
        probs = probs.float()
        for j, (z, y, x) in enumerate(chunk):
            total[:, z : z + ps[0], y : y + ps[1], x : x + ps[2]] += probs[j] * weight
            count[:, z : z + ps[0], y : y + ps[1], x : x + ps[2]] += weight

    averaged = (total / count)[:, : orig[0], : orig[1], : orig[2]]
    return averaged[1].cpu().numpy()
