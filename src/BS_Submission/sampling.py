"""Foreground-weighted crop-window sampling for whole-body PET/CT patches.

A window is drawn from the segmentation mask: uniformly at random for a case with no
lesion, and otherwise either a deliberately lesion-free background window (with
probability ``p_bg``) or a window containing a voxel of a lesion chosen with a
``1 / size ** small_boost`` weighting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from skimage.measure import label

PATCH_SIZE = (192, 192, 192)


def _rand_window(spatial: np.ndarray, ps: np.ndarray) -> tuple:
    """Uniform-random window, start clamped so the patch fits."""
    start = [np.random.randint(0, max(int(spatial[i] - ps[i]), 0) + 1) for i in range(len(ps))]
    return tuple(slice(start[i], start[i] + int(ps[i])) for i in range(len(ps)))


def _window_around(v: Sequence[int], spatial: np.ndarray, ps: np.ndarray) -> tuple:
    """Random window guaranteed to contain voxel `v`."""
    start = [
        np.random.randint(
            max(0, int(v[i] - ps[i] + 1)),
            min(int(v[i]), int(spatial[i] - ps[i])) + 1,
        )
        for i in range(len(ps))
    ]
    return tuple(slice(start[i], start[i] + int(ps[i])) for i in range(len(ps)))


def _tumor_free_window(mask, spatial: np.ndarray, ps: np.ndarray, max_tries: int = 20) -> tuple:
    """Background window: first tumor-free window found, else the least-tumor one seen."""
    best, best_fg = None, None
    for _ in range(max_tries):
        s = _rand_window(spatial, ps)
        fg = int(np.asarray(mask[s]).sum())
        if fg == 0:
            return s
        if best_fg is None or fg < best_fg:
            best, best_fg = s, fg
    return best


def _lesion_weights(sizes: np.ndarray, gamma: float) -> np.ndarray:
    """Per-lesion probability ~ 1 / size ** gamma; gamma 0 is per-lesion uniform."""
    w = 1.0 / np.power(np.asarray(sizes, dtype=np.float64), gamma)
    return w / w.sum()


def _fg_window_from_cc(seeds, offsets, sizes, spatial, ps, gamma):
    """Fast path: lesion by weight, voxel from its precomputed coord list, then a window."""
    n = int(np.asarray(sizes).shape[0])
    if n == 0:
        return None
    ci = int(np.random.choice(n, p=_lesion_weights(sizes, gamma)))
    a, b = int(offsets[ci]), int(offsets[ci + 1])
    v = seeds[a + np.random.randint(b - a)]
    return _window_around([int(v[i]) for i in range(len(ps))], spatial, ps)


def _fg_window_from_mask(mask, spatial, ps, gamma):
    """Slow path: relabel with 26-connectivity, pick a lesion by weight, then a voxel."""
    lbl, n = label(np.asarray(mask), return_num=True)
    if n == 0:
        return None
    sizes = np.bincount(lbl.ravel(), minlength=n + 1)[1:]
    comp = int(np.random.choice(n, p=_lesion_weights(sizes, gamma))) + 1
    idx = np.flatnonzero(lbl == comp)
    v = np.unravel_index(idx[np.random.randint(idx.size)], np.asarray(mask).shape)
    return _window_around(v, spatial, ps)


def _load_cc(cc: str | Mapping | None) -> dict | None:
    """Normalise the optional connected-component bundle into seeds/offsets/sizes."""
    if cc is None:
        return None
    if isinstance(cc, str):
        with np.load(cc) as z:
            return {"sizes": z["sizes"], "offsets": z["offsets"], "seeds": z["seeds"]}
    return {"sizes": cc["sizes"], "offsets": cc["offsets"], "seeds": cc["seeds"]}


def weighted_random_fg_crop_window(
    mask, patch_size=PATCH_SIZE, p_bg: float = 0.33, small_boost: float = 0.0, cc=None
) -> np.ndarray:
    """Sample a crop window as an (ndim, 2) array of [start, stop] bounds per axis.

    The JSON supplies only `patch_size`, so `p_bg` and `small_boost` take the reference implementation's defaults:
    0.33 (a third of tumor cases get a background window) and 0.0 (per-lesion uniform,
    i.e. no small-lesion boost).
    """
    cc = _load_cc(cc)
    if isinstance(mask, str):
        mask = np.load(mask, mmap_mode="r")
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Expected path, Tensor or ND-Array but found {type(mask)}")

    ps = np.asarray(patch_size, dtype=int)
    spatial = np.asarray(mask.shape, dtype=int)

    if int(np.asarray(mask).max()) == 0:
        window = _rand_window(spatial, ps)
    else:
        window = None
        if np.random.rand() < p_bg:
            window = _tumor_free_window(mask, spatial, ps)
        if window is None:
            if cc is not None:
                window = _fg_window_from_cc(cc["seeds"], cc["offsets"], cc["sizes"], spatial, ps, small_boost)
            else:
                window = _fg_window_from_mask(mask, spatial, ps, small_boost)
        if window is None:
            window = _rand_window(spatial, ps)

    return np.array([[window[i].start, window[i].stop] for i in range(len(ps))], dtype=int)
