"""Memory-mapped patch reads and heatmap scaling.

Volumes stay on disk; only the sampled crop window is read. Heatmap channels are stored
uint8-encoded and scaled back into [0, 1] before they meet the intensity channels."""

from __future__ import annotations

import numpy as np
import torch


def _crop_slices(window) -> tuple:
    """Rebuild slices from an (ndim, 2) bounds array, accepting the flat string form."""
    if isinstance(window, str):
        window = np.fromstring(window, sep=" ", dtype=int).reshape(-1, 2)
    window = np.asarray(window, dtype=int)
    if window.ndim != 2 or window.shape[1] != 2:
        raise ValueError(f"Expected window of shape (ndim, 2) but found {window.shape}")
    return tuple(slice(int(start), int(stop)) for start, stop in window)


def load_npy_crop(path: str, window) -> np.ndarray:
    """Memory-map a .npy and read only `window` from disk, as float16."""
    return np.load(path, mmap_mode="r")[_crop_slices(window)].astype(np.float16)


def scale_heatmap(sample, divisor: float = 255.0):
    """Scale a uint8-encoded heatmap (0..255) into [0, 1] as float32."""
    if isinstance(sample, np.ndarray):
        return sample.astype(np.float32) / np.float32(divisor)
    if isinstance(sample, torch.Tensor):
        return sample.float() / divisor
    raise TypeError(f"Expected Numpy array or Torch Tensor but found {type(sample)}")
