"""The preprocessing pipeline: one CSV row in, one training patch out.

There is no augmentation. A foreground-weighted window is sampled from the mask, the
intensity volumes and mask are read from it, the heatmap pair is loaded and scaled, and four
channels are stacked: ``[ct, pet, hm0, hm1]``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .sampling import PATCH_SIZE, weighted_random_fg_crop_window
from .volume_io import load_npy_crop, scale_heatmap


@dataclass
class PatchPipeline:
    """The preprocessing pipeline as a callable, configured by the fields below."""

    patch_size: Sequence[int] = PATCH_SIZE
    p_bg: float = 0.33
    small_boost: float = 0.0
    heatmap_divisor: float = 255.0

    def __call__(
        self, ct_path: str, pet_path: str, hm0_path: str, hm1_path: str, mask_path: str, cc=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one (4, *patch) float32 image and its (*patch,) float16 mask.

        Channels are [ct, pet, hm0, hm1]. The stack promotes the float16 ct/pet against the
        float32 heatmaps, so the image is float32; the mask never meets a float32 array and
        stays float16.
        """
        window = weighted_random_fg_crop_window(mask_path, self.patch_size, self.p_bg, self.small_boost, cc)
        ct = load_npy_crop(ct_path, window)
        pet = load_npy_crop(pet_path, window)
        mask = load_npy_crop(mask_path, window)
        hm0 = scale_heatmap(load_npy_crop(hm0_path, window), self.heatmap_divisor)
        hm1 = scale_heatmap(load_npy_crop(hm1_path, window), self.heatmap_divisor)

        image = np.stack([ct, pet, hm0, hm1], axis=0)
        return torch.from_numpy(image), torch.from_numpy(mask)
