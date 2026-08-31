"""Encoding live prompts onto the network's grid.

A prompt arrives as voxel coordinates on the **original** grid -- that is where a user points.
It has to reach the network on the target grid, quantised exactly as the stored training prompts
were, or the weights see a scale they were never trained on.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

from ..constants import PROMPT_QUANTISATION
from ..preprocessing.constants import TARGET_SPACING


class PromptEncoder:
    """Original-grid prompt coordinates to one target-grid channel.

    Coordinates are marked as 1.0, optionally blurred, resampled Linear onto the target grid,
    clipped to [0, 1] and quantised to 255 levels -- the same quantisation preparation applies to
    the stored prompts, so a live prompt and a training prompt land on one scale.

    ``sigma`` defaults to 0, which is an identity blur and is what the released model expects.
    It exists so a different encoding can be measured, not so it can be changed casually: any
    non-zero value puts prompts on a scale the weights were not trained against.
    """

    def __init__(self, original_shape, original_spacing, target_shape, sigma: float = 0.0) -> None:
        self.original_shape = np.asarray(original_shape, int)
        self.original_spacing = np.asarray(original_spacing, float)
        self.target_shape = np.asarray(target_shape, int)
        self.sigma = float(sigma)

    def __call__(self, coords) -> np.ndarray:
        """Encode a list of (x, y, z) coordinates. An empty list gives an all-zero channel."""
        volume = np.zeros(tuple(int(v) for v in self.original_shape), np.float32)
        for coord in np.asarray(coords, int).reshape(-1, 3) if len(coords) else ():
            if all(0 <= coord[i] < self.original_shape[i] for i in range(3)):
                volume[coord[0], coord[1], coord[2]] = 1.0
        if self.sigma > 0:
            # No renormalisation afterwards: the peak drops, which is the intended encoding.
            volume = gaussian_filter(volume, sigma=self.sigma)

        img = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(volume, (2, 1, 0))))
        img.SetSpacing([float(s) for s in self.original_spacing])
        f = sitk.ResampleImageFilter()
        f.SetSize([int(s) for s in self.target_shape])
        f.SetOutputSpacing([float(t) for t in TARGET_SPACING])
        f.SetOutputOrigin(img.GetOrigin())
        f.SetOutputDirection(img.GetDirection())
        f.SetInterpolator(sitk.sitkLinear)
        f.SetDefaultPixelValue(0.0)
        resampled = np.clip(np.transpose(sitk.GetArrayFromImage(f.Execute(img)), (2, 1, 0)), 0.0, 1.0)
        quantised = np.rint(resampled * PROMPT_QUANTISATION).astype(np.uint8)
        return quantised.astype(np.float32) / PROMPT_QUANTISATION
