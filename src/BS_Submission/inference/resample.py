"""Reversing the forward resample, to put a prediction back on the annotation grid."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from ..preprocessing.constants import TARGET_SPACING


def resample_to_original(arr, src_spacing, dst_spacing, dst_size, interpolator) -> np.ndarray:
    """Physical-space resample of an (x, y, z) array from the target grid back to the original.

    The exact inverse of the forward resample, because the forward pass kept origin and direction
    identical -- so identity geometry here reproduces it rather than approximating it.
    """
    img = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(arr, (2, 1, 0))))
    img.SetSpacing([float(s) for s in src_spacing])
    f = sitk.ResampleImageFilter()
    f.SetSize([int(s) for s in dst_size])
    f.SetOutputSpacing([float(s) for s in dst_spacing])
    f.SetOutputOrigin(img.GetOrigin())
    f.SetOutputDirection(img.GetDirection())
    f.SetInterpolator(interpolator)
    f.SetDefaultPixelValue(0)
    return np.transpose(sitk.GetArrayFromImage(f.Execute(img)), (2, 1, 0))


def resample_prediction_to_original(prob_map, spacing, target_spacing=None, original_shape=None) -> np.ndarray:
    """Put a probability map back on the original grid, Linear, clipped to [0, 1].

    The **continuous** probability is resampled and only then thresholded. Thresholding first and
    resampling the binary mask would put nearest-neighbour staircase edges on every lesion.
    """
    if target_spacing is None:
        target_spacing = TARGET_SPACING
    p = resample_to_original(
        np.asarray(prob_map, dtype=np.float32),
        target_spacing,
        spacing,
        np.asarray(original_shape).ravel(),
        sitk.sitkLinear,
    )
    return np.clip(p, 0.0, 1.0)
