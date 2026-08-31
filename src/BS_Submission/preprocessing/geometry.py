"""Grid arithmetic and resampling.

Every channel of a case ends up on one **reference grid**: the CT's field of view and
orientation, at TARGET_SPACING. That matters because the inputs do not agree on geometry --
CT is finer than the target, FDG heatmaps happen to match it, and PSMA heatmaps are coarser in
plane and finer through it. Resampling them all onto the CT-derived grid is what makes the four
channels stackable.
"""

from __future__ import annotations

import os

import numpy as np
import SimpleITK as sitk

from .constants import TARGET_SPACING

Grid = tuple[list[int], list[float], tuple[float, ...], tuple[float, ...]]


def reference_grid(ct: sitk.Image) -> Grid:
    """The grid every channel is resampled onto: the CT's frame at the target spacing."""
    size = [
        int(round(sz * sp / ts))
        # strict: a non-3D input would otherwise zip-truncate into a wrong grid.
        for sz, sp, ts in zip(ct.GetSize(), ct.GetSpacing(), TARGET_SPACING, strict=True)
    ]
    return size, TARGET_SPACING, ct.GetOrigin(), ct.GetDirection()


def resample(img: sitk.Image, grid: Grid, interpolator: int, default: float = 0.0) -> sitk.Image:
    """Resample onto `grid`. The interpolator is the caller's choice and it matters.

    Intensities use BSpline, prompt heatmaps Linear, and label maps NearestNeighbour -- anything
    smoothing applied to a label map would invent fractional classes along every boundary.
    """
    size, spacing, origin, direction = grid
    f = sitk.ResampleImageFilter()
    f.SetSize(size)
    f.SetOutputSpacing(spacing)
    f.SetOutputOrigin(origin)
    f.SetOutputDirection(direction)
    f.SetInterpolator(interpolator)
    f.SetDefaultPixelValue(default)
    return f.Execute(img)


def to_xyz(img: sitk.Image) -> np.ndarray:
    """SimpleITK hands back (z, y, x); the stored arrays are (x, y, z)."""
    return np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))


def same_grid(a: sitk.Image, b: sitk.Image, tol: float = 1e-4) -> bool:
    """Whether two images already share a grid, so resampling can be skipped."""
    if a.GetSize() != b.GetSize():
        return False
    return all(
        abs(x - y) <= tol
        for x, y in zip(
            a.GetSpacing() + a.GetOrigin() + a.GetDirection(),
            b.GetSpacing() + b.GetOrigin() + b.GetDirection(),
            strict=True,
        )
    )


def mask_on(ref_img: sitk.Image, mask_path: str, erode: int) -> np.ndarray | None:
    """Load an organ mask onto `ref_img`'s grid, optionally eroded. None if absent.

    Erosion drops the organ capsule and partial-volume edge voxels, which otherwise pull a
    reference-region statistic toward neighbouring tissue.
    """
    if not os.path.exists(mask_path):
        return None
    m = sitk.ReadImage(mask_path)
    if not same_grid(ref_img, m):
        m = sitk.Resample(m, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, m.GetPixelID())
    if erode:
        m = sitk.BinaryErode(sitk.Cast(m, sitk.sitkUInt8), [erode, erode, erode])
    return sitk.GetArrayFromImage(m) > 0
