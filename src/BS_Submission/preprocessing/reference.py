"""Per-scan PET reference region, and the cohort-wide statistics built from it.

PET is normalised as ``global_z(arcsinh(SUV / reference))``. The reference is a robust SUV taken
from a CT-derived organ mask on that same scan, which is what makes the axis comparable across
tracers without needing labels: each scan self-calibrates. The global z afterwards only
re-centres the already-harmonised axis, using values fixed over the cohort.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from .constants import (
    AORTA_ERODE,
    LIVER_ERODE,
    MAD_K,
    MIN_AORTA_VOX,
    MIN_LIVER_VOX,
)
from .geometry import mask_on, same_grid
from .layout import Layout


def organ_ref(pet: np.ndarray, mask: np.ndarray, min_vox: int) -> tuple[float | None, float, float]:
    """Robust reference SUV inside a mask.

    Tumour inside the reference organ would inflate a plain mean, so focal outliers above
    ``median + MAD_K * 1.4826 * MAD`` are dropped before averaging. Returns
    ``(ref, cv, frac_outlier)``, or ``(None, nan, nan)`` if the mask is too small to trust.
    """
    v = pet[mask]
    v = v[v > 0]
    if v.size < min_vox:
        return None, float("nan"), float("nan")
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) + 1e-6
    cv = 1.4826 * mad / med
    thr = med + MAD_K * 1.4826 * mad
    frac_out = float(np.mean(v > thr))
    clean = v[v <= thr]
    return float(clean.mean()), cv, frac_out


def reference(
    pet: np.ndarray, ct_arr: np.ndarray, masks: dict[str, np.ndarray | None], order: list[str]
) -> tuple[float, str, float, float]:
    """Try reference organs in `order`, falling back until one yields a usable value.

    The chain exists because whole-body scans are not guaranteed to contain a usable liver:
    partial-body acquisitions and resections happen. Falling back to the aortic blood pool, then
    a soft-tissue percentile, keeps every scan on a defined axis. Returns
    ``(ref, source, cv, frac_outlier)``; `source` records which rung was used.
    """
    for organ in order:
        if organ == "body":  # soft-tissue percentile fallback
            body = (ct_arr > -100) & (ct_arr < 100) & (pet > 0)
            if body.any():
                return float(np.percentile(pet[body], 50)), "body_p50", float("nan"), float("nan")
            continue
        m = masks.get(organ)
        if m is None:
            continue
        min_vox = MIN_LIVER_VOX if organ == "liver" else MIN_AORTA_VOX
        if int(m.sum()) < min_vox:
            continue
        ref, cv, frac = organ_ref(pet, m, min_vox)
        if ref is not None:
            return ref, organ, cv, frac
    return float(np.median(pet[pet > 0])), "global_median", float("nan"), float("nan")


def ref_case(case: str, order: list[str], layout: Layout):
    """One case: its PET reference, plus its contribution to the cohort-wide z statistics.

    The sums are returned rather than the values so the caller can accumulate them across a
    process pool and compute one mean and standard deviation over every body voxel in the cohort.
    """
    _, liver_p, aorta_p = layout.case_seg_paths(case)
    pet_img = sitk.ReadImage(layout.raw_pet(case))
    ct_img = sitk.ReadImage(layout.raw_ct(case))
    pet = sitk.GetArrayFromImage(pet_img).astype(np.float32)
    ct_arr = (
        sitk.GetArrayFromImage(ct_img).astype(np.float32)
        if same_grid(pet_img, ct_img)
        else sitk.GetArrayFromImage(sitk.Resample(ct_img, pet_img, sitk.Transform(), sitk.sitkLinear, -1000.0))
    )
    masks = {
        "liver": mask_on(pet_img, liver_p, LIVER_ERODE),
        "aorta": mask_on(pet_img, aorta_p, AORTA_ERODE),
    }
    ref, src, cv, frac = reference(pet, ct_arr, masks, order)
    body = pet[pet > 0]
    a = np.arcsinh(body / (ref + 1e-8))
    return case, ref, src, cv, frac, float(a.sum()), float((a * a).sum()), int(a.size)
