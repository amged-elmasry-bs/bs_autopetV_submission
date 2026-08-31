"""Writing one case's four channels and its masks onto the reference grid."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from .constants import CT_HIGH, CT_LOW, CT_MEAN, CT_STD
from .geometry import reference_grid, resample, to_xyz
from .layout import Layout


@dataclass(frozen=True)
class CaseJob:
    """One unit of work. Plain data so it survives being sent to a worker process."""

    case: str
    ref: float
    pet_mean: float
    pet_std: float
    layout: Layout
    pet_only: bool


def outputs_for(job: CaseJob) -> list[str]:
    """Every file this job is responsible for, so an interrupted run can resume."""
    layout, case = job.layout, job.case
    paths = [os.path.join(layout.pet_dir, f"{case}_0001.npy")]
    if not job.pet_only:
        paths += [
            os.path.join(layout.ct_dir, f"{case}_0000.npy"),
            os.path.join(layout.scribbles_dir, f"{case}_0002.npy"),
            os.path.join(layout.scribbles_dir, f"{case}_0003.npy"),
            os.path.join(layout.labels_out_dir, f"{case}.npy"),
            os.path.join(layout.labels_out_dir, f"{case}_orig.npy"),
        ]
    return paths


def normalise_pet(pet_img, grid, ref: float, pet_mean: float, pet_std: float) -> np.ndarray:
    """PET onto the reference grid: SUV over the scan's own reference, arcsinh, cohort z.

    Split out so a single-case run and a cohort run share one implementation. The cohort path is
    verified to reproduce the released arrays byte-for-byte, and anything calling this inherits
    that rather than being a second implementation free to drift.
    """
    suv = sitk.GetArrayFromImage(pet_img)
    z = (np.arcsinh(suv / (ref + 1e-8)) - pet_mean) / pet_std
    pet_norm = sitk.GetImageFromArray(z.astype(np.float32))
    pet_norm.CopyInformation(pet_img)
    return to_xyz(resample(pet_norm, grid, sitk.sitkBSpline)).astype(np.float16)


def normalise_ct(ct_img, grid) -> np.ndarray:
    """CT onto the reference grid: clipped to the cohort window, then the cohort z-score."""
    ct_c = sitk.Clamp(ct_img, lowerBound=CT_LOW, upperBound=CT_HIGH)
    ct_c = sitk.ShiftScale(ct_c, shift=-CT_MEAN, scale=1.0 / CT_STD)
    return to_xyz(resample(ct_c, grid, sitk.sitkBSpline)).astype(np.float16)


def process_case(job: CaseJob):
    """Normalise and resample one case. Returns ``(case, status, spacing, original shape)``.

    Channel by channel:

    * **PET** -- SUV over the scan's own reference, through arcsinh to compress the long
      upper tail, then the cohort-wide z. BSpline resample.
    * **CT** -- clip to the cohort percentiles, cohort z-score, BSpline resample.
    * **Prompt heatmaps** -- Linear resample, clipped to [0, 1], stored as uint8 x255. The
      published heatmaps arrive on several different grids and in more than one dtype, so this
      is a real resample, not a copy.
    * **Mask** -- written twice: nearest-neighbour on the reference grid for training, and once
      on its original grid, since evaluation belongs in the space the annotation was made in.
    """
    layout, case = job.layout, job.case
    pet_out = os.path.join(layout.pet_dir, f"{case}_0001.npy")

    reader = sitk.ImageFileReader()
    reader.SetFileName(layout.raw_ct(case))
    reader.ReadImageInformation()
    spacing = [float(s) for s in reader.GetSpacing()]
    orig_shape = [int(s) for s in reader.GetSize()]

    outs = outputs_for(job)
    if all(os.path.exists(p) for p in outs):
        return case, "skip", spacing, orig_shape

    try:
        ct_img = sitk.Cast(sitk.ReadImage(layout.raw_ct(case)), sitk.sitkFloat32)
        grid = reference_grid(ct_img)

        pet_img = sitk.Cast(sitk.ReadImage(layout.raw_pet(case)), sitk.sitkFloat32)
        np.save(pet_out, normalise_pet(pet_img, grid, job.ref, job.pet_mean, job.pet_std))

        if not job.pet_only:
            np.save(os.path.join(layout.ct_dir, f"{case}_0000.npy"), normalise_ct(ct_img, grid))
            for channel in (2, 3):
                hm_img = sitk.Cast(sitk.ReadImage(layout.raw_heatmap(case, channel)), sitk.sitkFloat32)
                hm = np.clip(to_xyz(resample(hm_img, grid, sitk.sitkLinear)), 0.0, 1.0)
                np.save(
                    os.path.join(layout.scribbles_dir, f"{case}_000{channel}.npy"),
                    np.rint(hm * 255.0).astype(np.uint8),
                )
            mask_img = sitk.ReadImage(layout.raw_mask(case))
            np.save(
                os.path.join(layout.labels_out_dir, f"{case}_orig.npy"),
                to_xyz(mask_img).astype(np.uint8),
            )
            mask_rs = resample(sitk.Cast(mask_img, sitk.sitkUInt8), grid, sitk.sitkNearestNeighbor)
            np.save(
                os.path.join(layout.labels_out_dir, f"{case}.npy"),
                to_xyz(mask_rs).astype(np.uint8),
            )
        return case, "done", spacing, orig_shape
    except Exception as exc:  # one bad case must not take the pool down
        return case, f"error: {type(exc).__name__}: {exc}", spacing, orig_shape
