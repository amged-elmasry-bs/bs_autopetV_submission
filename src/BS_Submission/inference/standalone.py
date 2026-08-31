"""Segment one case straight from its CT and PET, with no prepared dataset.

The batch path expects a whole cohort laid out on disk. This is the single-scan equivalent: read
two NIfTIs, derive the PET reference the same way the cohort does, normalise and resample through
the *same* functions, predict, and hand back a mask on the CT's own grid.

The PET reference is the substantive part. PET is normalised as
``global_z(arcsinh(SUV / reference))`` where the reference is a robust SUV inside a CT-derived
organ mask on that same scan -- so each scan self-calibrates and the axis stays comparable across
tracers. Getting it from the aorta needs a segmentation of the aorta, which is why
TotalSegmentator runs here unless a reference is supplied outright.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import SimpleITK as sitk

from ..preprocessing.constants import (
    AORTA_ERODE,
    LIVER_ERODE,
    PET_MEAN,
    PET_STD,
    REF_ORDER,
    SHIPPED_REF_ORGAN,
)
from ..preprocessing.geometry import mask_on, reference_grid, to_xyz
from ..preprocessing.pipeline import normalise_ct, normalise_pet
from ..preprocessing.reference import reference
from .predict import Prompts, predict_case

ORGAN_MASKS = ("liver", "aorta")


def segment_organs(ct_path: str, out_dir: str, device: str = "gpu:0") -> None:
    """Write liver and aorta masks for one CT, exactly as the cohort's `seg` stage does."""
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as exc:  # the traceback alone points deep into someone else's imports
        raise SystemExit(
            f"TotalSegmentator is needed to find the reference organ, and is not importable: {exc}\n"
            '  Install it with:  pip install -e ".[preprocess]"\n'
            "  Or pass --ref <suv> if you already know this scan's reference, which skips organ "
            "segmentation entirely."
        ) from exc

    os.makedirs(out_dir, exist_ok=True)
    totalsegmentator(
        input=ct_path,
        output=out_dir,
        task="total",
        fast=True,
        roi_subset=list(ORGAN_MASKS),
        device=device,
        quiet=True,
    )


def pet_reference(
    ct_img, pet_img, organ_dir: str, ref_organ: str = SHIPPED_REF_ORGAN
) -> tuple[float, str, float, float]:
    """The scan's reference SUV, from organ masks already written to ``organ_dir``.

    Returns ``(ref, source, cv, frac_outlier)``. ``source`` records which rung of the fallback
    chain produced it -- worth logging, because a scan that silently fell through to
    ``body_p50`` or ``global_median`` is not on the axis the released weights were trained
    against, even though it will still produce a mask.
    """
    pet = sitk.GetArrayFromImage(pet_img).astype(np.float32)
    ct_on_pet = sitk.GetArrayFromImage(
        sitk.Resample(ct_img, pet_img, sitk.Transform(), sitk.sitkLinear, -1000.0)
    ).astype(np.float32)
    masks = {
        "liver": mask_on(pet_img, os.path.join(organ_dir, "liver.nii.gz"), LIVER_ERODE),
        "aorta": mask_on(pet_img, os.path.join(organ_dir, "aorta.nii.gz"), AORTA_ERODE),
    }
    return reference(pet, ct_on_pet, masks, REF_ORDER[ref_organ])


def segment_case(
    ct_path: str,
    pet_path: str,
    model,
    ref: float | None = None,
    ref_organ: str = SHIPPED_REF_ORGAN,
    organ_dir: str = "",
    device: str = "cuda",
    threshold: float | None = None,
    sw_batch_size: int = 1,
    prompts: Prompts | None = None,
    pet_mean: float = PET_MEAN,
    pet_std: float = PET_STD,
    on_reference=None,
):
    """One case end to end. Returns ``(mask_image, probability_image)`` on the CT's own grid.

    ``ref`` skips the organ segmentation entirely when the reference SUV is already known.
    Otherwise TotalSegmentator runs into ``organ_dir``, or a temporary directory if none is
    given, and the reference is derived from the aortic blood pool.
    """
    ct_img = sitk.Cast(sitk.ReadImage(ct_path), sitk.sitkFloat32)
    pet_img = sitk.Cast(sitk.ReadImage(pet_path), sitk.sitkFloat32)

    temporary = None
    if ref is None:
        directory = organ_dir
        if not directory:
            temporary = tempfile.TemporaryDirectory(prefix="bs_organs_")
            directory = temporary.name
        if not all(os.path.exists(os.path.join(directory, f"{o}.nii.gz")) for o in ORGAN_MASKS):
            segment_organs(ct_path, directory, device="gpu:0" if device.startswith("cuda") else "cpu")
        ref, source, cv, frac = pet_reference(ct_img, pet_img, directory, ref_organ)
    else:
        source, cv, frac = "given", float("nan"), float("nan")
    if on_reference:
        on_reference(ref, source, cv, frac)

    grid = reference_grid(ct_img)
    ct = np.asarray(normalise_ct(ct_img, grid), np.float32)
    pet = np.asarray(normalise_pet(pet_img, grid, ref, pet_mean, pet_std), np.float32)

    original_shape = [int(s) for s in ct_img.GetSize()]
    original_spacing = [float(s) for s in ct_img.GetSpacing()]
    extra = {} if threshold is None else {"threshold": threshold}
    prediction = predict_case(
        model,
        ct,
        pet,
        original_shape=original_shape,
        original_spacing=original_spacing,
        prompts=prompts or Prompts(),
        device=device,
        sw_batch_size=sw_batch_size,
        **extra,
    )
    if temporary:
        temporary.cleanup()

    return as_image(prediction.mask, ct_img, sitk.sitkUInt8), as_image(
        prediction.probability, ct_img, sitk.sitkFloat32
    )


def as_image(array: np.ndarray, like, pixel_type):
    """An (x, y, z) array back to a SimpleITK image carrying ``like``'s geometry.

    The geometry is what makes the output openable beside its input and comparable against an
    annotation, so it is copied rather than left at the default identity frame.
    """
    image = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(array, (2, 1, 0))))
    image.CopyInformation(like)
    return sitk.Cast(image, pixel_type)
