"""Whole-volume prediction for one case."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..constants import SEGMENTATION_THRESHOLD
from ..preprocessing.constants import TARGET_SPACING
from .prompts import PromptEncoder
from .resample import resample_prediction_to_original
from .window import sliding_window_inference

PATCH_SIZE = (192, 192, 192)


@dataclass
class Prediction:
    """One case's result, on the grid the annotation lives on."""

    probability: np.ndarray
    mask: np.ndarray
    threshold: float
    n_foreground_prompts: int = 0
    n_background_prompts: int = 0


@dataclass
class Prompts:
    """Prompt coordinates on the original grid, accumulated across correction rounds."""

    foreground: list = field(default_factory=list)
    background: list = field(default_factory=list)

    def add(self, kind: str, coords) -> None:
        target = self.foreground if kind in ("tumor", "foreground") else self.background
        target.extend([list(map(int, c)) for c in coords])


def predict_case(
    model,
    ct: np.ndarray,
    pet: np.ndarray,
    original_shape,
    original_spacing,
    prompts: Prompts | None = None,
    device: str = "cuda",
    threshold: float = SEGMENTATION_THRESHOLD,
    sw_batch_size: int = 1,
    prompt_sigma: float = 0.0,
) -> Prediction:
    """Predict one case from prepared CT and PET, with optional prompts.

    ``ct`` and ``pet`` are already normalised and on the target grid -- the same arrays training
    reads. Prompt coordinates are on the *original* grid, which is where a user points, and are
    encoded onto the target grid here.

    The probability is reverse-resampled to the original grid **before** thresholding, so lesion
    boundaries come from interpolating a continuous field rather than from upsampling a binary
    mask.
    """
    prompts = prompts or Prompts()
    encoder = PromptEncoder(original_shape, original_spacing, ct.shape, prompt_sigma)
    stacked = np.stack([ct, pet, encoder(prompts.foreground), encoder(prompts.background)], 0).astype(np.float32)

    probability_target = sliding_window_inference(
        model,
        torch.from_numpy(stacked),
        PATCH_SIZE,
        num_classes=2,
        overlap=0.5,
        device=device,
        sw_batch_size=sw_batch_size,
    )
    probability = resample_prediction_to_original(
        probability_target, np.asarray(original_spacing, np.float32), TARGET_SPACING, original_shape
    )
    return Prediction(
        probability=probability,
        mask=(probability > threshold).astype(np.uint8),
        threshold=threshold,
        n_foreground_prompts=len(prompts.foreground),
        n_background_prompts=len(prompts.background),
    )
