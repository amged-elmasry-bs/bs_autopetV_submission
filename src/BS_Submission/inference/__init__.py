"""Whole-volume prediction, including interactive prompt corrections.

Normalisation, grids and the PET reference are **not** redefined here -- they are imported from
:mod:`BS_Submission.preprocessing`, so training and inference cannot drift onto different input
spaces. Only genuinely inference-side work lives in this package: sliding-window prediction,
reversing the resample, and encoding live prompts.
"""

from .prompts import PromptEncoder
from .resample import resample_prediction_to_original
from .window import sliding_window_inference

__all__ = ["PromptEncoder", "resample_prediction_to_original", "sliding_window_inference"]
