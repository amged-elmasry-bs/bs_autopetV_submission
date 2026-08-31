"""The model's input and output contract.

These are the facts a caller has to agree with the weights on. They are separate from
:mod:`BS_Submission.config`, which holds training hyperparameters: changing a value here
changes what the trained network *means*, not how it was fitted.
"""

from __future__ import annotations

# --- input channels, in order ---------------------------------------------------------------
# The network takes four channels stacked on axis 0. The prompt channels are not
# interchangeable: channel 2 carries foreground (lesion) prompts and channel 3 background ones,
# and swapping them inverts the correction a user is asking for.
CHANNEL_ORDER = ("ct", "pet", "prompt_foreground", "prompt_background")

CHANNEL_CT = 0
CHANNEL_PET = 1
CHANNEL_PROMPT_FOREGROUND = 2  # stored as `_0002`; "tumor" prompts
CHANNEL_PROMPT_BACKGROUND = 3  # stored as `_0003`

# --- prompt encoding ------------------------------------------------------------------------
# Prompts reach the network in [0, 1], quantised to 255 levels. Preparation stores them as
# uint8 scaled by this factor and divides by it at load; inference encodes live prompts the same
# way, so a prompt drawn at inference lands on the same scale as one seen during training.
PROMPT_QUANTISATION = 255.0

# --- output -------------------------------------------------------------------------------
# Probability above which a voxel is called lesion. Not 0.5: it was tuned on validation, and it
# is part of the released model's behaviour rather than a free parameter.
SEGMENTATION_THRESHOLD = 0.65

__all__ = [
    "CHANNEL_CT",
    "CHANNEL_ORDER",
    "CHANNEL_PET",
    "CHANNEL_PROMPT_BACKGROUND",
    "CHANNEL_PROMPT_FOREGROUND",
    "PROMPT_QUANTISATION",
    "SEGMENTATION_THRESHOLD",
]
