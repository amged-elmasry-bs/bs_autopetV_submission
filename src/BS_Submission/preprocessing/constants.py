"""Normalisation constants that define the model's input space.

These are committed as literals on purpose. They were previously held in `.npz` files on a
single machine, which meant the normalisation a trained checkpoint expects was neither auditable
nor recoverable if those files were lost -- and every released weight would be unusable without
them. Each value below is derived once over the training cohort and then fixed: they are
*inputs* at both training and inference time, not per-run outputs.

Override at the command line with --stats / --pet-stats if you have regenerated them for a
different cohort. Changing them invalidates every existing checkpoint.
"""

from __future__ import annotations

# --- CT: clip to robust percentiles of the cohort, then a fixed global z-score ---
# From the cohort-wide CT intensity statistics (`ct_norm_stats.npz`, pct_* entries).
CT_LOW = -811.7323071289062
CT_HIGH = 1137.4169641113294
CT_MEAN = 124.57824523175886
CT_STD = 280.88337926485696

# --- Resampling target, in mm (x, y, z) ---
# The FDG PET grid; CT is finer and PSMA PET is coarser, so both are resampled onto a
# CT-derived reference grid at this spacing.
TARGET_SPACING = [2.0364, 2.0364, 3.0]

# --- PET: liver-SUVr -> arcsinh -> fixed global z-score ---
# Mean and standard deviation of arcsinh(SUV / liver_ref) over body voxels of the whole training
# cohort. This only re-centres an already per-scan-harmonised axis; it does not re-normalise per
# scan, which is what keeps the axis comparable across tracers and at inference.
PET_MEAN = 0.090390644967556
PET_STD = 0.2537791430950165

# --- Reference-region tuning (dimensionless, hence tracer-agnostic) ---
LIVER_ERODE = 3  # voxels eroded off the liver mask, dropping capsule and partial-volume edges
AORTA_ERODE = 1
MIN_LIVER_VOX = 500  # below this the liver counts as absent (partial body, resection) -> fall back
MIN_AORTA_VOX = 100
MAD_K = 3.0  # focal-outlier cutoff at median + MAD_K * 1.4826 * MAD, to drop tumour hotspots

# Fallback order per configured primary reference organ. `body` is a soft-tissue percentile, the
# last resort before a global median.
#
# The released dataset used "aorta" for all 1611 cases. That is a deliberate choice, not a
# fallback: liver masks were available and usable for every case, but liver uptake differs
# sharply by tracer -- on the sampled PSMA scans the liver reference came out 5-6x the blood-pool
# value -- so the blood pool is what keeps one axis comparable across FDG and PSMA.
REF_ORDER = {"liver": ["liver", "aorta", "body"], "aorta": ["aorta", "body"]}
SHIPPED_REF_ORGAN = "aorta"
