"""Where the inputs and outputs live.

The original script kept these as module-level globals relative to one working directory. They
are a dataclass here so the pipeline can be pointed at any dataset, and so worker processes get
them explicitly rather than by import side effect.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    """Input and output directories for one preprocessing run."""

    dataset_dir: str
    heatmaps_dir: str
    out_dir: str
    # Optional overrides, so the released per-case references can be used in place rather than
    # copied into every output directory.
    ref_json_override: str = ""
    stats_npz_override: str = ""

    @property
    def images_dir(self) -> str:
        """Raw `{case}_0000.nii.gz` (CT) and `{case}_0001.nii.gz` (PET)."""
        return os.path.join(self.dataset_dir, "imagesTr")

    @property
    def labels_dir(self) -> str:
        """Raw `{case}.nii.gz` lesion masks."""
        return os.path.join(self.dataset_dir, "labelsTr")

    @property
    def seg_dir(self) -> str:
        """Cached organ segmentations, one subdirectory per case."""
        return os.path.join(self.out_dir, "seg")

    @property
    def ct_dir(self) -> str:
        return os.path.join(self.out_dir, "ct")

    @property
    def pet_dir(self) -> str:
        return os.path.join(self.out_dir, "pet")

    @property
    def labels_out_dir(self) -> str:
        return os.path.join(self.out_dir, "labels")

    @property
    def scribbles_dir(self) -> str:
        """Prompt heatmap channels, on the reference grid."""
        return os.path.join(self.out_dir, "scribbles", "v1")

    @property
    def ref_json(self) -> str:
        """Per-case PET reference values: the override if given, else this run's own."""
        return self.ref_json_override or os.path.join(self.out_dir, "ref.json")

    @property
    def stats_npz(self) -> str:
        """Cohort-wide PET global-z statistics: the override if given, else this run's own."""
        return self.stats_npz_override or os.path.join(self.out_dir, "pet_norm_stats.npz")

    @property
    def meta_json(self) -> str:
        """Per-case original spacing and shape, written by the `prep` stage."""
        return os.path.join(self.out_dir, "meta.json")

    def case_seg_paths(self, case: str) -> tuple[str, str, str]:
        """Directory and the two organ masks cached for one case."""
        directory = os.path.join(self.seg_dir, case.replace(" ", "_").replace("/", "_"))
        return (
            directory,
            os.path.join(directory, "liver.nii.gz"),
            os.path.join(directory, "aorta.nii.gz"),
        )

    def raw_ct(self, case: str) -> str:
        return os.path.join(self.images_dir, f"{case}_0000.nii.gz")

    def raw_pet(self, case: str) -> str:
        return os.path.join(self.images_dir, f"{case}_0001.nii.gz")

    def raw_mask(self, case: str) -> str:
        return os.path.join(self.labels_dir, f"{case}.nii.gz")

    def raw_heatmap(self, case: str, channel: int) -> str:
        return os.path.join(self.heatmaps_dir, f"{case}_000{channel}.nii.gz")
