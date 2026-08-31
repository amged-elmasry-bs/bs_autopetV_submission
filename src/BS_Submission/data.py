"""Dataset and dataloader construction.

The CSV holds one case per row as paths, not arrays; the pipeline memory-maps them and
reads only the sampled window. Training and validation use the same pipeline, so both
see randomly-cropped patches."""

from __future__ import annotations

import os

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .preprocess import PatchPipeline


class AutoPETDataset(Dataset):
    """One CSV row -> one patch. Both splits use the same preprocessing graph."""

    COLUMNS = ("ct", "pet", "hm0", "hm1", "mask")

    def __init__(self, csv_path: str, cfg: Config) -> None:
        self.rows = pd.read_csv(csv_path)
        missing = [c for c in self.COLUMNS if c not in self.rows.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing column(s): {missing}")
        self.cfg = cfg
        self.pipeline = PatchPipeline(
            patch_size=list(cfg.patch_size),
            p_bg=cfg.p_bg,
            small_boost=cfg.small_boost,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, value: str) -> str:
        """Resolve one CSV entry to a path on this machine.

        Relative entries -- how the shipped splits are written -- hang off ``data_root``. An
        absolute entry is used as-is, unless ``path_from`` is set, which rewrites its prefix so
        a CSV written on another machine can be reused without editing it.
        """
        value = value.strip()
        if self.cfg.path_from and value.startswith(self.cfg.path_from):
            return value.replace(self.cfg.path_from, self.cfg.path_to, 1)
        if os.path.isabs(value):
            return value
        return os.path.join(self.cfg.data_root, value)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        return self.pipeline(*(self._path(row[c]) for c in self.COLUMNS))


def build_loader(csv_path: str, cfg: Config, *, shuffle: bool, batch_size: int, pin_memory: bool) -> DataLoader:
    """DataLoader with the JSON's dataloader kwargs; _collate_stack == default torch.stack."""
    return DataLoader(
        AutoPETDataset(csv_path, cfg),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers else None,
        pin_memory=pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers else False,
        timeout=cfg.timeout,
    )
