"""Checkpoint retention: the best ``save_top_k`` by monitored metric, plus a rolling last."""

from __future__ import annotations

import math
import os

import torch

from .config import Config


class TopKCheckpointer:
    """ModelCheckpoint: keep the best `save_top_k` by the monitored metric, plus `last`.

    checkpoints are selected on the epoch-mean validation loss.
    """

    def __init__(self, cfg: Config, fabric) -> None:
        self.cfg = cfg
        self.fabric = fabric
        self.entries: list[dict] = []
        self.last_path: str | None = None
        os.makedirs(cfg.checkpoint_save_path, exist_ok=True)

    def _better(self, value: float, worst: float) -> bool:
        if self.cfg.ckpt_mode == "min":
            return value < (worst - self.cfg.ckpt_min_delta)
        return value > (worst + self.cfg.ckpt_min_delta)

    def _save(self, model, epoch: int, value: float, label: str) -> str:
        stem = f"{self.cfg.ckpt_filename_prefix}_{label}_epoch={epoch}_val_loss={value:.8g}"
        path = os.path.join(self.cfg.checkpoint_save_path, f"{stem}.pt")
        state = getattr(model, "_orig_mod", model)  # unwrap torch.compile
        state = getattr(state, "module", state)  # unwrap DDP
        torch.save(state.state_dict(), path)
        return path

    def update(self, model, epoch: int, value: float) -> None:
        """Called once per validation epoch, on global rank zero only."""
        if not self.fabric.is_global_zero or not math.isfinite(value):
            return
        if self.cfg.ckpt_save_last:
            if self.last_path and os.path.exists(self.last_path):
                os.remove(self.last_path)
            self.last_path = self._save(model, epoch, value, "last")
        if not self.cfg.ckpt_save_top_k:
            return
        qualifies = len(self.entries) < self.cfg.ckpt_save_top_k or self._better(value, self.entries[-1]["value"])
        if not qualifies:
            return
        self.entries.append({"value": value, "epoch": epoch, "path": self._save(model, epoch, value, "topk")})
        self.entries.sort(key=lambda e: e["value"], reverse=self.cfg.ckpt_mode == "max")
        for dropped in self.entries[self.cfg.ckpt_save_top_k :]:
            if os.path.exists(dropped["path"]):
                os.remove(dropped["path"])
        self.entries = self.entries[: self.cfg.ckpt_save_top_k]
