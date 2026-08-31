"""Architecture and optimizer construction."""

from __future__ import annotations

import importlib
import os

import torch
from torch import nn

from .config import Config


def build_model(cfg: Config) -> nn.Module:
    """Import the architecture by dotted path and load the pretrained checkpoint."""
    module_path, _, class_name = cfg.arch_path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Could not import `{module_path}`. Pass --arch-path pointing at a module that defines `{class_name}`."
        ) from exc
    model = getattr(module, class_name)(**cfg.arch_kwargs)

    if cfg.checkpoint_path:
        if not os.path.exists(cfg.checkpoint_path):
            raise ValueError(f"Path {cfg.checkpoint_path} does not exist")
        state_dict = torch.load(cfg.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[model] loaded {len(state_dict)} tensors from {cfg.checkpoint_path}")
    return model


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """SGD with nesterov momentum, exactly as the JSON's optimizer block specifies."""
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        nesterov=cfg.nesterov,
        dampening=cfg.dampening,
        weight_decay=cfg.weight_decay,
        maximize=cfg.maximize,
    )
