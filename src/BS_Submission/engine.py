"""The training engine: one training epoch then one validation epoch, repeated."""

from __future__ import annotations

import math

import torch

try:  # the umbrella `lightning` package is optional; lightning-fabric alone suffices
    import lightning as L
    from lightning.fabric.loggers import TensorBoardLogger
except ImportError:  # pragma: no cover - environment dependent
    import lightning_fabric as L
    from lightning_fabric.loggers import TensorBoardLogger

from .checkpointing import TopKCheckpointer
from .config import TAG_TRAIN, TAG_VAL, Config
from .data import build_loader
from .losses import DeepSupervisionLoss, TverskyCELoss
from .model import build_model, build_optimizer


def _epoch_mean(fabric, local_sum: float, local_count: int) -> float:
    """Reduce per-rank (sum, count) into the global mean; a no-op on one device."""
    if getattr(fabric, "world_size", 1) > 1:
        stats = torch.tensor([float(local_sum), float(local_count)], dtype=torch.float32, device=fabric.device)
        stats = fabric.all_reduce(stats, reduce_op="sum")
        local_sum, local_count = stats[0].item(), stats[1].item()
    return local_sum / local_count if local_count else float("nan")


def run(cfg: Config) -> None:
    """Alternate one training epoch and one validation epoch, `n_epochs` times."""
    logger = TensorBoardLogger(
        root_dir=cfg.log_root_dir, name=cfg.log_name, default_hp_metric=cfg.log_default_hp_metric
    )
    fabric = L.Fabric(
        devices=list(cfg.devices),
        num_nodes=cfg.num_nodes,
        accelerator=cfg.accelerator,
        strategy=cfg.strategy,
        precision=cfg.precision,
        loggers=[logger],
    )
    fabric.launch()

    model = build_model(cfg)
    if cfg.compile_model:
        model = torch.compile(model)
    optimizer = build_optimizer(model, cfg)
    model, optimizer = fabric.setup(model, optimizer)
    # No learning-rate scheduler: the rate is constant for the whole run.
    # `callbacks` list is empty, so it is never attached. See README.

    train_loader = fabric.setup_dataloaders(
        build_loader(
            cfg.train_csv,
            cfg,
            shuffle=cfg.train_shuffle,
            batch_size=cfg.train_batch_size,
            pin_memory=cfg.train_pin_memory,
        )
    )
    val_loader = fabric.setup_dataloaders(
        build_loader(
            cfg.val_csv, cfg, shuffle=cfg.val_shuffle, batch_size=cfg.val_batch_size, pin_memory=cfg.val_pin_memory
        )
    )

    criterion = DeepSupervisionLoss(
        TverskyCELoss(
            num_classes=cfg.num_classes,
            alpha=cfg.alpha,
            beta=cfg.beta,
            smooth=cfg.smooth,
            include_background=cfg.include_background,
        )
    )
    checkpointer = TopKCheckpointer(cfg, fabric)
    step = 0
    val_step = 0

    for epoch in range(cfg.n_epochs):
        model.train()
        running, counted = 0.0, 0
        for image, mask in train_loader:
            logits = model(image)
            loss = criterion(logits, mask)
            fabric.backward(loss)
            if cfg.grad_clip:
                # error_if_nonfinite=False matches nnU-Net: under fp16 the unscaled grads can
                # overflow early, and the scaler should handle it by skipping the step rather
                # than this raising.
                fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_clip, error_if_nonfinite=False)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            value = loss.item()
            if not math.isnan(value):
                running += value
                counted += 1
            fabric.log(TAG_TRAIN, value, step=step)
            if cfg.log_every and step % cfg.log_every == 0:
                fabric.print(
                    f"  ep {epoch}/{cfg.n_epochs} it {step % max(len(train_loader), 1)}"
                    f"/{len(train_loader)} loss {value:.5f}"
                )
            step += 1
        train_loss = _epoch_mean(fabric, running, counted)
        fabric.log(TAG_TRAIN + "_on_epoch", train_loss, step=epoch)

        # Validation: forward and loss only. the reference implementation reuses the same training-loop service with
        # mode="eval" and an empty grad_clip, so no backward and no optimizer step run.
        model.eval()
        running, counted = 0.0, 0
        grad_ctx = torch.no_grad() if cfg.val_no_grad else torch.enable_grad()
        with grad_ctx:
            for image, mask in val_loader:
                value = criterion(model(image), mask).item()
                if not math.isnan(value):
                    running += value
                    counted += 1
                # Per-batch, matching how the reference logger logs each validation batch.
                fabric.log(TAG_VAL, value, step=val_step)
                val_step += 1
        val_loss = _epoch_mean(fabric, running, counted)
        fabric.log(TAG_VAL + "_on_epoch", val_loss, step=epoch)
        checkpointer.update(model, epoch, val_loss)

        fabric.print(f"epoch {epoch + 1}/{cfg.n_epochs}  train {train_loss:.5f}  val {val_loss:.5f}")
