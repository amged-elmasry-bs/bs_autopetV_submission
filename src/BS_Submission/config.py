"""Run configuration.

Every field is an explicit hyperparameter of the experiment; nothing is inferred from a
class name or a file path. Defaults are the settings the reported runs used, so a bare
``Config()`` reproduces them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

TAG_TRAIN = "train_loss_train_loss"
TAG_VAL = "val_loss_eval_loss"


@dataclass
class Config:
    """The `world` config, flattened."""

    # --- devices, distribution and precision ---
    devices: Sequence[int] = (1,)
    num_nodes: int = 1
    accelerator: str = "cuda"
    strategy: str = "ddp_find_unused_parameters_true"
    precision: str = "16-mixed"

    # --- TensorBoard logging ---
    log_root_dir: str = "runs/logs"
    log_name: str = "lightning_logs"
    log_default_hp_metric: bool = True

    # --- architecture and initial weights ---
    arch_path: str = "BS_Submission.models.residual_encoder_unet.ResidualEncoderUNetOrganAutoPET"
    arch_kwargs: dict = field(
        default_factory=lambda: {
            "in_channels": 4,
            "num_classes": 2,
            "deep_supervision": True,
        }
    )
    compile_model: bool = True
    checkpoint_path: str = ""
    checkpoint_save_path: str = "runs/checkpoints"

    # optimizer (torch.optim.SGD)
    lr: float = 0.001
    momentum: float = 0.99
    nesterov: bool = True
    dampening: float = 0
    weight_decay: float = 0.00003
    maximize: bool = False

    # --- fold selection ---
    # `fold` resolves both the split pair and the initialisation weights by convention;
    # train_csv / val_csv / checkpoint_path override it when set explicitly.
    fold: int | None = None
    splits_dir: str = "splits"
    data_root: str = ""
    pretrained_dir: str = "assets/model"
    pretrained_template: str = "pretrained_fold{fold}.pt"

    # --- training split ---
    train_csv: str = ""
    train_batch_size: int = 2
    train_shuffle: bool = True
    train_pin_memory: bool = True

    # --- validation split ---
    val_csv: str = ""
    val_batch_size: int = 2
    val_shuffle: bool = False
    val_pin_memory: bool = False

    # shared dataloader kwargs
    drop_last: bool = True
    num_workers: int = 16
    prefetch_factor: int = 4
    persistent_workers: bool = True
    timeout: int = 0

    # loop_policy (both workflows)
    n_epochs: int = 500

    # --- gradient clipping (training only; validation runs no backward pass) ---
    grad_clip: float = 12

    # --- loss ---
    num_classes: int = 2
    alpha: float = 0.3
    beta: float = 0.7
    smooth: float = 1e-5
    include_background: bool = False

    # --- checkpoint retention ---
    ckpt_mode: str = "min"
    ckpt_save_top_k: int = 5
    ckpt_save_last: bool = True
    ckpt_min_delta: float = 0
    ckpt_filename_prefix: str = "model_pretrained_resenc"

    # preprocessing graph (weighted_random.json), used by BOTH train and val
    patch_size: Sequence[int] = (192, 192, 192)
    p_bg: float = 0.33
    small_boost: float = 0.0

    # Deviation from the reference implementation, numerically neutral -- see README.
    val_no_grad: bool = True

    # Progress printing only; 0 disables. Not part of the reproduced config.
    log_every: int = 25

    # Optional prefix rewrite, for CSVs holding absolute paths from another machine.
    path_from: str = ""
    path_to: str = ""
