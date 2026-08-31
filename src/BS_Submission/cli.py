"""Command-line entry point for training.

Flags map onto :class:`~BS_Submission.config.Config` fields and default to that dataclass.
``--fold`` is the usual way in: it selects the shipped split pair and the matching
initialisation weights, so a run is one flag plus the location of the volumes.
"""

from __future__ import annotations

import argparse
import os

from .config import Config
from .engine import run

N_FOLDS = 5


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser, grouped by what each flag affects."""
    parser = argparse.ArgumentParser(
        prog="bs-train",
        description="Train the PET/CT lesion segmentation model on one cross-validation fold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="example: bs-train --fold 0 --data-root /data/preprocessed",
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--fold",
        type=int,
        choices=range(N_FOLDS),
        metavar=f"{{0..{N_FOLDS - 1}}}",
        help="fold to train: selects splits/fold_<n>_{train,val}.csv and the fold's weights",
    )
    data.add_argument(
        "--data-root",
        default=Config.data_root,
        help="directory the relative paths in the split CSVs are resolved against",
    )
    data.add_argument("--splits-dir", default=Config.splits_dir, help="directory holding the split CSVs")
    data.add_argument("--train-csv", default=Config.train_csv, help="training split; overrides --fold")
    data.add_argument("--val-csv", default=Config.val_csv, help="validation split; overrides --fold")
    data.add_argument(
        "--path-from",
        default=Config.path_from,
        help="prefix to rewrite in absolute CSV paths, for data moved between machines",
    )
    data.add_argument("--path-to", default=Config.path_to, help="replacement for --path-from")
    data.add_argument("--num-workers", type=int, default=Config.num_workers, help="dataloader workers per split")

    model = parser.add_argument_group("model")
    model.add_argument("--arch-path", default=Config.arch_path, help="dotted path to the architecture class")
    model.add_argument("--pretrained-dir", default=Config.pretrained_dir, help="directory holding the per-fold weights")
    model.add_argument(
        "--checkpoint-path",
        default=Config.checkpoint_path,
        help="weights to initialise from; overrides --fold. 'none' trains from scratch",
    )
    model.add_argument("--no-compile", action="store_true", help="skip torch.compile")

    run_group = parser.add_argument_group("run")
    run_group.add_argument("--devices", default="1", help="comma-separated CUDA indices, e.g. '1' or '1,2'")
    run_group.add_argument(
        "--epochs", type=int, default=Config.n_epochs, help="training epochs, each followed by validation"
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--checkpoint-save-path", default=Config.checkpoint_save_path, help="directory for saved checkpoints"
    )
    out.add_argument("--log-root-dir", default=Config.log_root_dir, help="TensorBoard root directory")
    out.add_argument("--log-every", type=int, default=Config.log_every, help="print progress every N steps; 0 disables")
    return parser


def resolve_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    """Fill the split pair and initial weights in from ``--fold`` where not given explicitly.

    Returns ``(train_csv, val_csv, checkpoint_path)``. A fold trains from the weights of the
    same fold, so the initialisation never saw that fold's validation cases.
    """
    train, val, ckpt = args.train_csv, args.val_csv, args.checkpoint_path
    if args.fold is not None:
        train = train or os.path.join(args.splits_dir, f"fold_{args.fold}_train.csv")
        val = val or os.path.join(args.splits_dir, f"fold_{args.fold}_val.csv")
        ckpt = ckpt or os.path.join(args.pretrained_dir, Config.pretrained_template.format(fold=args.fold))
    return train, val, ckpt


def config_from_args(args: argparse.Namespace, *, check_paths: bool = False) -> Config:
    """Translate parsed arguments into a Config, resolving ``--fold`` first."""
    train, val, ckpt = resolve_paths(args)
    if check_paths:
        _require(train, "training split", args)
        _require(val, "validation split", args)
        if ckpt and ckpt.lower() != "none":
            _require(ckpt, "initialisation weights", args)
    return Config(
        fold=args.fold,
        splits_dir=args.splits_dir,
        data_root=args.data_root,
        pretrained_dir=args.pretrained_dir,
        train_csv=train,
        val_csv=val,
        arch_path=args.arch_path,
        checkpoint_path="" if ckpt.lower() == "none" else ckpt,
        checkpoint_save_path=args.checkpoint_save_path,
        log_root_dir=args.log_root_dir,
        devices=[int(d) for d in str(args.devices).split(",") if d != ""],
        n_epochs=args.epochs,
        num_workers=args.num_workers,
        path_from=args.path_from,
        path_to=args.path_to,
        compile_model=not args.no_compile,
        log_every=args.log_every,
    )


def _require(path: str, what: str, args: argparse.Namespace) -> None:
    """Fail early, and say exactly which file is missing rather than dying mid-epoch."""
    if not path:
        raise SystemExit(f"no {what} given: pass --fold, or set the path explicitly (see `bs-train --help`)")
    if not os.path.exists(path):
        hint = ""
        if args.fold is not None and "pretrained" in os.path.basename(path):
            hint = (
                f"\n  Per-fold weights are resolved as "
                f"<--pretrained-dir>/{Config.pretrained_template}; see assets/model/README.md "
                f"for where to obtain them, or pass --checkpoint-path none to train from scratch."
            )
        raise SystemExit(f"{what} not found: {path}{hint}")


def train_entry() -> None:
    """Console-script entry point: parse arguments and run training."""
    parser = build_parser()
    run(config_from_args(parser.parse_args(), check_paths=True))


if __name__ == "__main__":
    train_entry()
