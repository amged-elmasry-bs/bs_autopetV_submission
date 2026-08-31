"""Whole-body PET/CT lesion segmentation with prompt channels.

Attributes are resolved lazily. Importing this package must not pull in the training stack,
because dataset preparation is installed as a separate extra and runs on machines that have no
torch or lightning at all -- an eager ``from .engine import run`` here would make
``bs-preprocess`` unusable exactly where it is meant to run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "1.0.0"

__all__ = ["Config", "PatchPipeline", "run"]

_LAZY = {
    "Config": ("config", "Config"),
    "PatchPipeline": ("preprocess", "PatchPipeline"),
    "run": ("engine", "run"),
}

if TYPE_CHECKING:  # for type checkers and editors only; never executed at runtime
    from .config import Config
    from .engine import run
    from .preprocess import PatchPipeline


def __getattr__(name: str):
    """Import the owning module on first attribute access (PEP 562)."""
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(f".{module_name}", __name__), attr)


def __dir__() -> list[str]:
    return sorted(__all__)
