"""Segmentation objective: cross-entropy plus soft Tversky, with deep supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TverskyCELoss(nn.Module):
    """Cross-entropy + soft Tversky over foreground classes only.

    Tversky = TP / (TP + alpha*FP + beta*FN). Background is excluded from the Tversky mean
    on purpose: including it cancels the alpha/beta asymmetry, since a foreground FN is a
    background FP. Background stays supervised through CE.
    """

    def __init__(
        self,
        num_classes: int = 2,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.include_background = include_background
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(device=logits.device).long()
        ce = self.ce(logits, target)
        probs = torch.softmax(logits, dim=1)
        oh = F.one_hot(target, self.num_classes).movedim(-1, 1).float()
        dims = (0,) + tuple(range(2, logits.ndim))
        tp = (probs * oh).sum(dims)
        fp = (probs * (1.0 - oh)).sum(dims)
        fn = ((1.0 - probs) * oh).sum(dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        if not self.include_background:
            tversky = tversky[1:]
        return ce + (1.0 - tversky.mean())


class DeepSupervisionLoss(nn.Module):
    """nnU-Net deep supervision: weights [1, 1/2, 1/4, ...] normalised, coarsest level dropped."""

    def __init__(self, base_loss: nn.Module) -> None:
        super().__init__()
        self.base = base_loss

    def forward(self, logits_list, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(logits_list, (list, tuple)):
            return self.base(logits_list, target)
        n = len(logits_list)
        w = [1.0 / (2**i) for i in range(n)]
        w[-1] = 0.0
        s = sum(w)
        w = [x / s for x in w]
        total = 0.0
        for wi, logits in zip(w, logits_list):
            if wi == 0.0:
                continue
            if logits.shape[2:] == target.shape[1:]:
                t = target
            else:
                t = F.interpolate(target[:, None].float(), size=logits.shape[2:], mode="nearest")[:, 0].long()
            total = total + wi * self.base(logits, t)
        return total
