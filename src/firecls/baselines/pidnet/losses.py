"""PIDNet's four-term loss with independent sigmoid channels (fire, smoke).

Following upstream ``FullModel`` and Pesonen et al.:

    L = l0 * BCE(P-branch) + l1 * weighted-BCE(D-branch, edges) + l2 * BCE(main)
        + l3 * BCE(main | sigmoid(D) > t)            with l = (0.4, 20, 1, 1), t = 0.8

Upstream uses (OHEM) softmax cross-entropy for single-label Cityscapes; Pesonen et al. segment
one class with BCE. Our two-channel multi-label target uses BCE per channel, which reduces to
the paper's loss for a single smoke channel. All logits are upsampled to label resolution first.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from firecls.baselines.pidnet.model import upsample_like


def weighted_bce(boundary_logits: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
    """Upstream ``criterion.weighted_bce``: positives weighted by the negative fraction and vice versa."""
    logits = boundary_logits.reshape(-1)
    target = edge.reshape(-1).to(logits.dtype)
    positives = target == 1
    pos_num = positives.sum().to(logits.dtype)
    total = torch.tensor(float(target.numel()), device=logits.device, dtype=logits.dtype)
    weight = torch.where(positives, (total - pos_num) / total, pos_num / total)
    return F.binary_cross_entropy_with_logits(logits, target, weight, reduction="mean")


class PidnetLoss(torch.nn.Module):
    def __init__(
        self,
        lambda_p: float = 0.4,
        lambda_boundary: float = 20.0,
        lambda_main: float = 1.0,
        lambda_bas: float = 1.0,
        boundary_threshold: float = 0.8,
    ) -> None:
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_boundary = lambda_boundary
        self.lambda_main = lambda_main
        self.lambda_bas = lambda_bas
        self.boundary_threshold = boundary_threshold

    def forward(self, outputs, mask: torch.Tensor, edge: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """``outputs = [P, main, D]`` from ``PIDNet(augment=True)``; ``mask [B,2,H,W]``, ``edge [B,H,W]``."""
        size = mask.shape[-2:]
        p_logits, main_logits, d_logits = (upsample_like(o.float(), size) for o in outputs)
        mask = mask.float()
        loss_p = F.binary_cross_entropy_with_logits(p_logits, mask)
        loss_main = F.binary_cross_entropy_with_logits(main_logits, mask)
        loss_boundary = weighted_bce(d_logits[:, 0], edge)

        # Boundary-awareness (BAS) term: main-output BCE only where the D-branch sees a boundary.
        selected = (torch.sigmoid(d_logits) > self.boundary_threshold).expand_as(main_logits)
        per_pixel = F.binary_cross_entropy_with_logits(main_logits, mask, reduction="none")
        count = selected.sum()
        loss_bas = (per_pixel * selected).sum() / count.clamp(min=1)

        total = (
            self.lambda_p * loss_p
            + self.lambda_boundary * loss_boundary
            + self.lambda_main * loss_main
            + self.lambda_bas * loss_bas
        )
        parts = {
            "loss_p": float(loss_p.detach()),
            "loss_boundary": float(loss_boundary.detach()),
            "loss_main": float(loss_main.detach()),
            "loss_bas": float(loss_bas.detach()),
        }
        return total, parts
