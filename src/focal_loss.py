"""
Focal Loss for token classification.

Focal loss (Lin et al., 2017) dynamically down-weights easy, well-classified
examples so the model focuses its learning capacity on hard boundary tokens.
For BIO labelling this means common O tokens that are confidently classified
contribute almost nothing to the gradient, letting the model focus on the
ambiguous keyword/value boundary tokens.

FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

  γ=0  →  standard cross-entropy (no focusing)
  γ=2  →  well-classified examples downweighted by ~4× (recommended default)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class token classification.

    Parameters
    ----------
    gamma        : focusing parameter (0 = standard CE, 2 = recommended)
    weight       : per-class weight tensor (same as CrossEntropyLoss.weight)
    ignore_index : label value to ignore (default: -100, HuggingFace convention)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs  : (N, C) logits
        targets : (N,)  class indices, with ignore_index for masked positions
        """
        # Standard cross-entropy per token (unreduced)
        weight = self.weight.to(inputs.device) if self.weight is not None else None
        ce = F.cross_entropy(
            inputs,
            targets,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        # Mask ignored positions before computing focal weight
        valid = targets != self.ignore_index
        if not valid.any():
            return ce.sum() * 0.0

        p_t = torch.exp(-ce)  # probability of the correct class
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = (focal_weight * ce)[valid].mean()
        return loss
