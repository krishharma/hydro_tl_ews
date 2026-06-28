"""Loss functions for hydrological deep learning."""
from __future__ import annotations

import torch
from torch import nn


class NSELoss(nn.Module):
    r"""Differentiable Nash-Sutcliffe Efficiency loss (per-basin).

    Following Kratzert et al. (2019), the loss is the *negative* NSE so it can
    be minimized.  Per-basin variance ``σ_b^2`` is supplied so that high-flow
    basins do not dominate the gradient:

    .. math::
        L = \frac{1}{N}\sum_{b} \frac{(y_b - \hat{y}_b)^2}{(\sigma_b + \epsilon)^2}

    Equivalent to the basin-mean MSE divided by the squared basin standard
    deviation, yielding a unitless loss that promotes balanced multi-basin
    learning.
    """

    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                basin_std: torch.Tensor) -> torch.Tensor:
        squared_error = (y_pred.squeeze(-1) - y_true) ** 2
        weights = 1.0 / (basin_std + self.eps) ** 2
        return torch.mean(weights * squared_error)
