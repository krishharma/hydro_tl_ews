"""SHAP-based explainability for EA-LSTM streamflow predictions.

Uses :class:`shap.DeepExplainer` (or the gradient explainer fallback) to
attribute each meteorological forcing and static catchment attribute's
contribution to the final-day streamflow prediction.  Designed for both
global summary plots and per-event attribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SHAPResult:
    feature_names: list[str]
    shap_values: np.ndarray  # (N, F)
    base_value: float


def _ealstm_predict_fn(model, statics_tensor):
    """Closure that exposes the EA-LSTM as ``f(forcings) -> q_hat`` for SHAP.

    SHAP varies one input axis (the forcings tensor) while statics remain
    fixed for a given basin.
    """
    import torch

    @torch.no_grad()
    def fn(forcings_np: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(forcings_np.astype(np.float32))
        if x.ndim == 2:  # (N, L*F) -> (N, L, F)
            n, lf = x.shape
            f = statics_tensor.shape[-1]
            # Caller is expected to reshape; raise a clearer error otherwise
            raise ValueError("forcings must be 3D (N, L, F) for SHAP wrapper.")
        s = statics_tensor.expand(x.shape[0], -1)
        out = model(x.to(s.device), s).squeeze(-1).cpu().numpy()
        return out

    return fn


def explain_predictions(
    model,
    background_X: np.ndarray,
    background_S: np.ndarray,
    samples_X: np.ndarray,
    samples_S: np.ndarray,
    feature_names: list[str],
    nsamples: int = 200,
    device: str = "cpu",
) -> Optional[SHAPResult]:
    """Compute SHAP values for a batch of EA-LSTM predictions.

    Reduces each (L, F_dyn) sequence to a per-feature *mean* contribution by
    averaging gradients over the sequence axis.  This keeps the output
    comparable to the static feature dimension and tractable for plotting.
    """
    try:
        import shap
        import torch
    except ImportError as e:
        log.warning("shap/torch missing: %s", e)
        return None

    background_X_t = torch.from_numpy(background_X.astype(np.float32)).to(device)
    background_S_t = torch.from_numpy(background_S.astype(np.float32)).to(device)
    samples_X_t = torch.from_numpy(samples_X.astype(np.float32)).to(device)
    samples_S_t = torch.from_numpy(samples_S.astype(np.float32)).to(device)

    class WrappedModel(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, X_and_S):
            # X_and_S is concatenated: (B, L*F + F_static)
            ld = background_X.shape[1] * background_X.shape[2]
            X = X_and_S[:, :ld].view(-1, background_X.shape[1], background_X.shape[2])
            S = X_and_S[:, ld:]
            return self.m(X, S)

    flat_bg = torch.cat([background_X_t.flatten(start_dim=1), background_S_t], dim=1)
    flat_smp = torch.cat([samples_X_t.flatten(start_dim=1), samples_S_t], dim=1)

    explainer = shap.GradientExplainer(WrappedModel(model), flat_bg)
    sv = explainer.shap_values(flat_smp, nsamples=nsamples)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)

    # Reduce sequence axis to per-feature contributions
    L, F = background_X.shape[1], background_X.shape[2]
    seq_part = sv[:, : L * F].reshape(sv.shape[0], L, F).mean(axis=1)
    static_part = sv[:, L * F :]
    combined = np.concatenate([seq_part, static_part], axis=1)
    return SHAPResult(
        feature_names=feature_names,
        shap_values=combined,
        base_value=0.0,
    )


def global_importance(result: SHAPResult) -> "pd.Series":
    """Mean |SHAP| per feature, sorted descending."""
    import pandas as pd
    importance = np.abs(result.shap_values).mean(axis=0)
    return pd.Series(importance, index=result.feature_names).sort_values(ascending=False)
