"""Regional Frequency Analysis (RFA) for extreme event thresholds.

Calculating site-specific percentiles from a 2-year warmup window biases
threshold estimates (e.g. a drought year masquerading as normal).  RFA
estimates extreme quantiles from the *full* CAMELS record, providing
stable Q5/Q95/Q99 references that the operational warmup period can use
without contaminating evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExtremeThresholds:
    q5: float
    q95: float
    q99: float


def regional_thresholds(streamflow: pd.Series,
                        years_required: int = 20) -> ExtremeThresholds:
    """Compute Q5/Q95/Q99 from the full available record."""
    s = streamflow.dropna()
    n_years = len(s) / 365.25
    if n_years < years_required:
        raise ValueError(
            f"At least {years_required} years required; got {n_years:.1f}.")
    return ExtremeThresholds(
        q5=float(np.quantile(s, 0.05)),
        q95=float(np.quantile(s, 0.95)),
        q99=float(np.quantile(s, 0.99)),
    )


def warning_labels(observed_flow: pd.Series,
                   thresholds: ExtremeThresholds,
                   kind: str = "flood",
                   percentile: str = "q95",
                   lead_times: tuple[int, ...] = (1, 3, 7)) -> pd.DataFrame:
    """Build binary early-warning labels at multiple lead times.

    A label at date *t* with lead-time *L* is 1 if any day in
    ``[t+1, t+L]`` exceeds (flood) or falls below (drought) the threshold.
    """
    if kind == "flood":
        thr = getattr(thresholds, percentile)
        cmp = lambda x: x >= thr
    elif kind == "drought":
        thr = thresholds.q5
        cmp = lambda x: x <= thr
    else:
        raise ValueError(f"Unknown kind: {kind}")

    out = pd.DataFrame(index=observed_flow.index)
    for L in lead_times:
        future = observed_flow.shift(-1).rolling(L).apply(
            lambda w: float(cmp(w).any()), raw=False
        )
        out[f"{kind}_{percentile}_lead{L}d"] = (future > 0).astype(float)
    return out


def predicted_warning_probabilities(predicted_flow: pd.Series,
                                    thresholds: ExtremeThresholds,
                                    kind: str = "flood",
                                    percentile: str = "q95",
                                    sigma: float | None = None,
                                    lead_times: tuple[int, ...] = (1, 3, 7)) -> pd.DataFrame:
    """Convert deterministic predictions to warning probabilities.

    A simple operational mapping: assume Gaussian residual std ``sigma``
    (default = 25% of the threshold) and integrate over the threshold for
    each lead-time max (flood) or min (drought).
    """
    from math import erf, sqrt
    sigma = sigma or 0.25 * abs(getattr(thresholds, percentile))
    if kind == "flood":
        thr = getattr(thresholds, percentile)
        prob_one_day = 0.5 * (1 - np.array([
            erf((thr - x) / (sigma * sqrt(2))) for x in predicted_flow.values
        ]))
    else:
        thr = thresholds.q5
        prob_one_day = 0.5 * (1 + np.array([
            erf((thr - x) / (sigma * sqrt(2))) for x in predicted_flow.values
        ]))
    p = pd.Series(prob_one_day, index=predicted_flow.index)
    out = pd.DataFrame(index=predicted_flow.index)
    for L in lead_times:
        # P(any day in window has event) = 1 - prod(1 - P_t) over the window
        log1m = np.log1p(-np.clip(p, 1e-9, 1 - 1e-9))
        any_event = 1.0 - np.exp(log1m.rolling(L, min_periods=1).sum())
        out[f"{kind}_{percentile}_lead{L}d"] = any_event.shift(-1).fillna(0.0)
    return out
