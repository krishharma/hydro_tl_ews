"""Rolling-origin (walk-forward) backtester.

Simulates real-time operational forecasting by repeatedly fine-tuning on an
expanding training window and producing predictions for the next horizon.
Eliminates the temporal leakage that plagues random-split evaluations of
autocorrelated hydrological time series.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..data.camels import BasinData, DYNAMIC_FEATURES, STATIC_ATTRIBUTES
from ..data.preprocessing import (
    Normalizer,
    StaticNormalizer,
    make_sequences,
    quality_control,
)
from ..models.ealstm import EALSTM
from ..utils.logging import get_logger
from .trainer import Trainer
from .transfer import FineTuneConfig, fine_tune_conservative

log = get_logger(__name__)


@dataclass
class WalkForwardConfig:
    initial_train_end: str          # e.g. "2016-12-31" (warmup end)
    eval_end: str                   # e.g. "2020-12-31"
    refit_every_days: int = 90      # full fine-tune cadence
    online_bias_correction: bool = True
    sequence_length: int = 365
    batch_size: int = 256
    fine_tune_cfg: FineTuneConfig = field(default_factory=FineTuneConfig)
    reset_weights_each_refit: bool = True


@dataclass
class WalkForwardResult:
    dates: pd.DatetimeIndex
    observed: np.ndarray
    predicted: np.ndarray
    bias_corrections: np.ndarray
    refit_dates: list[pd.Timestamp]


def _build_loader(forcings: pd.DataFrame, streamflow: pd.Series,
                  attrs: pd.Series, dyn_norm: Normalizer,
                  static_norm: StaticNormalizer, cfg: WalkForwardConfig,
                  shuffle: bool = True) -> DataLoader | None:
    f, q = quality_control(forcings, streamflow)
    f_norm = dyn_norm.transform(f)[DYNAMIC_FEATURES]
    X, y = make_sequences(f_norm.to_numpy(), q.to_numpy(),
                          sequence_length=cfg.sequence_length)
    if len(X) == 0:
        return None
    statics = static_norm.transform(attrs.to_frame().T).reindex(
        columns=STATIC_ATTRIBUTES).to_numpy().astype(np.float32)[0]
    S = np.tile(statics, (len(X), 1))
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(S),
        torch.from_numpy(y),
    )
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle)


@torch.no_grad()
def _predict_window(model: EALSTM, loader: DataLoader,
                    device: str = "cpu") -> np.ndarray:
    model.eval()
    out = []
    for X, S, _ in loader:
        out.append(model(X.to(device), S.to(device)).squeeze(-1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def walk_forward(
    model: EALSTM,
    target_basin: BasinData,
    dyn_norm: Normalizer,
    static_norm: StaticNormalizer,
    cfg: WalkForwardConfig,
    device: str = "cpu",
    refit_fn: Callable | None = None,
) -> WalkForwardResult:
    """Run a rolling-origin backtest on the target basin.

    Parameters
    ----------
    model:
        Pre-trained EA-LSTM.  The function makes a deepcopy each refit so the
        caller's weights remain unchanged across calls.
    target_basin:
        BasinData with daily forcings and streamflow covering both warmup
        and evaluation periods.
    refit_fn:
        Optional override for the fine-tuning routine; defaults to
        :func:`fine_tune_conservative` (Approach A).
    """
    refit_fn = refit_fn or fine_tune_conservative

    # Establish full date range
    full_dates = target_basin.streamflow.dropna().index
    full_dates = full_dates.intersection(target_basin.forcings.dropna().index)
    full_dates = pd.DatetimeIndex(full_dates)

    init_end = pd.Timestamp(cfg.initial_train_end)
    eval_end = pd.Timestamp(cfg.eval_end)

    refit_dates: list[pd.Timestamp] = []
    bias_correction = 0.0
    all_dates, all_obs, all_pred, all_bias = [], [], [], []

    # Deepcopy original model once to ensure no side-effects on caller's model object
    base_model = copy.deepcopy(model)
    active_model = copy.deepcopy(base_model)

    cur_start = init_end + pd.Timedelta(days=1)
    while cur_start <= eval_end:
        chunk_end = min(cur_start + pd.Timedelta(days=cfg.refit_every_days - 1),
                        eval_end)
        # ---- Refit on data up to cur_start - 1 ----------------------------
        train_forcings = target_basin.forcings.loc[:cur_start - pd.Timedelta(days=1)]
        train_flow = target_basin.streamflow.loc[:cur_start - pd.Timedelta(days=1)]
        train_loader = _build_loader(
            train_forcings, train_flow, target_basin.attributes,
            dyn_norm, static_norm, cfg, shuffle=True,
        )
        if train_loader is not None:
            log.info("Refit at %s | window 0..%s", cur_start.date(),
                     (cur_start - pd.Timedelta(days=1)).date())
            if cfg.reset_weights_each_refit:
                # Reset to pristine pre-trained state to avoid weight drift
                active_model = copy.deepcopy(base_model)
            refit_fn(active_model, train_loader, None, cfg.fine_tune_cfg, device=device)
            refit_dates.append(cur_start)

        # ---- Predict next chunk -----------------------------------------
        eval_forcings = target_basin.forcings.loc[
            cur_start - pd.Timedelta(days=cfg.sequence_length - 1):chunk_end
        ]
        eval_flow = target_basin.streamflow.loc[
            cur_start - pd.Timedelta(days=cfg.sequence_length - 1):chunk_end
        ]
        eval_loader = _build_loader(
            eval_forcings, eval_flow, target_basin.attributes,
            dyn_norm, static_norm, cfg, shuffle=False,
        )
        if eval_loader is None:
            cur_start = chunk_end + pd.Timedelta(days=1)
            continue
        preds = _predict_window(active_model, eval_loader, device=device)
        # Sequence target dates correspond to last day of each window,
        # i.e. the days between cur_start and chunk_end inclusive.
        target_dates = pd.date_range(cur_start, periods=len(preds), freq="D")
        target_dates = target_dates[target_dates <= chunk_end]
        preds = preds[: len(target_dates)]
        obs = target_basin.streamflow.reindex(target_dates).to_numpy()

        if cfg.online_bias_correction:
            preds = preds + bias_correction
            valid = ~np.isnan(obs)
            if valid.any():
                bias_correction = float(np.nanmean(obs[valid] - preds[valid] + bias_correction))
        all_dates.extend(target_dates)
        all_obs.extend(obs.tolist())
        all_pred.extend(preds.tolist())
        all_bias.extend([bias_correction] * len(target_dates))

        cur_start = chunk_end + pd.Timedelta(days=1)

    return WalkForwardResult(
        dates=pd.DatetimeIndex(all_dates),
        observed=np.array(all_obs),
        predicted=np.array(all_pred),
        bias_corrections=np.array(all_bias),
        refit_dates=refit_dates,
    )
