"""End-to-end smoke pipeline.

Trains a tiny EA-LSTM on synthetic CAMELS-like data, fine-tunes it on the
most snow-dominated basin, runs walk-forward evaluation, computes warning
metrics, and emits a results bundle suitable for CI and the paper figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hydro_tl_ews.data.camels import DYNAMIC_FEATURES, STATIC_ATTRIBUTES
from hydro_tl_ews.data.datasets import MultiBasinSequenceDataset
from hydro_tl_ews.data.preprocessing import Normalizer, StaticNormalizer
from hydro_tl_ews.data.synthetic_camels import SyntheticCamels
from hydro_tl_ews.evaluation.extreme_thresholds import (
    predicted_warning_probabilities,
    regional_thresholds,
    warning_labels,
)
from hydro_tl_ews.evaluation.metrics import (
    auc_roc,
    brier_score,
    compute_all,
    f1_at_threshold,
    kge,
    nse,
    pbias,
    reliability_curve,
)
from hydro_tl_ews.models.ealstm import EALSTM, EALSTMConfig
from hydro_tl_ews.training.trainer import Trainer
from hydro_tl_ews.training.transfer import (
    FineTuneConfig,
    fine_tune_conservative,
    train_local_baseline,
)
from hydro_tl_ews.training.walk_forward import (
    WalkForwardConfig,
    walk_forward,
)
from hydro_tl_ews.utils.config import ExperimentConfig
from hydro_tl_ews.utils.logging import get_logger
from hydro_tl_ews.utils.seed import set_global_seed

log = get_logger(__name__)


def _build_dyn_normalizer(basins, period):
    frames = []
    for bd in basins.values():
        f = bd.forcings.loc[period[0]:period[1]]
        frames.append(f)
    return Normalizer.fit(pd.concat(frames))


def _build_loader(ds, batch_size, shuffle):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def run_smoke(cfg: ExperimentConfig) -> dict:
    set_global_seed(cfg.seed)
    out_dir = Path(cfg.output.get("results_dir", "results/smoke"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    ds = SyntheticCamels(
        n_basins=cfg.data.get("n_basins", 12),
        n_days=cfg.data.get("n_days", 3650),
        snow_fraction=cfg.data.get("snow_fraction", 0.4),
        seed=cfg.seed,
    )
    attrs = ds.load_attributes()
    target_id = attrs["frac_snow"].idxmax()
    log.info("Selected target basin (highest frac_snow): %s", target_id)

    donor_ids = [b for b in ds.basin_ids if b != target_id]
    donor_basins = {b: ds.basins[b] for b in donor_ids}
    target_basin = ds.basins[target_id]

    # Train/val period split (last 25% of donor history reserved for val)
    seq_len = cfg.data.get("sequence_length", 90)
    full_dates = donor_basins[donor_ids[0]].forcings.index
    train_end = full_dates[int(0.75 * len(full_dates))]
    pretrain_period = (str(full_dates[0].date()), str(train_end.date()))
    val_period = (str((train_end + pd.Timedelta(days=1)).date()),
                  str(full_dates[-1].date()))

    dyn_norm = _build_dyn_normalizer(donor_basins, pretrain_period)
    static_norm = StaticNormalizer.fit(attrs.loc[donor_ids])

    train_ds = MultiBasinSequenceDataset(
        donor_basins, pretrain_period, dyn_norm, static_norm,
        sequence_length=seq_len,
    )
    val_ds = MultiBasinSequenceDataset(
        donor_basins, val_period, dyn_norm, static_norm,
        sequence_length=seq_len,
    )
    log.info("Pre-train dataset: %d samples | val: %d samples", len(train_ds), len(val_ds))

    train_loader = _build_loader(train_ds, cfg.training.get("batch_size", 64), True)
    val_loader = _build_loader(val_ds, cfg.training.get("batch_size", 64), False)

    # --------------------------------------------------------------- model
    model_cfg = EALSTMConfig(
        dynamic_input_size=len(DYNAMIC_FEATURES),
        static_input_size=len(STATIC_ATTRIBUTES),
        hidden_size=cfg.model.get("hidden_size", 32),
        dropout=cfg.model.get("dropout", 0.2),
    )
    model = EALSTM(model_cfg)

    # --------------------------------------------------------------- pretrain
    pretrainer = Trainer(model=model, mode="pretrain",
                         head_lr=cfg.training.get("head_lr", 1e-3))
    pretrainer.fit(
        train_loader, val_loader,
        epochs=cfg.training.get("pretrain_epochs", 4),
        patience=cfg.training.get("pretrain_patience", 3),
        basin_std=float(np.nanstd(train_ds.y) if len(train_ds.y) else 1.0),
        checkpoint_path=out_dir / "pretrain.pt",
    )

    # ----------------------------------------------------- zero-shot eval
    target_dyn_norm = dyn_norm
    target_static_norm = static_norm

    target_basins = {target_id: target_basin}
    full_target_period = (str(target_basin.forcings.index[0].date()),
                          str(target_basin.forcings.index[-1].date()))
    target_full_ds = MultiBasinSequenceDataset(
        target_basins, full_target_period, target_dyn_norm, target_static_norm,
        sequence_length=seq_len,
    )
    target_full_loader = _build_loader(target_full_ds, 256, False)
    zs_pred, zs_obs = pretrainer.predict(target_full_loader)
    zs_metrics = {"NSE": nse(zs_obs, zs_pred), "KGE": kge(zs_obs, zs_pred),
                  "PBIAS": pbias(zs_obs, zs_pred)}
    log.info("Zero-shot metrics on target: %s", zs_metrics)

    # ----------------------------------------------------- conservative FT
    warmup_period = (
        str(target_basin.forcings.index[int(0.6 * len(target_basin.forcings))].date()),
        str(target_basin.forcings.index[int(0.7 * len(target_basin.forcings))].date()),
    )
    log.info("Warmup window for fine-tune: %s", warmup_period)
    warmup_ds = MultiBasinSequenceDataset(
        target_basins, warmup_period, target_dyn_norm, target_static_norm,
        sequence_length=seq_len,
    )
    warmup_loader = _build_loader(warmup_ds, 32, True)
    finetune_cfg = FineTuneConfig(
        head_lr=cfg.training.get("head_lr", 1e-3),
        lstm_lr=cfg.training.get("lstm_lr", 1e-5),
        epochs_head_only=cfg.training.get("finetune_epochs", 3),
        epochs_progressive=0,
        patience=cfg.training.get("finetune_patience", 2),
        unfreeze_fraction=0.0,
    )
    fine_tune_conservative(model, warmup_loader, None, finetune_cfg)

    # Capture conservative-FT predictions for later metrics
    ft_trainer = Trainer(model=model, mode="conservative",
                         head_lr=finetune_cfg.head_lr)
    ft_pred, ft_obs = ft_trainer.predict(target_full_loader)
    ft_metrics = {"NSE": nse(ft_obs, ft_pred), "KGE": kge(ft_obs, ft_pred),
                  "PBIAS": pbias(ft_obs, ft_pred)}
    log.info("Conservative FT metrics on target: %s", ft_metrics)

    # ----------------------------------------------------- local baseline
    local_model = EALSTM(model_cfg)
    local_trainer = Trainer(model=local_model, mode="local",
                            head_lr=cfg.training.get("head_lr", 1e-3))
    local_trainer.fit(
        warmup_loader, None,
        epochs=cfg.training.get("local_baseline_epochs", 6),
        patience=3,
    )
    lb_pred, lb_obs = local_trainer.predict(target_full_loader)
    lb_metrics = {"NSE": nse(lb_obs, lb_pred), "KGE": kge(lb_obs, lb_pred),
                  "PBIAS": pbias(lb_obs, lb_pred)}
    log.info("Local baseline metrics on target: %s", lb_metrics)

    # ----------------------------------------------------- walk-forward
    wf_cfg = WalkForwardConfig(
        initial_train_end=cfg.walk_forward.get("initial_train_end", "1996-12-31"),
        eval_end=cfg.walk_forward.get("eval_end", "1999-12-31"),
        refit_every_days=cfg.walk_forward.get("refit_every_days", 90),
        online_bias_correction=cfg.walk_forward.get("online_bias_correction", True),
        sequence_length=seq_len,
        batch_size=64,
        fine_tune_cfg=finetune_cfg,
    )
    # Reuse the conservatively-tuned model
    wf_result = walk_forward(model, target_basin, target_dyn_norm,
                             target_static_norm, wf_cfg)
    log.info("Walk-forward predictions: n=%d", len(wf_result.predicted))

    # ----------------------------------------------------- thresholds & EW
    rfa = regional_thresholds(target_basin.streamflow.loc[:wf_cfg.initial_train_end], years_required=5)
    log.info("RFA thresholds: Q5=%.3f Q95=%.3f Q99=%.3f", rfa.q5, rfa.q95, rfa.q99)

    obs_series = pd.Series(wf_result.observed, index=wf_result.dates)
    pred_series = pd.Series(wf_result.predicted, index=wf_result.dates)

    flood_labels = warning_labels(obs_series, rfa, kind="flood",
                                  percentile="q95")
    flood_probs = predicted_warning_probabilities(
        pred_series, rfa, kind="flood", percentile="q95")

    ew_metrics = {}
    for col in flood_labels.columns:
        ew_metrics[col] = {
            "AUC": auc_roc(flood_labels[col].to_numpy(), flood_probs[col].to_numpy()),
            "F1@0.5": f1_at_threshold(flood_labels[col].to_numpy(),
                                      flood_probs[col].to_numpy(), 0.5),
            "Brier": brier_score(flood_labels[col].to_numpy(),
                                 flood_probs[col].to_numpy()),
        }

    # ----------------------------------------------------- save artifacts
    pd.DataFrame({"observed": wf_result.observed,
                  "predicted": wf_result.predicted,
                  "bias_correction": wf_result.bias_corrections},
                 index=wf_result.dates).to_csv(out_dir / "walk_forward.csv")
    pd.concat([flood_labels, flood_probs.add_suffix("_prob")], axis=1).to_csv(
        out_dir / "warnings.csv")

    summary = {
        "target_basin": str(target_id),
        "thresholds": {"q5": rfa.q5, "q95": rfa.q95, "q99": rfa.q99},
        "metrics": {
            "zero_shot": zs_metrics,
            "fine_tune_conservative": ft_metrics,
            "local_baseline": lb_metrics,
            "walk_forward": {
                "NSE": nse(wf_result.observed, wf_result.predicted),
                "KGE": kge(wf_result.observed, wf_result.predicted),
                "PBIAS": pbias(wf_result.observed, wf_result.predicted),
            },
            "early_warning": ew_metrics,
        },
        "n_predictions": int(len(wf_result.predicted)),
        "n_refits": int(len(wf_result.refit_dates)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # Reliability bins for plotting
    rel_centers, rel_freq, rel_counts = reliability_curve(
        flood_labels["flood_q95_lead3d"].to_numpy(),
        flood_probs["flood_q95_lead3d"].to_numpy(),
        n_bins=8,
    )
    pd.DataFrame({"bin_center": rel_centers, "observed_freq": rel_freq,
                  "count": rel_counts}).to_csv(out_dir / "reliability_lead3.csv",
                                               index=False)

    log.info("Smoke summary: %s", json.dumps(summary, indent=2, default=float))
    return summary
