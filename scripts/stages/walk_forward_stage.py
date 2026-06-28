"""Phase 3 — Rolling-origin walk-forward evaluation on the target basin."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hydro_tl_ews.data.camels import CamelsDataset, STATIC_ATTRIBUTES
from hydro_tl_ews.data.preprocessing import Normalizer, StaticNormalizer
from hydro_tl_ews.evaluation.extreme_thresholds import (
    predicted_warning_probabilities,
    regional_thresholds,
    warning_labels,
)
from hydro_tl_ews.evaluation.metrics import (
    auc_roc,
    brier_score,
    f1_at_threshold,
    kge,
    nse,
    pbias,
)
from hydro_tl_ews.training.trainer import Trainer
from hydro_tl_ews.training.transfer import FineTuneConfig
from hydro_tl_ews.training.walk_forward import WalkForwardConfig, walk_forward
from hydro_tl_ews.utils.config import ExperimentConfig
from hydro_tl_ews.utils.logging import get_logger

log = get_logger(__name__)


def run_walk_forward(cfg: ExperimentConfig) -> None:
    ds = CamelsDataset(cfg.data["camels_root"])
    target_id = cfg.data["target_basin"]
    target = ds.load_basin(target_id)
    attrs = ds.load_attributes()

    full_period = cfg.data.get("full_period")
    if full_period:
        target.forcings = target.forcings.loc[full_period[0]:full_period[1]]
        target.streamflow = target.streamflow.loc[full_period[0]:full_period[1]]

    init_end = cfg.walk_forward["initial_train_end"]
    dyn_norm = Normalizer.fit(target.forcings.loc[:init_end])
    static_norm = StaticNormalizer.fit(attrs.drop(index=target_id).loc[:, STATIC_ATTRIBUTES])

    ckpt = cfg.model.get("pretrained_checkpoint")
    if not ckpt:
        raise ValueError("walk_forward stage requires model.pretrained_checkpoint")
    model = Trainer.load_model(ckpt)

    ft = cfg.walk_forward.get("fine_tune", {})
    wf_cfg = WalkForwardConfig(
        initial_train_end=init_end,
        eval_end=cfg.walk_forward["eval_end"],
        refit_every_days=cfg.walk_forward.get("refit_every_days", 90),
        online_bias_correction=cfg.walk_forward.get("online_bias_correction", True),
        sequence_length=cfg.data.get("sequence_length", 365),
        batch_size=64,
        fine_tune_cfg=FineTuneConfig(
            head_lr=ft.get("head_lr", 1e-3),
            lstm_lr=ft.get("lstm_lr", 1e-5),
            epochs_head_only=ft.get("epochs_head_only", 3),
            epochs_progressive=ft.get("epochs_progressive", 0),
            patience=ft.get("patience", 2),
            unfreeze_fraction=ft.get("unfreeze_fraction", 0.0),
        ),
    )
    result = walk_forward(model, target, dyn_norm, static_norm, wf_cfg)

    rfa = regional_thresholds(target.streamflow.loc[:init_end], years_required=20)
    obs_s = pd.Series(result.observed, index=result.dates)
    pred_s = pd.Series(result.predicted, index=result.dates)
    flood_labels = warning_labels(obs_s, rfa, kind="flood", percentile="q95")
    flood_probs = predicted_warning_probabilities(pred_s, rfa, kind="flood",
                                                  percentile="q95")

    metrics = {
        "continuous": {
            "NSE": nse(result.observed, result.predicted),
            "KGE": kge(result.observed, result.predicted),
            "PBIAS": pbias(result.observed, result.predicted),
        },
        "early_warning": {
            col: {
                "AUC": auc_roc(flood_labels[col].to_numpy(),
                               flood_probs[col].to_numpy()),
                "F1@0.5": f1_at_threshold(flood_labels[col].to_numpy(),
                                          flood_probs[col].to_numpy(), 0.5),
                "Brier": brier_score(flood_labels[col].to_numpy(),
                                     flood_probs[col].to_numpy()),
            }
            for col in flood_labels.columns
        },
        "n_predictions": int(len(result.predicted)),
        "n_refits": int(len(result.refit_dates)),
    }
    out_metrics = Path(cfg.output.get("metrics_path",
                                      "results/walk_forward_metrics.json"))
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2, default=float))

    out_df = pd.DataFrame({
        "observed": result.observed,
        "predicted": result.predicted,
        "bias_correction": result.bias_corrections,
    }, index=result.dates)
    out_path = Path(cfg.output.get("results_path", "results/walk_forward.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path) if out_path.suffix == ".parquet" else out_df.to_csv(out_path)
    log.info("Walk-forward complete | metrics: %s", metrics)
