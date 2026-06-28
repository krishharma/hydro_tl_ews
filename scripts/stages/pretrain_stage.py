"""Phase 1 — Regional pre-training on CAMELS-US."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hydro_tl_ews.data.camels import (
    CamelsDataset,
    DYNAMIC_FEATURES,
    STATIC_ATTRIBUTES,
)
from hydro_tl_ews.data.datasets import MultiBasinSequenceDataset
from hydro_tl_ews.data.preprocessing import Normalizer, StaticNormalizer
from hydro_tl_ews.models.ealstm import EALSTM, EALSTMConfig
from hydro_tl_ews.training.trainer import Trainer
from hydro_tl_ews.utils.config import ExperimentConfig
from hydro_tl_ews.utils.logging import get_logger

log = get_logger(__name__)


def run_pretrain(cfg: ExperimentConfig) -> None:
    ds = CamelsDataset(cfg.data["camels_root"])
    attrs = ds.load_attributes()
    target_basin = cfg.data.get("target_basin")
    donors = [b for b in attrs.index if b != target_basin]
    log.info("Pre-training on %d basins (target excluded: %s)", len(donors), target_basin)
    basins = ds.load_basins(donors)

    pretrain_period = tuple(cfg.data["pretrain_period"])
    val_period = tuple(cfg.data["validation_period"])
    seq_len = cfg.data.get("sequence_length", 365)

    # Fit normalizers on the pre-training period only
    forc_train = pd.concat([b.forcings.loc[pretrain_period[0]:pretrain_period[1]]
                            for b in basins.values()])
    dyn_norm = Normalizer.fit(forc_train)
    static_norm = StaticNormalizer.fit(attrs.loc[donors, STATIC_ATTRIBUTES])

    train_ds = MultiBasinSequenceDataset(basins, pretrain_period, dyn_norm,
                                         static_norm, sequence_length=seq_len)
    val_ds = MultiBasinSequenceDataset(basins, val_period, dyn_norm, static_norm,
                                       sequence_length=seq_len)

    bs = cfg.training.get("batch_size", 256)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=cfg.data.get("num_workers", 0))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=cfg.data.get("num_workers", 0))

    model_cfg = EALSTMConfig(
        dynamic_input_size=len(DYNAMIC_FEATURES),
        static_input_size=len(STATIC_ATTRIBUTES),
        hidden_size=cfg.model.get("hidden_size", 256),
        dropout=cfg.model.get("dropout", 0.4),
        initial_forget_bias=cfg.model.get("initial_forget_bias", 3.0),
    )
    model = EALSTM(model_cfg)
    trainer = Trainer(
        model=model, mode="pretrain",
        head_lr=cfg.training.get("learning_rate", 1e-3),
        weight_decay=cfg.training.get("weight_decay", 0.0),
        clip_grad_norm=cfg.training.get("clip_grad_norm", 1.0),
    )
    state = trainer.fit(
        train_loader, val_loader,
        epochs=cfg.training.get("epochs", 50),
        patience=cfg.training.get("patience", 10),
        basin_std=float(np.nanstd(train_ds.y) if len(train_ds.y) else 1.0),
        checkpoint_path=cfg.output.get("checkpoint_path",
                                       "results/checkpoints/pretrain.pt"),
    )

    history_path = Path(cfg.output.get("history_path",
                                       "results/history/pretrain.json"))
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(state.history, indent=2))
    log.info("Pre-training complete | best val=%.4f at epoch %d",
             state.best_val_loss, state.best_epoch)
