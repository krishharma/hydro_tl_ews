"""High-level transfer-learning orchestration.

Wraps :class:`Trainer` with the multi-phase fine-tuning recipes described in
the paper: conservative (Approach A) and progressive unfreezing (Approach B).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from ..models.ealstm import EALSTM
from ..utils.logging import get_logger
from .trainer import Trainer, TrainState

log = get_logger(__name__)


@dataclass
class FineTuneConfig:
    head_lr: float = 1e-3
    lstm_lr: float = 1e-5
    epochs_head_only: int = 5
    epochs_progressive: int = 5
    patience: int = 3
    unfreeze_fraction: float = 0.25
    weight_decay: float = 0.0


def fine_tune_conservative(
    model: EALSTM,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: FineTuneConfig,
    device: str = "cpu",
) -> TrainState:
    """Approach A — freeze the LSTM cell and train only the dense head."""
    log.info("Approach A: conservative fine-tune (head only)")
    trainer = Trainer(
        model=model,
        device=device,
        mode="conservative",
        head_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
    )
    return trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs_head_only,
        patience=cfg.patience,
    )


def fine_tune_progressive(
    model: EALSTM,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: FineTuneConfig,
    device: str = "cpu",
) -> TrainState:
    """Approach B — phase 1 trains the head only, phase 2 also unfreezes the
    last ``unfreeze_fraction`` of the LSTM with a 100× smaller LR.
    """
    log.info("Approach B: progressive unfreezing fine-tune")
    # Phase 1: head only
    phase1 = Trainer(
        model=model, device=device, mode="conservative",
        head_lr=cfg.head_lr, weight_decay=cfg.weight_decay,
    )
    phase1.fit(train_loader, val_loader,
               epochs=cfg.epochs_head_only, patience=cfg.patience)

    # Phase 2: unfreeze last fraction with differential LRs
    phase2 = Trainer(
        model=model, device=device, mode="progressive",
        head_lr=cfg.head_lr, lstm_lr=cfg.lstm_lr,
        unfreeze_fraction=cfg.unfreeze_fraction,
        weight_decay=cfg.weight_decay,
    )
    state = phase2.fit(train_loader, val_loader,
                       epochs=cfg.epochs_progressive, patience=cfg.patience)
    return state


def train_local_baseline(
    model: EALSTM,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int = 50,
    patience: int = 10,
    head_lr: float = 1e-3,
    device: str = "cpu",
) -> TrainState:
    """Train an EA-LSTM from scratch on the limited target-basin data."""
    trainer = Trainer(model=model, device=device, mode="local", head_lr=head_lr)
    return trainer.fit(train_loader, val_loader, epochs=epochs, patience=patience)
