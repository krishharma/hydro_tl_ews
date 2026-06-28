"""PyTorch ``Dataset`` wrappers for multi-basin EA-LSTM training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # torch optional during static analysis
    torch = None
    Dataset = object  # type: ignore[misc,assignment]

from .camels import BasinData, DYNAMIC_FEATURES, STATIC_ATTRIBUTES
from .preprocessing import Normalizer, StaticNormalizer, make_sequences, quality_control


@dataclass
class BasinSample:
    basin_id: str
    forcings: np.ndarray  # (L, F_dyn)
    statics: np.ndarray   # (F_static,)
    target: float


class MultiBasinSequenceDataset(Dataset):
    """Concatenates sliding-window samples from many basins for EA-LSTM training.

    Each item yields ``(forcings (L, F_dyn), statics (F_static,), target)`` —
    static attributes are *not* repeated along time inside the dataset; the
    EA-LSTM model handles that internally via its embedding gate.
    """

    def __init__(
        self,
        basins: Dict[str, BasinData],
        period: tuple[str, str],
        dyn_normalizer: Normalizer,
        static_normalizer: StaticNormalizer,
        sequence_length: int = 365,
    ):
        self.basin_ids: List[str] = list(basins.keys())
        self.sequence_length = sequence_length
        self.dyn_normalizer = dyn_normalizer
        self.static_normalizer = static_normalizer

        all_X, all_static, all_y, all_basin = [], [], [], []
        start, end = period
        for bid, bd in basins.items():
            f, q = quality_control(bd.forcings, bd.streamflow)
            f = f.loc[start:end]
            q = q.loc[start:end]
            if len(f) == 0:
                continue
            f_norm = dyn_normalizer.transform(f)[DYNAMIC_FEATURES]
            X, y = make_sequences(f_norm.to_numpy(), q.to_numpy(),
                                  sequence_length=sequence_length)
            if len(X) == 0:
                continue
            statics = static_normalizer.transform(
                bd.attributes.to_frame().T
            ).reindex(columns=STATIC_ATTRIBUTES).to_numpy().astype(np.float32)[0]
            all_X.append(X)
            all_static.append(np.tile(statics, (len(X), 1)))
            all_y.append(y)
            all_basin.extend([bid] * len(X))
        if all_X:
            self.X = np.concatenate(all_X, axis=0)
            self.S = np.concatenate(all_static, axis=0)
            self.y = np.concatenate(all_y, axis=0)
        else:
            F_dyn = len(DYNAMIC_FEATURES)
            F_static = len(STATIC_ATTRIBUTES)
            self.X = np.empty((0, sequence_length, F_dyn), dtype=np.float32)
            self.S = np.empty((0, F_static), dtype=np.float32)
            self.y = np.empty((0,), dtype=np.float32)
        self.basin_index = all_basin

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        if torch is None:
            raise RuntimeError("torch is required to iterate the dataset.")
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.S[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )
