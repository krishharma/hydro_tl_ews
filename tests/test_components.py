"""Targeted unit tests for the math-y bits."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hydro_tl_ews.evaluation.metrics import (
    auc_roc,
    brier_score,
    f1_at_threshold,
    kge,
    nse,
    pbias,
    reliability_curve,
)
from hydro_tl_ews.evaluation.extreme_thresholds import (
    regional_thresholds,
    warning_labels,
    predicted_warning_probabilities,
)


def test_nse_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert nse(y, y) == 1.0


def test_kge_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(kge(y, y) - 1.0) < 1e-9


def test_pbias_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert pbias(y, y) == 0.0


def test_auc_perfect_separation():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc_roc(labels, scores) == 1.0


def test_brier_zero():
    labels = np.array([1, 0, 1, 0])
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(labels, probs) == 0.0


def test_reliability_shape():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200)
    y = (rng.uniform(size=200) < p).astype(int)
    centers, freq, counts = reliability_curve(y, p, n_bins=5)
    assert len(centers) == 5
    assert counts.sum() == 200


def test_regional_thresholds():
    s = pd.Series(np.random.default_rng(0).gamma(2, 1, 365 * 6),
                  index=pd.date_range("2000-01-01", periods=365 * 6))
    rfa = regional_thresholds(s, years_required=5)
    assert rfa.q5 < rfa.q95 < rfa.q99


def test_warning_labels_and_probs():
    idx = pd.date_range("2000-01-01", periods=300)
    flow = pd.Series(np.linspace(0, 5, 300), index=idx)
    rfa = type("RFA", (), {"q5": 0.5, "q95": 4.0, "q99": 4.8})
    labels = warning_labels(flow, rfa, kind="flood", percentile="q95",
                            lead_times=(1, 3))
    assert labels.shape[1] == 2
    probs = predicted_warning_probabilities(flow, rfa, kind="flood",
                                            percentile="q95",
                                            sigma=0.5, lead_times=(1, 3))
    assert (probs.values >= 0).all() and (probs.values <= 1).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: OK")
