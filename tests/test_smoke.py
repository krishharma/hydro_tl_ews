"""Smoke test — exercises the whole pipeline end-to-end on synthetic data."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hydro_tl_ews.utils.config import ExperimentConfig
from scripts.smoke_pipeline import run_smoke


def test_smoke_pipeline_runs():
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "smoke_test.yaml")
    summary = run_smoke(cfg)
    assert "metrics" in summary
    assert summary["n_predictions"] > 0


if __name__ == "__main__":
    test_smoke_pipeline_runs()
    print("Smoke test PASSED")
