# Hydrological Transfer Learning Early Warning System

**Reproducible code accompanying** *"Leveraging Transfer Learning and Walk-Forward
Validation for Probabilistic Streamflow Early Warning in Data-Scarce Basins:
An Entity-Aware LSTM Framework with Explainable AI."*

This repository implements the full Phase 1 (regional pre-training) +
Phase 2 (conservative / progressive fine-tuning) + Phase 3 (rolling-origin
walk-forward evaluation) pipeline described in the paper, using the open-source
[CAMELS-US](https://ral.ucar.edu/solutions/products/camels) benchmark and a
custom Entity-Aware LSTM (Kratzert et al., 2019).

## Quick start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate hydro_tl_ews
pip install -e .

# 2. Run the synthetic smoke test (no CAMELS download required, ~3 min on CPU)
python -m pytest tests/ -v
python scripts/run_experiment.py --config configs/smoke_test.yaml --smoke

# 3. Real CAMELS pipeline (requires GPU and ~2.8 GB CAMELS download)
#    Place CAMELS at data/CAMELS_US/ then:
python scripts/run_experiment.py --config configs/pretrain.yaml
python scripts/run_experiment.py --config configs/finetune_conservative.yaml
python scripts/run_experiment.py --config configs/finetune_progressive.yaml
python scripts/run_experiment.py --config configs/local_baseline.yaml
python scripts/run_experiment.py --config configs/walk_forward.yaml
```

## Repository layout

```
hydro_tl_ews/
├── configs/                      # YAML configurations for each pipeline stage
├── src/hydro_tl_ews/
│   ├── data/                     # CAMELS / NWIS / synthetic loaders
│   ├── models/                   # EA-LSTM cell + NSE loss
│   ├── training/                 # Trainer, transfer recipes, walk-forward
│   ├── evaluation/               # Metrics + Regional Frequency Analysis
│   ├── xai/                      # SHAP wrappers
│   └── utils/                    # Seeding, config loader, logging
├── scripts/
│   ├── run_experiment.py         # CLI entry point
│   ├── smoke_pipeline.py         # End-to-end synthetic demo
│   └── stages/                   # One module per pipeline stage
├── tests/                        # pytest unit + smoke tests
├── results/                      # Auto-populated outputs (gitignored)
└── docs/                         # Paper and methodology notes
```

## Key components

| Module                                       | What it does                                                            |
| -------------------------------------------- | ----------------------------------------------------------------------- |
| `models/ealstm.py`                           | Entity-Aware LSTM cell with static input gate, freeze/unfreeze helpers  |
| `models/losses.py`                           | Differentiable per-basin-normalized NSE loss                            |
| `training/trainer.py`                        | Generic trainer + early stopping + checkpointing                        |
| `training/transfer.py`                       | Conservative (Approach A) and progressive (Approach B) fine-tuners      |
| `training/walk_forward.py`                   | Rolling-origin backtester with online bias correction                   |
| `evaluation/extreme_thresholds.py`           | Regional Frequency Analysis: Q5 / Q95 / Q99, multi-lead-time labels     |
| `evaluation/metrics.py`                      | NSE, KGE, PBIAS, AUC, F1, Brier, reliability curve                      |
| `xai/shap_analysis.py`                       | DeepExplainer / GradientExplainer wrappers, global importance summaries |
| `data/clustering.py`                         | k-means donor-basin selection (Ougahi & Rowan 2026)                     |

## Configuration

Every stage is driven by a YAML file in `configs/`.  Key knobs:

* `model.hidden_size`, `model.dropout`, `model.initial_forget_bias`
* `data.target_basin` — CAMELS gauge id (e.g. `11264500` Tuolumne)
* `data.warmup_period` — the simulated 2-year data-scarce window
* `walk_forward.refit_every_days` — how often to do a full fine-tune during
  operational simulation
* `walk_forward.fine_tune` — recipe used at every refit

## Data

* **CAMELS-US** — <https://ral.ucar.edu/solutions/products/camels>
  (DOIs 10.5065/D6G73C3Q and 10.5065/D6MW2F4D)
* **USGS NWIS** — accessible via `dataretrieval` Python package
* **Daymet** — <https://daymet.ornl.gov>

For users without GPU access, `data/synthetic_camels.py` provides a
physically motivated 12-basin synthetic generator that exercises every
component of the pipeline (used by the smoke test).

## Reproducibility

* Global seed is fixed in `utils/seed.py` (Python, NumPy, PyTorch, cuDNN).
* All normalization statistics are computed on training data only — no
  look-ahead bias.
* Walk-forward evaluation uses strict rolling-origin splits.
* Extreme event thresholds (Q95 / Q99 / Q5) are derived from the *full*
  30-year CAMELS record via Regional Frequency Analysis, never from the
  short warmup window.

## Citation

If you use this code, please cite the accompanying paper and the underlying
CAMELS dataset (Newman et al. 2015; Addor et al. 2017) and the EA-LSTM
formulation (Kratzert et al. 2019).

## License

MIT — see `LICENSE`.
