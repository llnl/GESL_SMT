# Modular LCN pipeline scaffold

This folder provides a refactor scaffold that lets you:

- keep shared LCN logic in reusable modules
- select **model** (`rf` vs `xgb`) at runtime
- select **imbalance handling** per hierarchy level (`secondary` and `tertiary`)
  - `none`, `smote`, `adasyn`, `gmm_rebalance`
- optionally enable **weighting**
  - RF: `balanced`
  - XGB: `scale_pos_weight`

## Layout

- `config.py` – experiment configuration
- `augment.py` – SMOTE/ADASYN and GMM rebalance
- `models.py` – RF / XGB factories + optional RandomizedSearchCV wrapper
- `train.py` – trains one-vs-rest binaries and saves pickles
- `inference.py` – loads pickles and runs hierarchical LCN validation
- `metrics.py` – hierarchical metrics + per-secondary breakdown
- `pipeline.py` – orchestrates training + validation + CSV outputs

## Entry point

Use `run_lcn.py` at the repository root.

Examples are at the top of `run_lcn.py`.
