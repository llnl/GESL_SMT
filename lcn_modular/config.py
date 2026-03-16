"""Configuration objects for the modular LCN training / evaluation pipeline.

This is intended as a lightweight refactor scaffold around the two scripts:
- smt_lcn.py (RF + ADASYN/SMOTE)
- smt_lcn_with_rebalancing.py (XGBoost + GMM-based rebalance + scale_pos_weight search)

The goal is to make the model choice and imbalance-handling choice configurable
from a single entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

ModelName = Literal["rf", "xgb"]
ResamplerName = Literal["none", "smote", "adasyn", "gmm_rebalance"]
WeightingName = Literal["none", "balanced", "scale_pos_weight"]
SamplerScope = Literal["per_label", "global"]


@dataclass
class LCNConfig:
    # IO
    input_csv: str = "Inputs/pmu_transmission_features.csv"
    sec_to_tert_json: str = "Inputs/sec_to_tert_dict_transmission.json"
    save_dir: str = "Classifiers"
    outputs_dir: str = "Outputs"

    # Data
    group_labels: tuple[str, ...] = ("Events", "State", "Equipment", "Conditions", "Phase")
    # Your original scripts use a 5% test split inside validation() and inside train_for_group()
    test_size: float = 0.05
    random_state: int = 42

    # Model selection
    model: ModelName = "rf"
    n_estimators: int = 1000
    max_depth: int = 8

    # XGBoost tuning (matches smt_lcn_with_rebalancing.py defaults)
    tune_xgb: bool = True
    xgb_n_iter: int = 100
    xgb_cv: int = 5
    xgb_scoring: str = "accuracy"
    xgb_early_stopping_rounds: int = 5
    xgb_val_size: float = 0.15

    # Imbalance handling (can differ per hierarchy level)
    secondary_resampler: ResamplerName = "none"
    tertiary_resampler: ResamplerName = "none"
    # How to apply SMOTE/ADASYN resampling:
    # - global: resample once per level on multiclass labels (like smt_lcn.py)
    # - per_label: resample separately for each binary one-vs-rest label (more standard)
    sampler_scope: SamplerScope = "per_label"

    # Weighting
    # - rf: "balanced" -> RandomForestClassifier(class_weight='balanced')
    # - xgb: "scale_pos_weight" -> set (or search around) scale_pos_weight based on imbalance ratio
    weighting: WeightingName = "none"

    # Resampler parameters (passed through to SMOTE/ADASYN)
    secondary_resampler_params: Dict[str, Any] = field(default_factory=dict)
    tertiary_resampler_params: Dict[str, Any] = field(default_factory=dict)

    # Experiment naming
    save_prefix: str = "RF"  # affects the pickle filenames

    # Control flow
    train_models: bool = True

    def resolved_save_prefix(self) -> str:
        """If user didn't override save_prefix, set a reasonable default."""
        if self.save_prefix and self.save_prefix.strip():
            return self.save_prefix
        return "XGB" if self.model == "xgb" else "RF"
