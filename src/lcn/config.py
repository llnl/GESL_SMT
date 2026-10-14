from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal


ModelType = Literal["rf", "xgb"]
ResamplerType = Literal["none", "smote", "adasyn", "gmm"]
ResampleScope = Literal["dataset", "binary"]
ValidationMode = Literal["holdout", "kfold"]
CVStratify = Literal["secondary", "hierarchy", "none"]


@dataclass(frozen=True)
class ResamplingConfig:
    """Configuration for data augmentation / resampling.

    - type: none/smote/adasyn/gmm
    - scope:
        - dataset: apply SMOTE/ADASYN once on the multiclass training set (like smt_lcn.py)
        - binary: apply per-binary task (needed for GMM and often preferable for strict binary control)
      Note: GMM is always treated as binary (scope is effectively ignored).
    """
    type: ResamplerType = "none"
    scope: ResampleScope = "dataset"
    params: Dict[str, Any] = field(default_factory=dict)
    # Only used for GMM rebalance: try to raise minority fraction to at least this value.
    target_minority_fraction: float = 0.33


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration.

    For XGB:
      - weighting: when True, use scale_pos_weight based on class imbalance
      - randomized_search: when True, wrap with RandomizedSearchCV similar to smt_lcn_with_rebalancing.py

    For RF:
      - weighting: when True, use class_weight='balanced'
    """
    model_type: ModelType = "rf"
    model_params: Dict[str, Any] = field(default_factory=dict)
    weighting: bool = False
    randomized_search: bool = False
    search_params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    primary_group: str
    model: ModelConfig = field(default_factory=ModelConfig)
    secondary_resampling: ResamplingConfig = field(default_factory=ResamplingConfig)
    tertiary_resampling: ResamplingConfig = field(default_factory=ResamplingConfig)

    # Paths
    inputs_dir: str = "Inputs"
    outputs_dir: str = "Outputs"
    classifiers_dir: str = "Classifiers"

    # Save prefix: if None, an informative prefix will be auto-generated.
    save_prefix: Optional[str] = None

    # Training toggles
    train: bool = True

    # Validation toggles
    validation_mode: ValidationMode = "holdout"
    n_splits: int = 5
    cv_stratify: CVStratify = "secondary"
    cv_shuffle: bool = True
    random_state: int = 42

    # Feature columns: your data assumes first 5 columns are IDs/labels; remaining are features.
    feature_start_col: int = 5
