from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import ModelConfig

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
except Exception:  # pragma: no cover
    RandomForestClassifier = None
    RandomizedSearchCV = None
    train_test_split = None

# XGBoost is optional
try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


def _require(obj, name: str) -> None:
    if obj is None:
        raise ImportError(f"{name} is not available in this environment.")


def make_estimator(
    cfg: ModelConfig,
    *,
    imbalance_ratio: Optional[float] = None,
    random_state: int = 42,
):
    """Create an estimator according to config.

    If cfg.randomized_search is True (supported for XGB), returns a RandomizedSearchCV wrapper.
    imbalance_ratio is typically neg/pos for binary labels, used for weighting.
    """
    if cfg.model_type == "rf":
        _require(RandomForestClassifier, "sklearn RandomForestClassifier")
        params = dict(cfg.model_params)
        if cfg.weighting:
            params.setdefault("class_weight", "balanced")
        params.setdefault("random_state", random_state)
        return RandomForestClassifier(**params)

    if cfg.model_type == "xgb":
        _require(xgb, "xgboost")
        params = dict(cfg.model_params)
        params.setdefault("random_state", random_state)

        # NOTE: weighting for XGB is commonly scale_pos_weight = neg/pos
        if cfg.weighting and imbalance_ratio is not None and np.isfinite(imbalance_ratio):
            params.setdefault("scale_pos_weight", float(imbalance_ratio))

        base = xgb.XGBClassifier(**params)

        if not cfg.randomized_search:
            return base

        # RandomizedSearchCV configuration (similar spirit to smt_lcn_with_rebalancing.py)
        _require(RandomizedSearchCV, "sklearn RandomizedSearchCV")
        _require(stats, "scipy.stats")

        spw_loc = float(imbalance_ratio) if (imbalance_ratio is not None and np.isfinite(imbalance_ratio)) else 1.0

        # Defaults that mirror your original idea: tune subsample/colsample_bytree and scale_pos_weight around ratio
        default_param_dist = {
            "subsample": stats.uniform(loc=0.5, scale=0.2),
            "colsample_bytree": stats.uniform(loc=0.5, scale=0.2),
            "scale_pos_weight": stats.uniform(loc=spw_loc, scale=1.0),
        }
        param_dist = dict(default_param_dist)
        param_dist.update(cfg.search_params.get("param_distributions", {}))

        n_iter = int(cfg.search_params.get("n_iter", 100))
        cv = int(cfg.search_params.get("cv", 5))
        scoring = cfg.search_params.get("scoring", "accuracy")

        return RandomizedSearchCV(
            estimator=base,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
        )

    raise ValueError(f"Unknown model_type: {cfg.model_type}")
