"""Model factory functions (RF and XGBoost)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


def build_rf(
    *,
    n_estimators: int = 1000,
    max_depth: int = 8,
    random_state: int = 42,
    class_weight: Optional[str] = None,
):
    from sklearn.ensemble import RandomForestClassifier

    kwargs: Dict[str, Any] = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    if class_weight is not None:
        kwargs["class_weight"] = class_weight

    return RandomForestClassifier(**kwargs)


def build_xgb_base(
    *,
    n_estimators: int = 1000,
    max_depth: int = 8,
    random_state: int = 42,
    early_stopping_rounds: int = 5,
    scale_pos_weight: Optional[float] = None,
):
    import xgboost as xgb

    kwargs: Dict[str, Any] = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
        # The sklearn wrapper defaults are okay; you can add learning_rate, subsample, etc.
    )
    if scale_pos_weight is not None:
        kwargs["scale_pos_weight"] = float(scale_pos_weight)

    return xgb.XGBClassifier(**kwargs)


def maybe_wrap_xgb_random_search(
    xgb_model,
    *,
    tune: bool,
    imbalance_ratio: Optional[float],
    n_iter: int = 100,
    cv: int = 5,
    scoring: str = "accuracy",
    random_state: int = 42,
):
    """Optionally wrap an XGBClassifier in RandomizedSearchCV.

    Mirrors smt_lcn_with_rebalancing.py: searches subsample/colsample_bytree and
    scale_pos_weight around the computed imbalance ratio.
    """
    if not tune:
        return xgb_model, None

    from sklearn.model_selection import RandomizedSearchCV
    import scipy.stats as stats

    # Guard against missing ratio
    loc = float(imbalance_ratio) if imbalance_ratio is not None else 1.0

    param_dist = {
        "subsample": stats.uniform(loc=0.5, scale=0.2),
        "colsample_bytree": stats.uniform(loc=0.5, scale=0.2),
        "scale_pos_weight": stats.uniform(loc=loc, scale=1),
    }

    search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
    )
    return search, param_dist
