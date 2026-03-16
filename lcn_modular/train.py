"""Training code for the modular LCN pipeline."""

from __future__ import annotations

import os
import pickle
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .augment import compute_imbalance_ratio, resample
from .config import LCNConfig
from .models import build_rf, build_xgb_base, maybe_wrap_xgb_random_search


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def train_for_group(df: pd.DataFrame, group_label: str, cfg: LCNConfig) -> None:
    """Train all binary classifiers under a primary group label."""

    _ensure_dir(cfg.save_dir)

    df1 = df.loc[df["primary"] == group_label].copy().dropna().reset_index()

    if df1.shape[0] == 0:
        print(f"[train] Group '{group_label}': no rows after filtering; skipping.")
        return

    arr_class = np.asanyarray(df1.groupby(["secondary"]).nunique().sort_values(by=["index"], ascending=False).index)
    arr_subclass = np.asanyarray(df1.groupby(["tertiary"]).nunique().sort_values(by=["index"], ascending=False).index)

    features = df1.iloc[:, -(df1.shape[1] - 5):]

    # Train SECONDARY (class) binary classifiers
    _train_level(
        features=features,
        labels=df1["secondary"],
        label_values=arr_class,
        level_name="secondary",
        cfg=cfg,
    )

    # Train TERTIARY (subclass) binary classifiers
    _train_level(
        features=features,
        labels=df1["tertiary"],
        label_values=arr_subclass,
        level_name="tertiary",
        cfg=cfg,
    )


def _train_level(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    label_values: Sequence[str],
    level_name: str,
    cfg: LCNConfig,
) -> None:
    """Train a batch of one-vs-rest classifiers for a hierarchy level."""

    if level_name not in {"secondary", "tertiary"}:
        raise ValueError("level_name must be 'secondary' or 'tertiary'")

    resampler = cfg.secondary_resampler if level_name == "secondary" else cfg.tertiary_resampler
    resampler_params = cfg.secondary_resampler_params if level_name == "secondary" else cfg.tertiary_resampler_params

    # Optional global (multiclass) resampling once per level, like smt_lcn.py
    X_global = features
    y_global = labels

    if cfg.sampler_scope == "global" and resampler in {"smote", "adasyn"}:
        X_global, y_global, _ = resample(
            X_global,
            y_global,
            method=resampler,
            random_state=cfg.random_state,
            **resampler_params,
        )

    if cfg.sampler_scope == "global" and resampler == "gmm_rebalance":
        raise ValueError("gmm_rebalance is binary-only; use sampler_scope='per_label' for this method")

    for evttag in label_values:
        print(f"[train] {level_name}: training binary classifier for label='{evttag}'")

        if cfg.sampler_scope == "global" and resampler in {"smote", "adasyn"}:
            # Binary label derived after multiclass resampling
            y_bin = (pd.Series(y_global) == evttag).astype(int)
            X_used = X_global
            # No additional resampling
            X_reb, y_reb = X_used, y_bin
            imbalance_ratio = compute_imbalance_ratio(y_reb)
        else:
            # Per-label binary label
            y_bin = (labels == evttag).astype(int)

            # Match your xgb script: resample (if any) on the full feature set, then take a train split.
            X_reb, y_reb, info = resample(
                features,
                y_bin,
                method=resampler,
                random_state=cfg.random_state,
                **resampler_params,
            )

            # If the resampler computed an imbalance ratio (gmm_rebalance), use it; otherwise compute.
            imbalance_ratio = float(info.get("imbalance_ratio")) if info.get("imbalance_ratio") is not None else compute_imbalance_ratio(y_reb)

        # Small held-out chunk discarded (mirrors train_test_split(..., test_size=0.05) in both scripts)
        X_train, _, y_train, _ = train_test_split(
            X_reb,
            y_reb,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
        )

        model = _fit_single_model(
            X_train=X_train,
            y_train=y_train,
            imbalance_ratio=imbalance_ratio,
            cfg=cfg,
        )

        fname = os.path.join(cfg.save_dir, f"{cfg.resolved_save_prefix()}_{evttag}.pickle")
        with open(fname, "wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _fit_single_model(*, X_train, y_train, imbalance_ratio: float, cfg: LCNConfig):
    """Fit either RF or XGB (optionally tuned)."""

    if cfg.model == "rf":
        class_weight = "balanced" if cfg.weighting == "balanced" else None
        rf = build_rf(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_state,
            class_weight=class_weight,
        )
        rf.fit(X_train, np.asarray(y_train).astype(int))
        return rf

    if cfg.model == "xgb":
        # For XGB, you can either set scale_pos_weight directly or (as in your script)
        # tune it via RandomizedSearchCV.
        spw = float(imbalance_ratio) if cfg.weighting == "scale_pos_weight" and not cfg.tune_xgb else None

        base = build_xgb_base(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_state,
            early_stopping_rounds=cfg.xgb_early_stopping_rounds,
            scale_pos_weight=spw,
        )

        model, _ = maybe_wrap_xgb_random_search(
            base,
            tune=cfg.tune_xgb,
            imbalance_ratio=float(imbalance_ratio),
            n_iter=cfg.xgb_n_iter,
            cv=cfg.xgb_cv,
            scoring=cfg.xgb_scoring,
            random_state=cfg.random_state,
        )

        # Early stopping needs an eval_set. We'll create a validation split.
        X_sub, X_val, y_sub, y_val = train_test_split(
            X_train,
            y_train,
            test_size=cfg.xgb_val_size,
            random_state=cfg.random_state,
        )

        model.fit(X_sub, y_sub, eval_set=[(X_val, y_val)], verbose=False)
        return model

    raise ValueError(f"Unknown model type: {cfg.model}")
