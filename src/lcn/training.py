from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .augmentation import apply_multiclass_resampling, apply_binary_resampling
from .config import PipelineConfig
from .models import make_estimator
from .store import ClassifierStore


def train_primary(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    store: ClassifierStore,
    random_state: int | None = None,
) -> None:
    """Train binary classifiers for all primary labels.

    Trains one binary classifier per primary label across the entire dataset.
    """
    rs = cfg.random_state if random_state is None else random_state
    df1 = df.copy().dropna().reset_index(drop=True)

    if df1.shape[0] == 0:
        return

    # Features assumed to start at column index cfg.feature_start_col
    X = df1.iloc[:, cfg.feature_start_col :].to_numpy()

    # ---- Primary level ----
    y_prim = df1["primary"].to_numpy()
    X_train, y_train = _safe_train_slice(X, y_prim, random_state=rs)

    # Note: Using secondary_resampling config for primary level
    # You can add a separate primary_resampling config if needed
    if cfg.secondary_resampling.scope == "dataset":
        X_train, y_train = apply_multiclass_resampling(
            X_train, y_train, cfg.secondary_resampling, random_state=rs
        )

    prim_labels = (
        df1.groupby(["primary"]).size().sort_values(ascending=False).index.to_list()
    )

    for label in prim_labels:
        y_bin = (y_train == label).astype(int)
        if np.unique(y_bin).size < 2:
            continue

        Xb, yb, ratio = (X_train, y_bin, _imbalance_ratio(y_bin))
        if cfg.secondary_resampling.scope == "binary" or cfg.secondary_resampling.type == "gmm":
            Xb, yb, ratio = apply_binary_resampling(
                X_train, y_bin, cfg.secondary_resampling, random_state=rs
            )

        if np.unique(yb).size < 2:
            continue

        model = make_estimator(cfg.model, imbalance_ratio=ratio, random_state=rs)
        _fit_model(model, Xb, yb, random_state=rs)
        store.save("primary", label, model)


def train_for_group(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    store: ClassifierStore,
    random_state: int | None = None,
) -> None:
    """Train binary classifiers for all secondary and tertiary labels under cfg.primary_group.

    Supports interchangeable model choice (RF / XGB) and augmentation choice
    (none / SMOTE / ADASYN / GMM rebalance) with per-level configuration.
    """
    rs = cfg.random_state if random_state is None else random_state
    df1 = (
        df.loc[df["primary"] == cfg.primary_group]
        .copy()
        .dropna()
        .reset_index(drop=True)
    )

    if df1.shape[0] == 0:
        return

    _train_on_filtered_df(df1, cfg, store=store, random_state=rs)


def train_on_dataframe(
    df_subset: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    store: ClassifierStore,
    random_state: int | None = None,
) -> None:
    """Train on a caller-provided dataframe subset.

    Used by outer K-fold validation so each fold only sees its training partition.
    """
    rs = cfg.random_state if random_state is None else random_state
    df1 = df_subset.copy().dropna().reset_index(drop=True)
    if df1.shape[0] == 0:
        return
    _train_on_filtered_df(df1, cfg, store=store, random_state=rs)


def train_secondary_all(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    store: ClassifierStore,
    random_state: int | None = None,
) -> None:
    """Train binary classifiers for all secondary labels, segregated by primary label.

    For each primary label, train secondary classifiers using only data from that primary group.
    """
    rs = cfg.random_state if random_state is None else random_state
    df_clean = df.copy().dropna().reset_index(drop=True)

    if df_clean.shape[0] == 0:
        return

    # Get all unique primary labels
    primary_labels = df_clean["primary"].unique()

    for primary_label in primary_labels:
        print(f"  Training secondary classifiers for primary='{primary_label}'")

        # Filter by this primary label
        df_primary = df_clean[df_clean["primary"] == primary_label].copy().reset_index(drop=True)

        if df_primary.shape[0] == 0:
            continue

        # Train secondary level for this primary group
        _train_secondary_level(df_primary, cfg, store=store, random_state=rs, primary_label=primary_label)


def train_tertiary_all(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    store: ClassifierStore,
    random_state: int | None = None,
) -> None:
    """Train binary classifiers for all tertiary labels, segregated by secondary label.

    For each secondary label, train tertiary classifiers using only data from that secondary group.
    """
    rs = cfg.random_state if random_state is None else random_state
    df_clean = df.copy().dropna().reset_index(drop=True)

    if df_clean.shape[0] == 0:
        return

    # Get all unique secondary labels
    secondary_labels = df_clean["secondary"].unique()

    for secondary_label in secondary_labels:
        print(f"  Training tertiary classifiers for secondary='{secondary_label}'")

        # Filter by this secondary label
        df_secondary = df_clean[df_clean["secondary"] == secondary_label].copy().reset_index(drop=True)

        if df_secondary.shape[0] == 0:
            continue

        # Train tertiary level for this secondary group
        _train_tertiary_level(df_secondary, cfg, store=store, random_state=rs, secondary_label=secondary_label)


def _train_secondary_level(
    df1: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    store: ClassifierStore,
    random_state: int = 42,
    primary_label: str = None,
) -> None:
    """Train secondary classifiers on a dataframe (already filtered by primary if needed)."""
    X = df1.iloc[:, cfg.feature_start_col :].to_numpy()

    y_sec = df1["secondary"].to_numpy()
    X_train, y_train = _safe_train_slice(X, y_sec, random_state=random_state)

    if cfg.secondary_resampling.scope == "dataset":
        X_train, y_train = apply_multiclass_resampling(
            X_train, y_train, cfg.secondary_resampling, random_state=random_state
        )

    sec_labels = (
        df1.groupby(["secondary"]).size().sort_values(ascending=False).index.to_list()
    )

    for label in sec_labels:
        y_bin = (y_train == label).astype(int)
        if np.unique(y_bin).size < 2:
            continue

        Xb, yb, ratio = (X_train, y_bin, _imbalance_ratio(y_bin))
        if cfg.secondary_resampling.scope == "binary" or cfg.secondary_resampling.type == "gmm":
            Xb, yb, ratio = apply_binary_resampling(
                X_train, y_bin, cfg.secondary_resampling, random_state=random_state
            )

        if np.unique(yb).size < 2:
            continue

        model = make_estimator(cfg.model, imbalance_ratio=ratio, random_state=random_state)
        _fit_model(model, Xb, yb, random_state=random_state)

        # Save with primary label context if provided
        save_key = f"{primary_label}_{label}" if primary_label else label
        store.save("secondary", save_key, model)


def _train_tertiary_level(
    df1: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    store: ClassifierStore,
    random_state: int = 42,
    secondary_label: str = None,
) -> None:
    """Train tertiary classifiers on a dataframe (already filtered by secondary if needed)."""
    X = df1.iloc[:, cfg.feature_start_col :].to_numpy()

    y_ter = df1["tertiary"].to_numpy()
    X_train, y_train = _safe_train_slice(X, y_ter, random_state=random_state)

    if cfg.tertiary_resampling.scope == "dataset":
        X_train, y_train = apply_multiclass_resampling(
            X_train, y_train, cfg.tertiary_resampling, random_state=random_state
        )

    ter_labels = (
        df1.groupby(["tertiary"]).size().sort_values(ascending=False).index.to_list()
    )

    for label in ter_labels:
        y_bin = (y_train == label).astype(int)
        if np.unique(y_bin).size < 2:
            continue

        Xb, yb, ratio = (X_train, y_bin, _imbalance_ratio(y_bin))
        if cfg.tertiary_resampling.scope == "binary" or cfg.tertiary_resampling.type == "gmm":
            Xb, yb, ratio = apply_binary_resampling(
                X_train, y_bin, cfg.tertiary_resampling, random_state=random_state
            )

        if np.unique(yb).size < 2:
            continue

        model = make_estimator(cfg.model, imbalance_ratio=ratio, random_state=random_state)
        _fit_model(model, Xb, yb, random_state=random_state)

        # Save with secondary label context if provided
        save_key = f"{secondary_label}_{label}" if secondary_label else label
        store.save("tertiary", save_key, model)


def _train_on_filtered_df(
    df1: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    store: ClassifierStore,
    random_state: int = 42,
) -> None:
    # Features assumed to start at column index cfg.feature_start_col
    X = df1.iloc[:, cfg.feature_start_col :].to_numpy()

    # ---- Secondary level ----
    y_sec = df1["secondary"].to_numpy()
    X_train, y_train = _safe_train_slice(X, y_sec, random_state=random_state)

    if cfg.secondary_resampling.scope == "dataset":
        X_train, y_train = apply_multiclass_resampling(
            X_train, y_train, cfg.secondary_resampling, random_state=random_state
        )

    sec_labels = (
        df1.groupby(["secondary"]).size().sort_values(ascending=False).index.to_list()
    )

    for label in sec_labels:
        y_bin = (y_train == label).astype(int)
        if np.unique(y_bin).size < 2:
            continue

        Xb, yb, ratio = (X_train, y_bin, _imbalance_ratio(y_bin))
        if cfg.secondary_resampling.scope == "binary" or cfg.secondary_resampling.type == "gmm":
            Xb, yb, ratio = apply_binary_resampling(
                X_train, y_bin, cfg.secondary_resampling, random_state=random_state
            )

        if np.unique(yb).size < 2:
            continue

        model = make_estimator(cfg.model, imbalance_ratio=ratio, random_state=random_state)
        _fit_model(model, Xb, yb, random_state=random_state)
        store.save("secondary", label, model)

    # ---- Tertiary level ----
    y_ter = df1["tertiary"].to_numpy()
    X_train, y_train = _safe_train_slice(X, y_ter, random_state=random_state)

    if cfg.tertiary_resampling.scope == "dataset":
        X_train, y_train = apply_multiclass_resampling(
            X_train, y_train, cfg.tertiary_resampling, random_state=random_state
        )

    ter_labels = (
        df1.groupby(["tertiary"]).size().sort_values(ascending=False).index.to_list()
    )

    for label in ter_labels:
        y_bin = (y_train == label).astype(int)
        if np.unique(y_bin).size < 2:
            continue

        Xb, yb, ratio = (X_train, y_bin, _imbalance_ratio(y_bin))
        if cfg.tertiary_resampling.scope == "binary" or cfg.tertiary_resampling.type == "gmm":
            Xb, yb, ratio = apply_binary_resampling(
                X_train, y_bin, cfg.tertiary_resampling, random_state=random_state
            )

        if np.unique(yb).size < 2:
            continue

        model = make_estimator(cfg.model, imbalance_ratio=ratio, random_state=random_state)
        _fit_model(model, Xb, yb, random_state=random_state)
        store.save("tertiary", label, model)


def _safe_train_slice(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    if len(y) < 2:
        return X, y
    try:
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.05, random_state=random_state, stratify=y if np.unique(y).size > 1 else None
        )
        return X_train, y_train
    except Exception:
        return X, y


def _fit_model(model, X: np.ndarray, y: np.ndarray, random_state: int = 42) -> None:
    """Fit model; if it supports early stopping (XGB) and uses randomized search,
    we mimic original validation split usage when possible."""
    name = model.__class__.__name__
    if name == "RandomizedSearchCV":
        try:
            X_sub, X_val, y_sub, y_val = train_test_split(X, y, test_size=0.15, random_state=random_state, stratify=y if np.unique(y).size > 1 else None)
            model.fit(X_sub, y_sub, eval_set=[(X_val, y_val)], verbose=False)
            return
        except TypeError:
            pass
        except Exception:
            pass

    model.fit(X, y)


def _imbalance_ratio(y_bin: np.ndarray) -> float:
    y_bin = np.asarray(y_bin).astype(int)
    pos = float(np.sum(y_bin))
    neg = float(len(y_bin) - pos)
    if pos <= 0:
        return float("inf")
    return neg / pos
