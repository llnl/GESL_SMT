from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from imblearn.over_sampling import SMOTE, ADASYN
except Exception:  # pragma: no cover
    SMOTE = None
    ADASYN = None

try:
    from sklearn.mixture import GaussianMixture
except Exception:  # pragma: no cover
    GaussianMixture = None

from .config import ResamplingConfig


def _require(pkg_obj, name: str) -> None:
    if pkg_obj is None:
        raise ImportError(
            f"{name} is not available. Install the required dependency (e.g., imbalanced-learn or scikit-learn)."
        )


def _safe_neighbor_value(requested: int, min_class_count: int) -> int:
    """Return a safe neighbor value.

    If the requested value is too large for the smallest class, use 1.
    """
    if min_class_count <= 2:
        return 1
    if requested >= min_class_count:
        return 1
    return max(1, requested)


def apply_multiclass_resampling(
    X: np.ndarray, y: np.ndarray, cfg: ResamplingConfig, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE/ADASYN once to a multiclass dataset (scope='dataset').

    GMM is not supported here because it is a binary-only approach in your original code.
    """
    y = np.asarray(y)

    if cfg.type == "none":
        return X, y

    if cfg.type == "gmm":
        # Not meaningful for multiclass; treat as no-op.
        return X, y

    classes, counts = np.unique(y, return_counts=True)

    # Cannot resample if only one class is present
    if len(classes) < 2:
        print("[WARN] Multiclass resampling skipped: only one class present.")
        return X, y

    min_count = int(np.min(counts))

    # If any class has fewer than 2 samples, neighbor-based methods are unsafe
    if min_count < 2:
        print(
            f"[WARN] Multiclass resampling skipped: smallest class has only {min_count} sample(s)."
        )
        return X, y

    if cfg.type == "smote":
        _require(SMOTE, "SMOTE")
        params = dict(cfg.params)
        requested_k = int(params.get("k_neighbors", 5))
        used_k = _safe_neighbor_value(requested_k, min_count)
        params["k_neighbors"] = used_k

        if used_k != requested_k:
            print(
                f"[WARN] SMOTE k_neighbors reduced from {requested_k} to {used_k} "
                f"because min class count is {min_count}."
            )

        try:
            sampler = SMOTE(random_state=random_state, **params)
            Xr, yr = sampler.fit_resample(X, y)
            return np.asarray(Xr), np.asarray(yr)
        except Exception as e:
            print(f"[WARN] SMOTE failed in multiclass resampling: {e}")
            print("[WARN] Returning original data without multiclass resampling.")
            return X, y

    if cfg.type == "adasyn":
        _require(ADASYN, "ADASYN")
        params = dict(cfg.params)
        requested_n = int(params.get("n_neighbors", 5))
        used_n = _safe_neighbor_value(requested_n, min_count)
        params["n_neighbors"] = used_n

        if used_n != requested_n:
            print(
                f"[WARN] ADASYN n_neighbors reduced from {requested_n} to {used_n} "
                f"because min class count is {min_count}."
            )

        try:
            sampler = ADASYN(random_state=random_state, **params)
            Xr, yr = sampler.fit_resample(X, y)
            return np.asarray(Xr), np.asarray(yr)
        except Exception as e:
            print(f"[WARN] ADASYN failed in multiclass resampling: {e}")
            print("[WARN] Returning original data without multiclass resampling.")
            return X, y

    raise ValueError(f"Unknown resampler type: {cfg.type}")


def apply_binary_resampling(
    X: np.ndarray, y_bin: np.ndarray, cfg: ResamplingConfig, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply augmentation to a binary task.

    Returns:
      X_aug, y_aug, imbalance_ratio (neg/pos after augmentation)
    """
    y_bin = np.asarray(y_bin).astype(int)

    if cfg.type == "none":
        return X, y_bin, _imbalance_ratio(y_bin)

    classes, counts = np.unique(y_bin, return_counts=True)
    if len(classes) < 2:
        print("[WARN] Binary resampling skipped: only one class present.")
        return X, y_bin, _imbalance_ratio(y_bin)

    min_count = int(np.min(counts))

    if cfg.type == "smote":
        _require(SMOTE, "SMOTE")

        if min_count < 2:
            print(
                f"[WARN] Binary SMOTE skipped: smallest class has only {min_count} sample(s)."
            )
            return X, y_bin, _imbalance_ratio(y_bin)

        params = dict(cfg.params)
        requested_k = int(params.get("k_neighbors", 5))
        used_k = _safe_neighbor_value(requested_k, min_count)
        params["k_neighbors"] = used_k

        if used_k != requested_k:
            print(
                f"[WARN] Binary SMOTE k_neighbors reduced from {requested_k} to {used_k} "
                f"because min class count is {min_count}."
            )

        try:
            sampler = SMOTE(random_state=random_state, **params)
            Xr, yr = sampler.fit_resample(X, y_bin)
            yr = np.asarray(yr).astype(int)
            return np.asarray(Xr), yr, _imbalance_ratio(yr)
        except Exception as e:
            print(f"[WARN] Binary SMOTE failed: {e}")
            print("[WARN] Returning original binary data without resampling.")
            return X, y_bin, _imbalance_ratio(y_bin)

    if cfg.type == "adasyn":
        _require(ADASYN, "ADASYN")

        if min_count < 2:
            print(
                f"[WARN] Binary ADASYN skipped: smallest class has only {min_count} sample(s)."
            )
            return X, y_bin, _imbalance_ratio(y_bin)

        params = dict(cfg.params)
        requested_n = int(params.get("n_neighbors", 5))
        used_n = _safe_neighbor_value(requested_n, min_count)
        params["n_neighbors"] = used_n

        if used_n != requested_n:
            print(
                f"[WARN] Binary ADASYN n_neighbors reduced from {requested_n} to {used_n} "
                f"because min class count is {min_count}."
            )

        try:
            sampler = ADASYN(random_state=random_state, **params)
            Xr, yr = sampler.fit_resample(X, y_bin)
            yr = np.asarray(yr).astype(int)
            return np.asarray(Xr), yr, _imbalance_ratio(yr)
        except Exception as e:
            print(f"[WARN] Binary ADASYN failed: {e}")
            print("[WARN] Returning original binary data without resampling.")
            return X, y_bin, _imbalance_ratio(y_bin)

    if cfg.type == "gmm":
        _require(GaussianMixture, "GaussianMixture")
        return gmm_rebalance(
            X,
            y_bin,
            target_minority_fraction=cfg.target_minority_fraction,
            random_state=random_state,
        )

    raise ValueError(f"Unknown resampler type: {cfg.type}")


def gmm_rebalance(
    X: np.ndarray, y_bin: np.ndarray, target_minority_fraction: float = 0.33, random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Reproduce the core idea of smt_lcn_with_rebalancing.py's `rebalance()`.

    If minority class fraction < target_minority_fraction, fit 1-component GMM to minority samples and
    sample synthetic points to raise minority fraction to the target.
    """
    y_bin = np.asarray(y_bin).astype(int)
    n = len(y_bin)
    if n == 0:
        return X, y_bin, 1.0

    pos_frac = float(np.mean(y_bin))
    neg_frac = 1.0 - pos_frac

    if pos_frac <= 0:
        return X, y_bin, float("inf")

    if pos_frac >= target_minority_fraction:
        return X, y_bin, neg_frac / pos_frac

    pos_idx = np.where(y_bin == 1)[0]
    X_pos = X[pos_idx, :]

    # Need at least 2 minority samples for GMM to be meaningful/safe
    if X_pos.shape[0] < 2:
        print(
            f"[WARN] GMM rebalance skipped: minority class has only {X_pos.shape[0]} sample(s)."
        )
        return X, y_bin, neg_frac / pos_frac

    # Fit a 1-component Gaussian Mixture Model to minority class
    gmm = GaussianMixture(n_components=1, random_state=random_state).fit(X_pos)

    # Solve for how many synthetic positives we need so that:
    # (pos + k) / (n + k) >= target
    # => k >= (target*n - pos) / (1 - target)
    pos = int(pos_frac * n)
    target = target_minority_fraction
    k = int(np.ceil((target * n - pos) / (1.0 - target)))
    if k <= 0:
        return X, y_bin, neg_frac / pos_frac

    X_syn = gmm.sample(n_samples=k)[0]
    X_aug = np.vstack([X, X_syn])
    y_aug = np.hstack([y_bin, np.ones(k, dtype=int)])

    pos_frac2 = float(np.mean(y_aug))
    neg_frac2 = 1.0 - pos_frac2
    ratio = neg_frac2 / pos_frac2 if pos_frac2 > 0 else float("inf")
    return X_aug, y_aug, ratio


def _imbalance_ratio(y_bin: np.ndarray) -> float:
    y_bin = np.asarray(y_bin).astype(int)
    pos = float(np.sum(y_bin))
    neg = float(len(y_bin) - pos)
    if pos <= 0:
        return float("inf")
    return neg / pos