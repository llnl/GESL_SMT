"""Imbalance handling: oversampling / synthetic data / weighting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd


def _to_numpy(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    return X.values if isinstance(X, pd.DataFrame) else X


def compute_imbalance_ratio(y_binary: Union[np.ndarray, pd.Series, list]) -> float:
    """Return n_negative / n_positive for binary labels.

    If positives are zero, returns a large number (1e9) to avoid division by zero.
    """
    y = np.asarray(y_binary).astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    return float(n_neg) / float(n_pos if n_pos > 0 else 1e-9)


def rebalance_gmm(
    X: Union[pd.DataFrame, np.ndarray],
    y_binary: Union[pd.Series, np.ndarray, list],
    *,
    min_pos_fraction: float = 0.33,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """GMM-based minority class synthetic sampling.

    This matches the logic from smt_lcn_with_rebalancing.py: if the positive
    class fraction is below ~0.33, fit a 1-component GaussianMixture on the
    minority features and sample until positives are at least min_pos_fraction
    of the majority count (approx.).

    Returns (X_reb, y_reb, imbalance_ratio).
    """
    from sklearn.mixture import GaussianMixture

    Xn = _to_numpy(X)
    y = np.asarray(y_binary).astype(int)
    num_obs = len(y)

    class0_weight = 1.0 - float(y.mean())
    class1_weight = float(y.mean())
    imbalance_ratio = class0_weight / (class1_weight if class1_weight > 0 else 1e-9)

    if class1_weight < min_pos_fraction and class1_weight > 0:
        minority_idx = np.where(y == 1)[0]
        minority_X = Xn[minority_idx, :]

        gmm = GaussianMixture(n_components=1, random_state=random_state).fit(minority_X)

        # Sample until positives reach roughly min_pos_fraction of majority
        num_needed = int((min_pos_fraction * (num_obs * class0_weight)) - (num_obs * class1_weight))
        if num_needed > 0:
            syn_samples = gmm.sample(n_samples=num_needed)[0]
            Xn = np.vstack([Xn, syn_samples])
            y = np.hstack([y, np.ones(num_needed, dtype=int)])

            class0_weight = 1.0 - float(y.mean())
            class1_weight = float(y.mean())
            imbalance_ratio = class0_weight / (class1_weight if class1_weight > 0 else 1e-9)

    return Xn, y, float(imbalance_ratio)


def resample(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray, list],
    *,
    method: str = "none",
    random_state: int = 42,
    **params: Any,
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray], Dict[str, Any]]:
    """General resampling front door.

    method:
      - none
      - smote
      - adasyn
      - gmm_rebalance (binary-only)

    Returns (X_res, y_res, info_dict).
    """

    method = (method or "none").lower()

    if method == "none":
        info = {"method": "none", "imbalance_ratio": None}
        return X, y, info

    if method in {"smote", "adasyn"}:
        import numpy as np

        # Ensure y is 1D numpy
        y_arr = np.asarray(y).ravel()

        # Count class samples
        classes, counts = np.unique(y_arr, return_counts=True)

        # If binary one-vs-rest, minority count is the smaller class
        if len(counts) < 2:
            # Only one class present → cannot resample
            info = {"method": f"{method}_skipped_single_class", "sampler": None}
            return X, y, info

        min_count = counts.min()

        try:
            if method == "smote":
                from imblearn.over_sampling import SMOTE

                # SMOTE requires at least 2 minority samples
                required_min = 2  # because k_neighbors=1

                if min_count < required_min:
                    info = {"method": "smote_skipped_too_few_samples", "sampler": None}
                    return X, y, info

                sampler = SMOTE(k_neighbors=1, random_state=random_state)

            else:
                from imblearn.over_sampling import ADASYN

                # ADASYN(n_neighbors=2) effectively requires >= 3 samples
                required_min = 3

                if min_count < required_min:
                    info = {"method": "adasyn_skipped_too_few_samples", "sampler": None}
                    return X, y, info

                sampler = ADASYN(n_neighbors=2, random_state=random_state)

        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Could not import imblearn. Install 'imbalanced-learn' to use SMOTE/ADASYN."
            ) from e

        X_res, y_res = sampler.fit_resample(X, y)
        info = {"method": method, "sampler": sampler}
        return X_res, y_res, info

    if method == "gmm_rebalance":
        X_res, y_res, ratio = rebalance_gmm(
            X,
            y,
            min_pos_fraction=float(params.get("min_pos_fraction", 0.33)),
            random_state=int(params.get("random_state", 0)),
        )
        info = {"method": method, "imbalance_ratio": ratio}
        return X_res, y_res, info

    raise ValueError(f"Unknown resampling method: {method}")
