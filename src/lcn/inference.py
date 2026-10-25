from __future__ import annotations

from typing import List

import numpy as np

from .store import ClassifierStore
import shap
from lime.lime_tabular import LimeTabularExplainer

FEATURE_NAMES = ["P001.df_mean","P001.df_std","P001.df_skew","P001.df_kurtosis","P001.f_mean","P001.f_std","P001.f_skew","P001.f_kurtosis","P001.ip_a_mean",'P001.ip_a_std','P001.ip_a_skew','P001.ip_a_kurtosis','P001.ip_m_mean','P001.ip_m_std','P001.ip_m_skew','P001.ip_m_kurtosis','P001.vp_a_mean','P001.vp_a_std','P001.vp_a_skew','P001.vp_a_kurtosis','P001.vp_m_mean','P001.vp_m_std','P001.vp_m_skew','P001.vp_m_kurtosis']

def classify_primary(
    test_features: np.ndarray,
    label_list: List[str],
    *,
    store: ClassifierStore,
) -> str:
    """Primary level classifier selection.

    Loads a binary classifier per primary label and selects the label that yields the max predicted probability.
    """
    return successive_binary_classifier(
        test_features,
        label_list,
        store=store,
        level="primary"
    )

from collections import Counter, defaultdict
import numpy as np

def merge_forest_paths(model, test_features, feature_names=None):
    paths, ptrs = model.decision_path(test_features)
    x = np.asarray(test_features).ravel()

    per_tree_steps = []

    for i in range(len(ptrs) - 1):
        start, end = ptrs[i], ptrs[i + 1]
        tree_nodes = paths[:, start:end].indices
        tree = model.estimators_[i]
        t = tree.tree_

        steps = []
        for node_id in tree_nodes:
            feature_id = t.feature[node_id]
            if feature_id == -2:
                continue

            threshold = t.threshold[node_id]
            value = x[feature_id]
            direction = "left" if value <= threshold else "right"

            left = t.children_left[node_id]
            right = t.children_right[node_id]
            gain = None
            if left != -1 and right != -1:
                gain = (
                    t.impurity[node_id]
                    - (t.n_node_samples[left] / t.n_node_samples[node_id]) * t.impurity[left]
                    - (t.n_node_samples[right] / t.n_node_samples[node_id]) * t.impurity[right]
                )

            steps.append({
                "node_id": int(node_id),
                "feature_id": int(feature_id),
                "feature_name": feature_names[feature_id] if feature_names else int(feature_id),
                "threshold": float(threshold),
                "value": float(value),
                "direction": direction,
                "gain": float(gain) if gain is not None else None,
            })

        per_tree_steps.append(steps)

    max_depth = max((len(s) for s in per_tree_steps), default=0)
    merged = []

    for depth in range(max_depth):
        depth_feats = []
        depth_values = defaultdict(list)
        depth_thresholds = defaultdict(list)
        depth_gains = defaultdict(list)

        for steps in per_tree_steps:
            if depth < len(steps):
                step = steps[depth]
                f = step["feature_id"]
                depth_feats.append(f)
                depth_values[f].append(step["value"])
                depth_thresholds[f].append(step["threshold"])
                if step["gain"] is not None:
                    depth_gains[f].append(step["gain"])
            else:
                depth_feats.append(-1)

        valid_feats = [f for f in depth_feats if f != -1]
        if valid_feats:
            common_feat, count = Counter(valid_feats).most_common(1)[0]
            total = len(valid_feats)
            support_pct = 100.0 * count / total
            avg_value = float(np.mean(depth_values[common_feat]))
            avg_threshold = float(np.mean(depth_thresholds[common_feat]))
            avg_gain = float(np.mean(depth_gains[common_feat])) if depth_gains[common_feat] else None
        else:
            common_feat = -1
            support_pct = 0.0
            avg_value = None
            avg_threshold = None
            avg_gain = None

        merged.append({
            "depth": depth,
            "feature_id": common_feat,
            "feature_name": feature_names[common_feat] if feature_names and common_feat != -1 else common_feat,
            "support_pct": support_pct,
            "avg_value": avg_value,
            "avg_threshold": avg_threshold,
            "avg_gain": avg_gain,
        })

    return merged


def successive_binary_classifier(
    test_features: np.ndarray,
    train_features: np.ndarray,
    label_list: List[str],
    *,
    store: ClassifierStore,
    level: str,
    compute_explanations: bool = False,
) -> str:
    """Successive (best-score) binary classifier selection.

    Loads a binary classifier per label and selects the label that yields the max predicted probability.

    level: 'secondary' or 'tertiary'
    compute_explanations: If True, compute SHAP, LIME, MDI, and path explanations (expensive). Default False.
    """
    outputs = []
    output_dict={}
    shaps = {}
    limes = {}
    merged_paths = {}
    mdis = {}  # Mean Decrease in Impurity (feature importances)

    for evttag in label_list:
        model = store.load(level, evttag)

        # Compute explanations only if requested
        if compute_explanations:
            # SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(test_features)

            # For binary classification, SHAP may return:
            # - A list [class_0_values, class_1_values], or
            # - A single array with interleaved [c0_f0, c1_f0, c0_f1, c1_f1, ...]
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # List format: take class 1 array
                shap_values = shap_values[1]
            else:
                # Single array: flatten first
                if shap_values.ndim > 1:
                    shap_values = shap_values.ravel()

                # Check if interleaved (length = 2 * num_features)
                expected_features = len([f for f in FEATURE_NAMES if f != 'P001.df_mean'])
                if len(shap_values) == 2 * expected_features:
                    # Interleaved format: extract class 1 values (every other value starting at index 1)
                    shap_values = shap_values[1::2]

            shaps[evttag] = shap_values

            # LIME explanations
            try:
                # Get actual feature names (exclude P001.df_mean which isn't in the data)
                actual_feature_names = [f for f in FEATURE_NAMES if f != 'P001.df_mean']

                print(f"LIME debug for {evttag}: train shape={train_features.shape}, test shape={test_features.shape}, n_features={len(actual_feature_names)}")

                # Clean training data (replace NaN with 0)
                train_clean = np.nan_to_num(train_features, nan=0.0)

                # Normalize both train and test to ensure similar scales
                # Compute robust statistics (use percentiles to avoid outlier influence)
                train_mins = np.percentile(train_clean, 1, axis=0)
                train_maxs = np.percentile(train_clean, 99, axis=0)
                train_ranges = train_maxs - train_mins
                # Avoid division by zero
                train_ranges[train_ranges < 1e-10] = 1.0

                # Normalize to [0, 1] range
                train_normalized = (train_clean - train_mins) / train_ranges
                test_normalized = (test_features - train_mins) / train_ranges

                # Clip to [0, 1] (in case test is outside training range)
                test_normalized = np.clip(test_normalized, 0, 1)

                # Use moderate kernel width on normalized data
                kernel_width_val = np.sqrt(train_features.shape[1]) * 0.75

                # Check model prediction on test sample first
                test_pred = model.predict_proba(test_features)[0]
                print(f"  Model prediction for {evttag}: {test_pred}")

                lime_explainer = LimeTabularExplainer(
                    training_data=train_normalized,
                    feature_names=actual_feature_names,
                    class_names=['0', '1'],
                    mode="classification",
                    discretize_continuous=False,
                    kernel_width=kernel_width_val,
                    feature_selection='none'  # Use all features
                )

                # Wrap prediction function to denormalize before passing to model
                def debug_predict_fn(X_normalized):
                    # Denormalize back to original scale
                    X_original = X_normalized * train_ranges + train_mins
                    preds = model.predict_proba(X_original)
                    if len(preds) > 10:
                        pred_class1 = preds[:, 1]
                        print(f"    Sample predictions for {evttag}: min={pred_class1.min():.3f}, max={pred_class1.max():.3f}, std={pred_class1.std():.3f}")
                    return preds

                exp = lime_explainer.explain_instance(
                    data_row=test_normalized.ravel(),
                    predict_fn=debug_predict_fn,
                    num_samples=5000,
                    num_features=len(actual_feature_names)  # Return all 23 features
                )
                limes[evttag] = exp.as_list()

                # Debug: check actual values
                non_zero_count = sum(1 for _, val in limes[evttag] if abs(val) > 1e-10)
                print(f"LIME success for {evttag}: {len(limes[evttag])} feature importances, {non_zero_count} non-zero")

                if non_zero_count > 0:
                    print(f"  Sample non-zero values: {[(f, v) for f, v in limes[evttag] if abs(v) > 1e-10][:3]}")
            except Exception as e:
                # Handle any LIME failures gracefully
                print(f"LIME failed for {evttag}: {e}. Using fallback.")
                limes[evttag] = [(feat, 0.0) for feat in actual_feature_names if feat != 'P001.df_mean']

            # Mean Decrease in Impurity (MDI) - Random Forest feature importances
            # Get the actual estimator if wrapped in search CV
            actual_model = model
            if hasattr(model, 'best_estimator_'):
                actual_model = model.best_estimator_

            if hasattr(actual_model, 'feature_importances_'):
                mdis[evttag] = actual_model.feature_importances_
            else:
                mdis[evttag] = np.zeros(len(FEATURE_NAMES))

            # Decision path explanations
            # Use 23 feature names (excluding P001.df_mean) to match training data
            actual_feature_names = [f for f in FEATURE_NAMES if f != 'P001.df_mean']
            merged_paths[evttag] = merge_forest_paths(
                model,
                test_features,
                feature_names=actual_feature_names
            )

            items = [s for s in merged_paths[evttag] if s["avg_gain"] is not None]
            top_items = sorted(items, key=lambda s: s["avg_gain"], reverse=True)[:10]
            top_items = sorted(top_items, key=lambda s: s["depth"])
            merged_paths[evttag] = top_items

        # Always compute probabilities
        proba = _predict_proba(model, test_features)
        # take probability of class 1
        if proba.ndim == 2 and proba.shape[1] >= 2:
            outputs.append(float(proba[:, 1].ravel()[0]))
            output_dict[evttag] = float(proba[:, 1].ravel()[0])
        else:
            outputs.append(float(proba.ravel()[0]))
            output_dict[evttag] = float(proba.ravel()[0])

    return output_dict, shaps, limes, merged_paths, mdis


def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    # RandomizedSearchCV stores best_estimator_ after fit; but predict_proba should work on wrapper too.
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))
    if hasattr(model, "best_estimator_") and hasattr(model.best_estimator_, "predict_proba"):
        return np.asarray(model.best_estimator_.predict_proba(X))
    # fallback: use decision_function then sigmoid-ish scaling
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X))
        s = 1.0 / (1.0 + np.exp(-s))
        return np.column_stack([1 - s, s])
    raise AttributeError("Model does not support probability prediction.")
