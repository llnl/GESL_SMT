"""Model loading + hierarchical (LCN) inference / validation."""

from __future__ import annotations

import os
import pickle
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .metrics import hierarchical_metrics_from_results


def model_path(save_dir: str, save_prefix: str, label: str) -> str:
    return os.path.join(save_dir, f"{save_prefix}_{label}.pickle")


def successive_binary_classifier(
    test_features: pd.DataFrame,
    label_list: Sequence[str],
    *,
    save_dir: str,
    save_prefix: str,
    skip_missing: bool = True,
) -> Tuple[str, np.ndarray]:
    """Run a single sample against a list of one-vs-rest binary classifiers.

    Returns:
      (predicted_label, prob_table)

    prob_table is a numpy array of shape (k, 2): [label, positive_probability]
    for every model that was found + evaluated.
    """

    proba_temp = -1.0
    pred_label = ""
    prob_static: list[list[object]] = []

    for evttag in label_list:
        fpath = model_path(save_dir, save_prefix, evttag)
        if skip_missing and not os.path.exists(fpath):
            continue

        with open(fpath, "rb") as handle:
            model = pickle.load(handle)

        proba = model.predict_proba(test_features)
        pos = float(proba[0][1])
        prob_static.append([evttag, pos])

        if pos > proba_temp:
            proba_temp = pos
            pred_label = evttag

    prob_static_arr = np.array(prob_static, dtype=object) if len(prob_static) else np.empty((0, 2), dtype=object)
    return pred_label, prob_static_arr


def validation(
    df: pd.DataFrame,
    group_label: str,
    sec_to_tert_dict: dict,
    *,
    save_dir: str,
    save_prefix: str,
    test_size: float = 0.05,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, float, float, float, float]:
    """Hierarchical (LCN) validation for one group label."""

    df1 = df.loc[df["primary"] == group_label].copy().dropna().reset_index()

    if df1.shape[0] == 0:
        empty = pd.DataFrame(
            columns=["Sig_ID", "Group", "Actual_Class", "Predicted_Class", "Actual_Subclass", "Predicted_Subclass"]
        )
        return empty, 0.0, 0.0, 0.0, 0.0

    arr_class = np.asanyarray(df1.groupby(["secondary"]).nunique().sort_values(by=["index"], ascending=False).index)

    features = df1.iloc[:, -(df1.shape[1] - 5):]
    label = df1["secondary"]
    _, x_test, _, _ = train_test_split(features, label, test_size=test_size, random_state=random_state)

    sigid: list = []
    group: list = []
    actual_class: list = []
    actual_subclass: list = []
    pred_class: list = []
    pred_subclass: list = []

    for idx in x_test.index:
        test_features = x_test[x_test.index == idx]

        sigid.append(df1.iloc[idx]["sigid"])
        group.append(group_label)
        actual_class.append(df1.iloc[idx]["secondary"])
        actual_subclass.append(df1.iloc[idx]["tertiary"])

        pred_class_1, _ = successive_binary_classifier(
            test_features,
            arr_class,
            save_dir=save_dir,
            save_prefix=save_prefix,
        )
        pred_class.append(pred_class_1)

        tert_list = sec_to_tert_dict.get(pred_class_1, [])
        pred_subclass_1, _ = successive_binary_classifier(
            test_features,
            tert_list,
            save_dir=save_dir,
            save_prefix=save_prefix,
        )
        pred_subclass.append(pred_subclass_1)

    group_results = pd.DataFrame(
        {
            "Sig_ID": sigid,
            "Group": group,
            "Actual_Class": actual_class,
            "Predicted_Class": pred_class,
            "Actual_Subclass": actual_subclass,
            "Predicted_Subclass": pred_subclass,
        }
    )

    # Match your scripts: accuracy is evaluated at the subclass level
    correct = (group_results["Actual_Subclass"] == group_results["Predicted_Subclass"]).sum()
    acc = float(correct) / float(len(group_results)) * 100.0 if len(group_results) else 0.0

    hP, hR, hF = hierarchical_metrics_from_results(group_results)

    return group_results, acc, hP, hR, hF
