"""Metrics helpers used by the hierarchical LCN evaluation."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


def intersection(list_a: Iterable, list_b: Iterable) -> list:
    return [elem for elem in list_a if elem in list_b]


def hierarchical_metrics_from_results(
    group_results: pd.DataFrame,
    *,
    beta: float = 1.0,
) -> Tuple[float, float, float]:
    """Compute hP/hR/hF as implemented in your scripts.

    group_results must have columns:
      ['Actual_Class', 'Predicted_Class', 'Actual_Subclass', 'Predicted_Subclass']
    """

    hP_num = hP_denom = 0.0
    hR_num = hR_denom = 0.0

    for row in range(group_results.shape[0]):
        # Two-level path: class -> subclass
        actual_path = [group_results["Actual_Class"][row], group_results["Actual_Subclass"][row]]
        predicted_path = [group_results["Predicted_Class"][row], group_results["Predicted_Subclass"][row]]

        inter_len = len(set(actual_path) & set(predicted_path))

        hP_num += inter_len
        hP_denom += len(predicted_path)

        hR_num += inter_len
        hR_denom += len(actual_path)

    hP = (hP_num / hP_denom * 100.0) if hP_denom > 0 else float("nan")
    hR = (hR_num / hR_denom * 100.0) if hR_denom > 0 else float("nan")

    if (hP + hR) > 0:
        hF = ((beta**2 + 1) * hP * hR) / (beta**2 * hP + hR)
    else:
        hF = float("nan")

    return float(hP), float(hR), float(hF)


def compute_secondary_metrics(group_results: pd.DataFrame) -> pd.DataFrame:
    """Secondary-level (per secondary label) metrics.

    This is equivalent to the compute_secondary_metrics() you added in
    smt_lcn_with_rebalancing.py.

    Expects group_results with columns:
      ['Sig_ID', 'Group', 'Actual_Class', 'Predicted_Class',
       'Actual_Subclass', 'Predicted_Subclass']
    """

    rows = []
    beta = 1.0

    for sec_label in group_results["Actual_Class"].unique():
        df_sec = group_results[group_results["Actual_Class"] == sec_label]

        acc_sec = (df_sec["Actual_Class"] == df_sec["Predicted_Class"]).mean() * 100.0

        hP_num = hP_den = hR_num = hR_den = 0.0
        for _, row in df_sec.iterrows():
            actual_path = [row["Actual_Class"], row["Actual_Subclass"]]
            predicted_path = [row["Predicted_Class"], row["Predicted_Subclass"]]
            inter_len = len(set(actual_path) & set(predicted_path))

            hP_num += inter_len
            hP_den += len(predicted_path)
            hR_num += inter_len
            hR_den += len(actual_path)

        hP_sec = (hP_num / hP_den * 100.0) if hP_den > 0 else float("nan")
        hR_sec = (hR_num / hR_den * 100.0) if hR_den > 0 else float("nan")
        if (hP_sec + hR_sec) > 0:
            hF_sec = ((beta**2 + 1) * hP_sec * hR_sec) / (beta**2 * hP_sec + hR_sec)
        else:
            hF_sec = float("nan")

        rows.append(
            {
                "Group": df_sec["Group"].iloc[0] if len(df_sec) else None,
                "Secondary": sec_label,
                "Accuracy_sec": acc_sec,
                "hP_sec": hP_sec,
                "hR_sec": hR_sec,
                "hF_sec": hF_sec,
            }
        )

    return pd.DataFrame(rows)
