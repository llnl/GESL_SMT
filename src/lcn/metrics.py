# Copyright (c) 2025, Lawrence Livermore National Security, LLC. All rights reserved. LLNL-CODE-2025335
# This work was produced at the Lawrence Livermore National Laboratory (LLNL) under contract no. DE-AC52-07NA27344 (Contract 44)
# between the U.S. Department of Energy (DOE) and Lawrence Livermore National Security, LLC (LLNS) for the operation of LLNL.
# See license for disclaimers, notice of U.S. Government Rights and license terms and conditions.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_secondary_metrics(group_results: pd.DataFrame) -> pd.DataFrame:
    """
    From a DataFrame with columns:
      ['Sig_ID', 'Group', 'Actual_Class', 'Predicted_Class',
       'Actual_Subclass', 'Predicted_Subclass']
    compute, for each Actual_Class (secondary label):
      - Accuracy_sec
      - hP_sec, hR_sec, hF_sec
    """
    rows = []
    beta = 1.0

    for sec_label in group_results["Actual_Class"].unique():
        df_sec = group_results[group_results["Actual_Class"] == sec_label]
        # 1) Accuracy on the secondary label
        acc_sec = (df_sec["Actual_Class"] == df_sec["Predicted_Class"]).mean() * 100

        # 2) Hierarchical precision/recall on the two-level path
        hP_num = hP_den = hR_num = hR_den = 0
        for _, row in df_sec.iterrows():
            actual_path = [row["Actual_Class"], row["Actual_Subclass"]]
            predicted_path = [row["Predicted_Class"], row["Predicted_Subclass"]]
            inter_len = len(set(actual_path) & set(predicted_path))

            hP_num += inter_len
            hP_den += len(predicted_path)
            hR_num += inter_len
            hR_den += len(actual_path)

        hP_sec = (hP_num / hP_den * 100) if hP_den > 0 else np.nan
        hR_sec = (hR_num / hR_den * 100) if hR_den > 0 else np.nan
        if (hP_sec + hR_sec) > 0:
            hF_sec = ((beta**2 + 1) * hP_sec * hR_sec) / (beta**2 * hP_sec + hR_sec)
        else:
            hF_sec = np.nan

        rows.append(
            {
                "Group": df_sec["Group"].iloc[0],
                "Secondary": sec_label,
                "Accuracy_sec": acc_sec,
                "hP_sec": hP_sec,
                "hR_sec": hR_sec,
                "hF_sec": hF_sec,
            }
        )

    return pd.DataFrame(rows)
