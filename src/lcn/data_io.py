# Copyright (c) 2025, Lawrence Livermore National Security, LLC. All rights reserved. LLNL-CODE-2025335
# This work was produced at the Lawrence Livermore National Laboratory (LLNL) under contract no. DE-AC52-07NA27344 (Contract 44)
# between the U.S. Department of Energy (DOE) and Lawrence Livermore National Security, LLC (LLNS) for the operation of LLNL.
# See license for disclaimers, notice of U.S. Government Rights and license terms and conditions.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import pandas as pd


def load_inputs(
    features_csv_path: str = "Inputs/pmu_transmission_features.csv",
    sec_to_tert_json_path: str = "Inputs/sec_to_tert_dict_transmission.json",
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Load the features dataframe and secondary->tertiary mapping."""
    df = pd.read_csv(features_csv_path)
    with open(sec_to_tert_json_path, "r") as f:
        sec_to_tert_dict = json.load(f)
    return df, sec_to_tert_dict
