"""IO helpers."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_dataframe(csv_path: str, *, strip_prefix: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if strip_prefix:
        df.columns = [c.replace(strip_prefix, "") for c in df.columns]
    return df


def load_sec_to_tert(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r") as f:
        return json.load(f)
