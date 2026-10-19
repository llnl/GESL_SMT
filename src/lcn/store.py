from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .config import PipelineConfig


def make_default_save_prefix(cfg: PipelineConfig) -> str:
    m = cfg.model.model_type
    w = "w" if cfg.model.weighting else "now"
    rs = "rs" if cfg.model.randomized_search else "nors"

    def rtag(rcfg):
        if rcfg.type == "none":
            return "none"
        if rcfg.type in ("smote", "adasyn"):
            return f"{rcfg.type}-{rcfg.scope}"
        if rcfg.type == "gmm":
            return f"gmm-t{rcfg.target_minority_fraction}"
        return rcfg.type

    sec = rtag(cfg.secondary_resampling)
    ter = rtag(cfg.tertiary_resampling)
    return f"{m}_{w}_{rs}__sec-{sec}__ter-{ter}"

class ClassifierStore:
    """Handles consistent save/load paths for per-label binary classifiers."""

    def __init__(self, root_dir: str, save_prefix: str):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.save_prefix = save_prefix

    def model_path(self, level: str, label: str) -> Path:
        safe_label = str(label).replace(os.sep, "_")
        return self.root_dir / f"{self.save_prefix}__{level}__{safe_label}.pickle"

    def save(self, level: str, label: str, model) -> Path:
        p = self.model_path(level, label)
        import pickle
        with open(p, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        return p

    def load(self, level: str, label: str):
        p = self.model_path(level, label)
        if not p.exists():
            raise FileNotFoundError(f"Classifier not found: {p}")
        import pickle
        with open(p, "rb") as f:
            return pickle.load(f)

    def list_saved_labels(self, level: str) -> list[str]:
        """List all saved classifier labels for a given level."""
        pattern = f"{self.save_prefix}__{level}__*.pickle"
        matching_files = self.root_dir.glob(pattern)

        labels = []
        prefix_to_remove = f"{self.save_prefix}__{level}__"

        for file in matching_files:
            # Extract label by removing the prefix
            # Format: {save_prefix}__{level}__{label}.pickle
            filename = file.stem  # Remove .pickle extension
            if filename.startswith(prefix_to_remove):
                label = filename[len(prefix_to_remove):]
                labels.append(label)

        return labels

    def save_config(self, cfg: PipelineConfig) -> Path:
        p = self.root_dir / f"{self.save_prefix}__config.json"
        with open(p, "w") as f:
            json.dump(asdict(cfg), f, indent=2)
        return p
