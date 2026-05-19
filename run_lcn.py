from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from lcn.config import ModelConfig, PipelineConfig, ResamplingConfig
from lcn.data_io import load_inputs
from lcn.pipeline import run_pipeline


def _json_or_empty(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    return json.loads(s)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run LCN hierarchical classification with interchangeable model + augmentation.")

    p.add_argument("--features", default="Inputs/pmu_transmission_features.csv", help="Path to features CSV")
    p.add_argument("--sec-to-tert", default="Inputs/sec_to_tert_dict_transmission.json", help="Path to secondary->tertiary JSON")
    p.add_argument("--group", default="A", help="Primary group label to train/validate (df['primary'])")

    p.add_argument("--model", choices=["rf", "xgb"], default="rf", help="Model type")
    p.add_argument("--weighting", action="store_true", help="Enable class-imbalance weighting (RF: class_weight='balanced', XGB: scale_pos_weight)")
    p.add_argument("--random-search", action="store_true", help="For XGB only: wrap with RandomizedSearchCV (tune subsample/colsample_bytree/scale_pos_weight)")
    p.add_argument("--model-params", default="", help="JSON dict of model constructor params")
    p.add_argument("--search-params", default="", help="JSON dict of search params")

    p.add_argument("--sec-aug", choices=["none", "smote", "adasyn", "gmm"], default="adasyn", help="Secondary augmentation method")
    p.add_argument("--sec-scope", choices=["dataset", "binary"], default="dataset", help="Where to apply secondary SMOTE/ADASYN (ignored for GMM)")
    p.add_argument("--sec-aug-params", default="", help="JSON dict of secondary sampler params")
    p.add_argument("--sec-gmm-target", type=float, default=0.33, help="For GMM: target minority fraction (default 0.33)")

    p.add_argument("--ter-aug", choices=["none", "smote", "adasyn", "gmm"], default="smote", help="Tertiary augmentation method")
    p.add_argument("--ter-scope", choices=["dataset", "binary"], default="dataset", help="Where to apply tertiary SMOTE/ADASYN (ignored for GMM)")
    p.add_argument("--ter-aug-params", default="", help="JSON dict of tertiary sampler params")
    p.add_argument("--ter-gmm-target", type=float, default=0.33, help="For GMM: target minority fraction (default 0.33)")

    p.add_argument("--validation-mode", choices=["holdout", "kfold"], default="holdout", help="Validation approach")
    p.add_argument("--n-splits", type=int, default=5, help="Number of folds for outer K-fold validation")
    p.add_argument("--cv-stratify", choices=["secondary", "hierarchy", "none"], default="secondary", help="How to stratify outer folds")
    p.add_argument("--random-state", type=int, default=42, help="Random seed used by splitters/models where applicable")

    p.add_argument("--no-train", action="store_true", help="Skip training and only run validation using existing saved classifiers")
    p.add_argument("--save-prefix", default="", help="Optional explicit save prefix. If omitted, one is auto-generated.")

    p.add_argument("--classifiers-dir", default="Classifiers")
    p.add_argument("--outputs-dir", default="Outputs")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    df, sec_to_tert = load_inputs(args.features, args.sec_to_tert)

    model_cfg = ModelConfig(
        model_type=args.model,
        model_params=_json_or_empty(args.model_params),
        weighting=bool(args.weighting),
        randomized_search=bool(args.random_search),
        search_params=_json_or_empty(args.search_params),
    )

    sec_aug_cfg = ResamplingConfig(
        type=args.sec_aug,
        scope=args.sec_scope,
        params=_json_or_empty(args.sec_aug_params),
        target_minority_fraction=float(args.sec_gmm_target),
    )

    ter_aug_cfg = ResamplingConfig(
        type=args.ter_aug,
        scope=args.ter_scope,
        params=_json_or_empty(args.ter_aug_params),
        target_minority_fraction=float(args.ter_gmm_target),
    )

    cfg = PipelineConfig(
        primary_group=args.group,
        model=model_cfg,
        secondary_resampling=sec_aug_cfg,
        tertiary_resampling=ter_aug_cfg,
        outputs_dir=args.outputs_dir,
        classifiers_dir=args.classifiers_dir,
        save_prefix=(args.save_prefix or None),
        train=(not args.no_train),
        validation_mode=args.validation_mode,
        n_splits=int(args.n_splits),
        cv_stratify=args.cv_stratify,
        random_state=int(args.random_state),
    )

    run_pipeline(df=df, sec_to_tert_dict=sec_to_tert, cfg=cfg)


if __name__ == "__main__":
    main()
