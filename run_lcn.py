"""Entry point for the modular LCN pipeline.

Example usage:
  # RF + ADASYN for secondary and SMOTE for tertiary (similar to smt_lcn.py)
  python run_lcn.py --model rf --secondary-resampler adasyn --secondary-resampler-params '{"n_neighbors": 2}' \
                    --tertiary-resampler smote --tertiary-resampler-params '{"k_neighbors": 1}' \
                    --sampler-scope global --save-prefix RF_ADASYN_SMOTE

  # XGB + GMM rebalance + scale_pos_weight tuning (similar to smt_lcn_with_rebalancing.py)
  python run_lcn.py --model xgb --secondary-resampler gmm_rebalance --tertiary-resampler gmm_rebalance \
                    --weighting scale_pos_weight --tune-xgb --save-prefix XGB_gmm_spw
"""

from __future__ import annotations

import argparse
import json

from lcn_modular.config import LCNConfig
from lcn_modular.pipeline import run


def _json_dict(s: str) -> dict:
    if not s:
        return {}
    return json.loads(s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run modular LCN training + evaluation")

    p.add_argument("--input-csv", default="Inputs/pmu_transmission_features.csv")
    p.add_argument("--sec-to-tert-json", default="Inputs/sec_to_tert_dict_transmission.json")
    p.add_argument("--save-dir", default="Classifiers")
    p.add_argument("--outputs-dir", default="Outputs")

    p.add_argument("--model", choices=["rf", "xgb"], default="rf")
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--max-depth", type=int, default=8)

    p.add_argument("--secondary-resampler", choices=["none", "smote", "adasyn", "gmm_rebalance"], default="none")
    p.add_argument("--tertiary-resampler", choices=["none", "smote", "adasyn", "gmm_rebalance"], default="none")
    p.add_argument("--secondary-resampler-params", default="", help="JSON dict, e.g. '{\"n_neighbors\": 2}'")
    p.add_argument("--tertiary-resampler-params", default="", help="JSON dict, e.g. '{\"k_neighbors\": 1}'")

    p.add_argument("--sampler-scope", choices=["per_label", "global"], default="per_label")

    p.add_argument("--weighting", choices=["none", "balanced", "scale_pos_weight"], default="none")

    p.add_argument("--tune-xgb", action="store_true")
    p.add_argument("--no-tune-xgb", dest="tune_xgb", action="store_false")
    p.set_defaults(tune_xgb=True)

    p.add_argument("--save-prefix", default="", help="Prefix used in model pickle filenames and output CSV names")

    p.add_argument("--train", dest="train_models", action="store_true")
    p.add_argument("--no-train", dest="train_models", action="store_false")
    p.set_defaults(train_models=True)

    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = LCNConfig(
        input_csv=args.input_csv,
        sec_to_tert_json=args.sec_to_tert_json,
        save_dir=args.save_dir,
        outputs_dir=args.outputs_dir,
        model=args.model,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        secondary_resampler=args.secondary_resampler,
        tertiary_resampler=args.tertiary_resampler,
        secondary_resampler_params=_json_dict(args.secondary_resampler_params),
        tertiary_resampler_params=_json_dict(args.tertiary_resampler_params),
        sampler_scope=args.sampler_scope,
        weighting=args.weighting,
        tune_xgb=args.tune_xgb,
        save_prefix=args.save_prefix,
        train_models=args.train_models,
    )

    run(cfg)


if __name__ == "__main__":
    main()
