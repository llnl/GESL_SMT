"""End-to-end runner for the modular pipeline."""

from __future__ import annotations

import pandas as pd

from .config import LCNConfig
from .inference import validation
from .io_utils import ensure_dir, load_dataframe, load_sec_to_tert
from .metrics import compute_secondary_metrics
from .train import train_for_group


def run(cfg: LCNConfig) -> None:
    """Run training (optional) + evaluation and write output CSV files."""

    ensure_dir(cfg.outputs_dir)

    df = load_dataframe(cfg.input_csv, strip_prefix=None)
    sec_to_tert_dict = load_sec_to_tert(cfg.sec_to_tert_json)

    save_prefix = cfg.resolved_save_prefix()

    if cfg.train_models:
        for group_label in cfg.group_labels:
            train_for_group(df, group_label, cfg)

    acc: list[float] = []
    hP: list[float] = []
    hR: list[float] = []
    hF: list[float] = []

    all_results: list[pd.DataFrame] = []

    for group_label in cfg.group_labels:
        group_results, acc_g, hp_g, hr_g, hf_g = validation(
            df,
            group_label,
            sec_to_tert_dict,
            save_dir=cfg.save_dir,
            save_prefix=save_prefix,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
        )
        all_results.append(group_results)
        acc.append(acc_g)
        hP.append(hp_g)
        hR.append(hr_g)
        hF.append(hf_g)

    group_results_df = pd.concat(all_results, ignore_index=True) if len(all_results) else pd.DataFrame()

    # Save per-sample results
    test_path = f"{cfg.outputs_dir}/lcn_{save_prefix}_test_results.csv"
    group_results_df.to_csv(test_path, index=False)

    # Save primary-level summary metrics (by group label)
    metrics_df = pd.DataFrame(
        {
            "Group": list(cfg.group_labels),
            "Accuracy": acc,
            "hP": hP,
            "hR": hR,
            "hF": hF,
        }
    )

    metrics_path = f"{cfg.outputs_dir}/lcn_{save_prefix}_metrics_results.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Save secondary-level metrics (optional but useful; matches your XGB script addition)
    sec_metrics_df = compute_secondary_metrics(group_results_df) if not group_results_df.empty else pd.DataFrame()
    sec_path = f"{cfg.outputs_dir}/lcn_{save_prefix}_secondary_metrics_results.csv"
    sec_metrics_df.to_csv(sec_path, index=False)

    print(f"[done] Wrote: {test_path}")
    print(f"[done] Wrote: {metrics_path}")
    print(f"[done] Wrote: {sec_path}")
