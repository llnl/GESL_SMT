from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

from .config import PipelineConfig
from .inference import successive_binary_classifier
from .utils import intersection
from .store import ClassifierStore, make_default_save_prefix
from .training import train_for_group, train_on_dataframe


def validation(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    sec_to_tert_dict: Dict[str, List[str]],
    *,
    store: ClassifierStore,
    sec_labels: Optional[List[str]] = None,
    available_tert_labels: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, float, float, float, float]:
    """Hierarchical validation: secondary first, then tertiary conditioned on predicted secondary.

    By default it evaluates the provided dataframe. For K-fold, call it with the held-out fold only and
    pass label lists derived from the corresponding training fold so loading matches trained models.
    """
    df1 = (
        df.loc[df["primary"] == cfg.primary_group]
        .copy()
        .dropna()
        .reset_index(drop=True)
    )

    if df1.shape[0] == 0:
        empty = pd.DataFrame()
        return empty, 0.0, 0.0, 0.0, 0.0

    if sec_labels is None:
        sec_labels = (
            df1.groupby(["secondary"]).size().sort_values(ascending=False).index.to_list()
        )
    if available_tert_labels is None:
        available_tert_labels = (
            df1.groupby(["tertiary"]).size().sort_values(ascending=False).index.to_list()
        )
    available_tert_labels = list(available_tert_labels)

    X_test = df1.iloc[:, cfg.feature_start_col :].to_numpy()

    sec_pred = []
    for i in range(X_test.shape[0]):
        pred = successive_binary_classifier(
            X_test[i : i + 1, :], sec_labels, store=store, level="secondary"
        )
        sec_pred.append(pred)
    sec_pred = np.asarray(sec_pred)

    tert_pred = []
    for i in range(X_test.shape[0]):
        sec_hat = str(sec_pred[i])
        tert_list = [t for t in sec_to_tert_dict.get(sec_hat, []) if t in available_tert_labels]
        if len(tert_list) == 0:
            tert_list = available_tert_labels
        pred = successive_binary_classifier(
            X_test[i : i + 1, :], tert_list, store=store, level="tertiary"
        )
        tert_pred.append(pred)
    tert_pred = np.asarray(tert_pred)

    sig_col = "Sig_ID" if "Sig_ID" in df1.columns else ("signature_id" if "signature_id" in df1.columns else df1.columns[0])

    group_results = pd.DataFrame(
        columns=[
            "Sig_ID",
            "Group",
            "Actual_Class",
            "Predicted_Class",
            "Actual_Subclass",
            "Predicted_Subclass",
        ]
    )
    group_results["Sig_ID"] = df1[sig_col].to_numpy()
    group_results["Group"] = df1["primary"].to_numpy()
    group_results["Actual_Class"] = df1["secondary"].to_numpy()
    group_results["Predicted_Class"] = sec_pred
    group_results["Actual_Subclass"] = df1["tertiary"].to_numpy()
    group_results["Predicted_Subclass"] = tert_pred

    acc, hP, hR, hF = _compute_hierarchical_metrics(group_results)
    return group_results, acc, hP, hR, hF



def run_pipeline(
    df: pd.DataFrame,
    sec_to_tert_dict: Dict[str, List[str]],
    cfg: PipelineConfig,
) -> Tuple[pd.DataFrame, float, float, float, float]:
    save_prefix = cfg.save_prefix or make_default_save_prefix(cfg)
    base_store = ClassifierStore(cfg.classifiers_dir, save_prefix)
    base_store.save_config(replace(cfg, save_prefix=save_prefix))

    if cfg.validation_mode == "kfold":
        return _run_kfold_pipeline(df, sec_to_tert_dict, replace(cfg, save_prefix=save_prefix))

    if cfg.train:
        train_for_group(df, cfg, store=base_store, random_state=cfg.random_state)

    results, acc, hP, hR, hF = validation(df, cfg, sec_to_tert_dict, store=base_store)
    _write_holdout_outputs(results, acc, hP, hR, hF, cfg, save_prefix)
    return results, acc, hP, hR, hF


def _run_kfold_pipeline(
    df: pd.DataFrame,
    sec_to_tert_dict: Dict[str, List[str]],
    cfg: PipelineConfig,
) -> Tuple[pd.DataFrame, float, float, float, float]:
    import os

    os.makedirs(cfg.outputs_dir, exist_ok=True)
    df1 = (
        df.loc[df["primary"] == cfg.primary_group]
        .copy()
        .dropna()
        .reset_index(drop=True)
    )
    if df1.empty:
        empty = pd.DataFrame()
        return empty, 0.0, 0.0, 0.0, 0.0

    splitter = _make_splitter(df1, cfg)
    all_results = []
    fold_metrics_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter, start=1):
        train_df = df1.iloc[train_idx].copy().reset_index(drop=True)
        test_df = df1.iloc[test_idx].copy().reset_index(drop=True)

        fold_prefix = f"{cfg.save_prefix}__fold{fold_idx:02d}"
        fold_store = ClassifierStore(cfg.classifiers_dir, fold_prefix)
        fold_store.save_config(replace(cfg, save_prefix=fold_prefix))

        if cfg.train:
            train_on_dataframe(train_df, cfg, store=fold_store, random_state=cfg.random_state + fold_idx)

        sec_labels = train_df.groupby(["secondary"]).size().sort_values(ascending=False).index.to_list()
        tert_labels = train_df.groupby(["tertiary"]).size().sort_values(ascending=False).index.to_list()

        results, acc, hP, hR, hF = validation(
            test_df,
            cfg,
            sec_to_tert_dict,
            store=fold_store,
            sec_labels=sec_labels,
            available_tert_labels=tert_labels,
        )
        results.insert(0, "Fold", fold_idx)
        all_results.append(results)
        fold_metrics_rows.append({"fold": fold_idx, "n_test": len(test_df), "acc": acc, "hP": hP, "hR": hR, "hF": hF})

        fold_results_path = f"{cfg.outputs_dir}/{cfg.save_prefix}_{cfg.primary_group}_fold{fold_idx:02d}_results.csv"
        results.to_csv(fold_results_path, index=False)

    all_results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    fold_metrics_df = pd.DataFrame(fold_metrics_rows)
    summary_df = _make_summary_df(fold_metrics_df)

    all_results_df.to_csv(f"{cfg.outputs_dir}/{cfg.save_prefix}_{cfg.primary_group}_kfold_results_all.csv", index=False)
    fold_metrics_df.to_csv(f"{cfg.outputs_dir}/{cfg.save_prefix}_{cfg.primary_group}_kfold_metrics_by_fold.csv", index=False)
    summary_df.to_csv(f"{cfg.outputs_dir}/{cfg.save_prefix}_{cfg.primary_group}_kfold_metrics_summary.csv", index=False)

    with open(f"{cfg.outputs_dir}/{cfg.save_prefix}_{cfg.primary_group}_kfold_metrics_summary.txt", "w") as f:
        f.write(f"validation_mode=kfold\n")
        f.write(f"n_splits={cfg.n_splits}\n")
        f.write(f"cv_stratify={cfg.cv_stratify}\n")
        for metric in ["acc", "hP", "hR", "hF"]:
            vals = fold_metrics_df[metric].astype(float).to_numpy()
            f.write(f"{metric}_mean={np.mean(vals):.4f}\n")
            f.write(f"{metric}_std={np.std(vals, ddof=1) if len(vals) > 1 else 0.0:.4f}\n")

    if fold_metrics_df.empty:
        return all_results_df, 0.0, 0.0, 0.0, 0.0

    return (
        all_results_df,
        float(fold_metrics_df["acc"].mean()),
        float(fold_metrics_df["hP"].mean()),
        float(fold_metrics_df["hR"].mean()),
        float(fold_metrics_df["hF"].mean()),
    )


def _make_splitter(df1: pd.DataFrame, cfg: PipelineConfig):
    y = None
    if cfg.cv_stratify == "secondary":
        y = df1["secondary"].astype(str).to_numpy()
    elif cfg.cv_stratify == "hierarchy":
        y = (df1["secondary"].astype(str) + "__" + df1["tertiary"].astype(str)).to_numpy()

    if y is not None:
        try:
            skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=cfg.cv_shuffle, random_state=cfg.random_state)
            return skf.split(df1, y)
        except ValueError:
            pass

    kf = KFold(n_splits=cfg.n_splits, shuffle=cfg.cv_shuffle, random_state=cfg.random_state)
    return kf.split(df1)


def _compute_hierarchical_metrics(group_results: pd.DataFrame) -> Tuple[float, float, float, float]:
    correct = (group_results["Actual_Subclass"] == group_results["Predicted_Subclass"]).sum()
    acc = correct / group_results.shape[0] * 100.0 if group_results.shape[0] else 0.0

    hP_num = hP_denom = hR_num = hR_denom = 0
    beta = 1.0

    for row in range(group_results.shape[0]):
        actual_path = [group_results["Actual_Class"].iloc[row], group_results["Actual_Subclass"].iloc[row]]
        predicted_path = [group_results["Predicted_Class"].iloc[row], group_results["Predicted_Subclass"].iloc[row]]
        inter = intersection(actual_path, predicted_path)
        hP_num += len(inter)
        hP_denom += len(predicted_path)
        hR_num += len(inter)
        hR_denom += len(actual_path)

    hP = (hP_num / hP_denom) * 100.0 if hP_denom else 0.0
    hR = (hR_num / hR_denom) * 100.0 if hR_denom else 0.0
    hF = ((beta ** 2 + 1) * hP * hR) / (beta**2 * hP + hR) if (beta**2 * hP + hR) else 0.0
    return acc, hP, hR, hF


def _make_summary_df(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["acc", "hP", "hR", "hF"]:
        vals = fold_metrics_df[metric].astype(float).to_numpy()
        rows.append({
            "metric": metric,
            "mean": float(np.mean(vals)) if len(vals) else 0.0,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)) if len(vals) else 0.0,
            "max": float(np.max(vals)) if len(vals) else 0.0,
        })
    return pd.DataFrame(rows)


def _write_holdout_outputs(results: pd.DataFrame, acc: float, hP: float, hR: float, hF: float, cfg: PipelineConfig, save_prefix: str) -> None:
    import os
    os.makedirs(cfg.outputs_dir, exist_ok=True)
    out_path = f"{cfg.outputs_dir}/{save_prefix}_{cfg.primary_group}_results.csv"
    results.to_csv(out_path, index=False)

    metrics_path = f"{cfg.outputs_dir}/{save_prefix}_{cfg.primary_group}_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"validation_mode=holdout\n")
        f.write(f"acc={acc:.4f}\n")
        f.write(f"hP={hP:.4f}\n")
        f.write(f"hR={hR:.4f}\n")
        f.write(f"hF={hF:.4f}\n")
