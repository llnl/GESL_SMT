from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import json


current_results = {}

from src.lcn.config import ModelConfig, PipelineConfig, ResamplingConfig
from src.lcn.data_io import load_inputs
from src.lcn.store import ClassifierStore
from src.lcn.training import train_for_group
from src.lcn.inference import successive_binary_classifier
import numpy as np

FEATURE_NAMES = ["P001.df_mean","P001.df_std","P001.df_skew","P001.df_kurtosis","P001.f_mean","P001.f_std","P001.f_skew","P001.f_kurtosis","P001.ip_a_mean",'P001.ip_a_std','P001.ip_a_skew','P001.ip_a_kurtosis','P001.ip_m_mean','P001.ip_m_std','P001.ip_m_skew','P001.ip_m_kurtosis','P001.vp_a_mean','P001.vp_a_std','P001.vp_a_skew','P001.vp_a_kurtosis','P001.vp_m_mean','P001.vp_m_std','P001.vp_m_skew','P001.vp_m_kurtosis']

def format_results(outputs, shap_values, lime_values, paths, mdi_values, filter_nonspecific=False):
    predictions = []
    explanations = {}
    Paths = {}

    for label, confidence in outputs.items():
        # Skip nonspecific tertiary labels if filtering is enabled
        if filter_nonspecific and 'nonspecific' in label.lower():
            continue
        predictions.append({
            'label': label,
            'confidence': float(confidence)
        })

        shap_array = shap_values.get(label, [])
        lime_array = lime_values.get(label, [])
        mdi_array = mdi_values.get(label, [])

        lime_features = [f for f, _ in lime_array]
        lime_vals = [v for _, v in lime_array]

        # Handle SHAP values
        shap_lookup = {}
        if hasattr(shap_array, 'tolist'):
            shap_array = shap_array.tolist()

        # Use actual feature names (excluding P001.df_mean)
        actual_feature_names = [f for f in FEATURE_NAMES if f != 'P001.df_mean']

        for i, feature in enumerate(actual_feature_names):
            if i < len(shap_array):
                shap_lookup[feature] = shap_array[i]

        aligned_shap = [shap_lookup.get(f, 0.0) for f in lime_features]

        # Handle MDI values
        mdi_lookup = {}
        if hasattr(mdi_array, 'tolist'):
            mdi_array = mdi_array.tolist()

        for i, feature in enumerate(actual_feature_names):
            if i < len(mdi_array):
                mdi_lookup[feature] = mdi_array[i]

        aligned_mdi = [mdi_lookup.get(f, 0.0) for f in lime_features]

        explanations[label] = {
            'features': lime_features,
            'shap': aligned_shap,
            'lime': lime_vals,
            'mdi': aligned_mdi
        }

        Paths[label] = paths.get(label, [])

    return predictions, explanations, Paths


def load_sample():
    global current_results
    # Note: sec_to_tert file doesn't exist, pass None
    df = pd.read_csv("./Data/pmu_transmission_features.csv")
    prim_labels = list(df["primary"].unique())
    sec_labels = list(df["secondary"].unique())
    ter_labels = list(df["tertiary"].unique())

    cfg = PipelineConfig(
        primary_group="State",
        model=ModelConfig(model_type="rf", weighting=True),
        secondary_resampling=ResamplingConfig(type="adasyn"),
        tertiary_resampling=ResamplingConfig(type="adasyn"),
        classifiers_dir="Classifiers",
        outputs_dir="Outputs"
    )

    store = ClassifierStore(cfg.classifiers_dir, "default")

    X_test = df.iloc[:, cfg.feature_start_col :].to_numpy() # TODO: Replace with actual File Loading from User
    X_train = df.iloc[:, cfg.feature_start_col :].to_numpy()

    i=0

    outputs_prim, shap_values_prim, lime_values_prim, paths_prim, mdi_values_prim = successive_binary_classifier(
        X_test[i : i + 1, :], X_train, prim_labels, store=store, level="primary", compute_explanations=True
    )
    outputs_sec, shap_values_sec, lime_values_sec, paths_sec, mdi_values_sec = successive_binary_classifier(
        X_test[i : i + 1, :], X_train, sec_labels, store=store, level="secondary", compute_explanations=True
    )
    outputs_ter, shap_values_ter, lime_values_ter, paths_ter, mdi_values_ter = successive_binary_classifier(
        X_test[i : i + 1, :], X_train, ter_labels, store=store, level="tertiary", compute_explanations=True
    )

    result_prim = format_results(
        outputs_prim,
        shap_values=shap_values_prim,
        lime_values=lime_values_prim,
        paths=paths_prim,
        mdi_values=mdi_values_prim
    )

    result_sec = format_results(
        outputs_sec,
        shap_values=shap_values_sec,
        lime_values=lime_values_sec,
        paths=paths_sec,
        mdi_values=mdi_values_sec
    )

    result_ter = format_results(
        outputs_ter,
        shap_values=shap_values_ter,
        lime_values=lime_values_ter,
        paths=paths_ter,
        mdi_values=mdi_values_ter,
        filter_nonspecific=True
    )

    current_results = {
        'predictions': {
            'primary': result_prim[0],
            'secondary': result_sec[0],
            'tertiary': result_ter[0]
        },
        'explanations': {
            'primary': result_prim[1],
            'secondary': result_sec[1],
            'tertiary': result_ter[1]
        },
        'paths': {
            'primary': result_prim[2],
            'secondary': result_sec[2],
            'tertiary': result_ter[2]
        }
    }



def get_path(level, label):
    path = current_results.get('paths', {}).get(level, {}).get(label, [])

def get_explanation(level, label):
    explanations = current_results.get('explanations', {}).get(level, {}).get(label, {})

    features = explanations.get('features', [])
    shap_raw = explanations.get('shap', [])
    lime_raw = explanations.get('lime', [])

    shap_values = list(shap_raw)

    if lime_raw and isinstance(lime_raw[0], (list, tuple)) and len(lime_raw[0]) == 2:
        lime_features = [f for f, _ in lime_raw]
        lime_values = [v for _, v in lime_raw]
    else:
        lime_features = features[:len(lime_raw)]
        lime_values = list(lime_raw)


if __name__ == "__main__":
    load_sample()
    with open('xai.json', 'w') as f:
        json.dump(current_results, f)