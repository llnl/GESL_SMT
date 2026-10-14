from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import dice_ml
import os
import re
import json
from openai import OpenAI

# Import existing infrastructure
from src.lcn.config import ModelConfig, PipelineConfig
from src.lcn.store import ClassifierStore
from src.lcn.inference import successive_binary_classifier
from sklearn.ensemble import RandomForestClassifier

FEATURE_NAMES = ["P001.df_mean","P001.df_std","P001.df_skew","P001.df_kurtosis","P001.f_mean","P001.f_std","P001.f_skew","P001.f_kurtosis","P001.ip_a_mean",'P001.ip_a_std','P001.ip_a_skew','P001.ip_a_kurtosis','P001.ip_m_mean','P001.ip_m_std','P001.ip_m_skew','P001.ip_m_kurtosis','P001.vp_a_mean','P001.vp_a_std','P001.vp_a_skew','P001.vp_a_kurtosis','P001.vp_m_mean','P001.vp_m_std','P001.vp_m_skew','P001.vp_m_kurtosis']


app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global state
current_data = None
actual_feature_names = None  # Will be populated on load
sample_feature_values = None  # Store actual input values
classifier_store = None  # Store for accessing models
test_sample = None  # Store test sample for counterfactuals
train_data_stats = None  # Store training data statistics for counterfactuals
df_global = None #Store for CounterFactual

def compute_features_from_raw(df_raw):
    """
    Compute statistical features from raw time-series data.
    Expected columns: Time, P001.df, P001.f, P001.ip_a, P001.ip_m, P001.vp_a, P001.vp_m

    IMPORTANT: Training data was normalized (z-score per signal) before computing statistics.
    We must do the same: normalize each signal, then compute mean/std/skew/kurtosis.

    Returns single-row DataFrame with 23 features (excluding P001.df_mean to match training data)
    """
    # Drop Time column
    if 'Time' in df_raw.columns:
        df_raw = df_raw.drop(columns=['Time'])

    features = {}

    # For each signal column: NORMALIZE FIRST, then compute statistics
    for col in df_raw.columns:
        values = df_raw[col].values

        # Z-score normalization: (x - mean) / std
        # This matches how training data was preprocessed
        # Use ddof=0 for population std (matches demo data preprocessing)
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=0)
        if std_val > 1e-10:
            normalized_values = (values - mean_val) / std_val
        else:
            normalized_values = values - mean_val  # Constant signal

        # Now compute statistics on NORMALIZED values
        # Use ddof=1 for sample statistics (matches demo data)
        # Skip mean for P001.df to match training data iloc[:, 5:] which starts after P001.df_mean
        if col != 'P001.df':
            features[f"{col}_mean"] = np.mean(normalized_values)

        features[f"{col}_std"] = np.std(normalized_values, ddof=1)
        features[f"{col}_skew"] = pd.Series(normalized_values).skew()
        features[f"{col}_kurtosis"] = pd.Series(normalized_values).kurtosis()

    # Create DataFrame with single row
    features_df = pd.DataFrame([features])

    # Expected feature names (23 features, excluding P001.df_mean)
    expected_features = [f for f in FEATURE_NAMES if f != 'P001.df_mean']

    # Reorder columns to match expected features
    features_df = features_df[expected_features]

    return features_df

def load_data_and_mappings():
    """Load data and build hierarchical mappings"""
    df = pd.read_csv("./Data/pmu_transmission_features.csv")

    # Build hierarchical mappings
    prim_to_sec = {}
    sec_to_ter = {}

    for _, row in df.iterrows():
        prim = row['primary']
        sec = row['secondary']
        ter = row['tertiary']

        if prim not in prim_to_sec:
            prim_to_sec[prim] = set()
        prim_to_sec[prim].add(sec)

        if sec not in sec_to_ter:
            sec_to_ter[sec] = set()
        sec_to_ter[sec].add(ter)

    # Convert sets to lists for JSON serialization
    prim_to_sec = {k: sorted(list(v)) for k, v in prim_to_sec.items()}
    sec_to_ter = {k: sorted(list(v)) for k, v in sec_to_ter.items()}

    # Get unique labels
    prim_labels = sorted(df["primary"].unique().tolist())
    sec_labels = sorted(df["secondary"].unique().tolist())
    ter_labels = sorted(df["tertiary"].unique().tolist())

    # Get first sample features (columns 5+: models were trained with 23 features starting from column 5)
    # This skips sigid, primary, secondary, tertiary, AND P001.df_mean (column 4)
    X = df.iloc[0, 5:].values.reshape(1, -1)  # Column 5 onwards are features (23 features)
    X_train = df.iloc[:, 5:].values  # For LIME if needed later

    # Store feature names and first sample values for path display
    feature_names = df.columns[5:].tolist()
    sample_values = dict(zip(feature_names, X[0]))

    return df, X, X_train, prim_labels, sec_labels, ter_labels, prim_to_sec, sec_to_ter, sample_values


@app.route('/summarize/<level>/<label>', methods=['GET'])
def summarize(level, label):
    if current_data is None:
        return jsonify({"error": "No data loaded. Call /load first"}), 400

    try:
        dict_ = current_data['explanations'][level]['lime']
        lime_vals = dict_.get(label, [])

        dict_ = current_data['explanations'][level]['shap']
        shap_vals = dict_.get(label, [])

        dict_ = current_data['explanations'][level]['paths']
        path_list = dict_.get(label, [])

        dict_ = current_data['explanations'][level]['mdi']
        mdi_list = dict_.get(label, [])

        # Convert numpy arrays to lists for JSON serialization
        if hasattr(shap_vals, 'tolist'):
            shap_vals = shap_vals.tolist()
        if hasattr(mdi_list, 'tolist'):
            mdi_list = mdi_list.tolist()

        all_merged = {
            "lime": lime_vals,
            "shap": shap_vals,
            "path": path_list,
            "mdi": mdi_list
        }

        # Format the data more clearly for the LLM
        prompt_parts = []

        # SHAP values - convert to feature importance
        if shap_vals:
            shap_list = list(shap_vals) if isinstance(shap_vals, (list, tuple)) else [shap_vals]
            if len(shap_list) == len(actual_feature_names):
                top_shap = sorted(zip(actual_feature_names, shap_list), key=lambda x: abs(x[1]), reverse=True)[:5]
                prompt_parts.append("Top 5 SHAP values (feature importance):\n" +
                                  "\n".join([f"  - {feat}: {val:.4f}" for feat, val in top_shap]))

        # LIME values
        if lime_vals:
            lime_str = "\n".join([f"  - {feat}: {val:.4f}" for feat, val in lime_vals[:5]])
            prompt_parts.append(f"Top 5 LIME values (local feature importance):\n{lime_str}")

        # MDI values
        if mdi_list and len(mdi_list) == len(actual_feature_names):
            top_mdi = sorted(zip(actual_feature_names, mdi_list), key=lambda x: x[1], reverse=True)[:5]
            prompt_parts.append("Top 5 MDI values (Random Forest feature importance):\n" +
                              "\n".join([f"  - {feat}: {val:.4f}" for feat, val in top_mdi]))

        # Decision paths
        if path_list:
            path_str = "\n".join([f"  - Depth {p.get('depth', 0)}: {p.get('feature_name', 'Unknown')} (support: {p.get('support_pct', 0):.1f}%)"
                                 for p in path_list[:5]])
            prompt_parts.append(f"Top decision tree paths:\n{path_str}")

        data_summary = "\n\n".join(prompt_parts)

        # Use the structured prompt template
        user_prompt = f"""You are explaining explainable-AI results to a reader who understands the application domain but has no knowledge of XAI.

Your task is to explain what the supplied metrics suggest about the model's prediction.

Follow these rules exactly:

1. Use only the four Markdown sections defined below.
2. Do not add an introduction, conclusion, or additional sections.
3. Do not explain how XAI methods work unless needed to interpret a result.
4. Do not describe an association as a proven cause.
5. Do not invent explanations that are not supported by the supplied data.
6. Use clear, nontechnical language.
7. When mentioning a technical metric, immediately explain what it means in practical terms.
8. Refer to the model as "the model," not "the AI."
9. Use consistent terminology throughout the response.
10. The entire response must contain between 140 and 190 words.

Required output format:

## Overall Interpretation

Write exactly three sentences.

Sentence 1 must state the overall result.

Sentence 2 must identify the strongest evidence supporting that result.

Sentence 3 must state how confident the reader should be in the interpretation, based only on the supplied evidence.

## Key Drivers

Write exactly three bullet points.

Each bullet must follow this format:

* **[Feature or factor]:** [One sentence explaining its influence in plain language.]

Order the bullets from most influential to least influential.

## What This Means

Write exactly one paragraph containing three sentences.

Explain what a non-XAI expert should take away from the results. Focus on the prediction and the factors associated with it. Do not repeat metric values unless they are necessary for understanding.

## Limitations

Write exactly two sentences.

State what cannot be concluded from the XAI results. Mention uncertainty, instability, disagreement between methods, or limited evidence when applicable.

Before producing the response, silently verify that:

* All four sections are present.
* There are exactly three driver bullets.
* No unsupported causal claims are made.
* The response is between 140 and 190 words.
* The terminology is consistent.

XAI results:

{data_summary}

Relevant prediction context:

Classification Level: {level}
Predicted Label: {label}
"""

        system_prompt = "You are an expert at interpreting explainability results for domain experts with no XAI background. Follow the template exactly."

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role":"system", "content":f"{system_prompt}"},
                      {"role": "user", "content": f"{user_prompt}"},],
        )

        summary = response.choices[0].message.content

        # Strip markdown code fence if present
        if summary.strip().startswith('```markdown'):
            summary = summary.strip()
            # Remove opening ```markdown
            summary = summary[len('```markdown'):].strip()
            # Remove closing ```
            if summary.endswith('```'):
                summary = summary[:-3].strip()
        elif summary.strip().startswith('```'):
            summary = summary.strip()
            # Remove opening ```
            summary = summary[3:].strip()
            # Remove closing ```
            if summary.endswith('```'):
                summary = summary[:-3].strip()

        # Debug: Print the raw summary output
        print("="*80)
        print(f"SUMMARY FOR {level}/{label}:")
        print("="*80)
        print(summary)
        print("="*80)

        return jsonify({
            'label': label,
            'level': level,
            'summary': summary
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in /summarize: {error_trace}")
        return jsonify({"error": str(e), "trace": error_trace}), 500
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process features"""
    global current_data, actual_feature_names, sample_feature_values, classifier_store, test_sample, train_data_stats, df_global

    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Only CSV files are allowed"}), 400

        # Save uploaded file (sanitize filename)
        filename = re.sub(r'[^\w\s.-]', '', file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"File uploaded: {filepath}")

        # Read the uploaded CSV (raw time-series data)
        df_raw = pd.read_csv(filepath)
        print(f"Raw data shape: {df_raw.shape}")
        print(f"Raw data columns: {list(df_raw.columns)}")

        # Compute features
        features_df = compute_features_from_raw(df_raw)
        print(f"Computed features shape: {features_df.shape}")
        print(f"Computed features columns: {list(features_df.columns)}")

        # Prepare X for inference
        X = features_df.values  # Shape: (1, 23) - excludes P001.df_mean
        actual_feature_names = list(features_df.columns)
        sample_feature_values = features_df.iloc[0].to_dict()
        test_sample = X

        # Load hierarchical mappings
        print("Loading hierarchical mappings...")
        df_train = pd.read_csv("./Data/pmu_transmission_features.csv")
        df_global = df_train

        prim_to_sec = {}
        sec_to_ter = {}
        for _, row in df_train.iterrows():
            prim = row['primary']
            sec = row['secondary']
            ter = row['tertiary']
            if prim not in prim_to_sec:
                prim_to_sec[prim] = set()
            prim_to_sec[prim].add(sec)
            if sec not in sec_to_ter:
                sec_to_ter[sec] = set()
            sec_to_ter[sec].add(ter)

        prim_to_sec = {k: sorted(list(v)) for k, v in prim_to_sec.items()}
        sec_to_ter = {k: sorted(list(v)) for k, v in sec_to_ter.items()}

        prim_labels = sorted(df_train["primary"].unique().tolist())
        sec_labels = sorted(df_train["secondary"].unique().tolist())
        ter_labels = sorted(df_train["tertiary"].unique().tolist())

        # Get training features for LIME (columns 5+ for 23 features)
        X_train = df_train.iloc[:, 5:].values

        print(f"X shape: {X.shape}, X_train shape: {X_train.shape}")

        # Verify shapes match
        if X.shape[1] != X_train.shape[1]:
            return jsonify({
                "error": f"Feature mismatch: uploaded file has {X.shape[1]} features but training data has {X_train.shape[1]} features"
            }), 400

        # Setup config and store
        print("Creating config and store...")
        cfg = PipelineConfig(
            primary_group="State",
            model=ModelConfig(model_type="rf", weighting=True),
            classifiers_dir="Classifiers"
        )
        store = ClassifierStore(cfg.classifiers_dir, "default")
        classifier_store = store
        print("Config and store created successfully")

        # Run inference for all three levels
        print("Running primary inference...")
        outputs_prim, shap_prim, lime_prim, paths_prim, mdi_prim = successive_binary_classifier(
            X, X_train, prim_labels, store=store, level="primary", compute_explanations=True
        )
        print(f"Primary predictions: {len(outputs_prim)}")

        print("Running secondary inference...")
        outputs_sec, shap_sec, lime_sec, paths_sec, mdi_sec = successive_binary_classifier(
            X, X_train, sec_labels, store=store, level="secondary", compute_explanations=True
        )
        print(f"Secondary predictions: {len(outputs_sec)}")

        print("Running tertiary inference...")
        outputs_ter, shap_ter, lime_ter, paths_ter, mdi_ter = successive_binary_classifier(
            X, X_train, ter_labels, store=store, level="tertiary", compute_explanations=True
        )
        print(f"Tertiary predictions: {len(outputs_ter)}")

        # Format predictions
        def format_predictions(outputs_dict):
            predictions = []
            for label, confidence in outputs_dict.items():
                predictions.append({
                    'label': label,
                    'confidence': float(confidence)
                })
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            return predictions

        # Store globally
        current_data = {
            'predictions': {
                'primary': format_predictions(outputs_prim),
                'secondary': format_predictions(outputs_sec),
                'tertiary': format_predictions(outputs_ter)
            },
            'explanations': {
                'primary': {'shap': shap_prim, 'lime': lime_prim, 'paths': paths_prim, 'mdi': mdi_prim},
                'secondary': {'shap': shap_sec, 'lime': lime_sec, 'paths': paths_sec, 'mdi': mdi_sec},
                'tertiary': {'shap': shap_ter, 'lime': lime_ter, 'paths': paths_ter, 'mdi': mdi_ter}
            },
            'mappings': {
                'prim_to_sec': prim_to_sec,
                'sec_to_ter': sec_to_ter
            }
        }

        print("Upload and inference complete!")
        return jsonify({"status": "success", "filename": filename})

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in /upload: {error_trace}")
        return jsonify({"status": "error", "message": str(e), "trace": error_trace}), 500

@app.route('/load', methods=['POST'])
def load_sample():
    """Run inference on first sample and store results"""
    global current_data, actual_feature_names, sample_feature_values, classifier_store, test_sample, train_data_stats, df_global

    try:
        print("Loading data and mappings...")
        # Load data
        df, X, X_train, prim_labels, sec_labels, ter_labels, prim_to_sec, sec_to_ter, sample_values = load_data_and_mappings()

        # Store actual feature names and sample values (from column 5 onwards)
        actual_feature_names = df.columns[5:].tolist()
        sample_feature_values = sample_values
        test_sample = X  # Store test sample

        # Store training data statistics for counterfactuals (use nanmean to handle NaN)
        df_global = df

        # Replace any remaining NaN with safe defaults

        print(f"Data loaded. Shape: {df.shape}, X shape: {X.shape}")
        print(f"Feature names: {actual_feature_names[:5]}... (total: {len(actual_feature_names)})")
        print(f"Primary labels: {prim_labels}")
        print(f"Secondary labels count: {len(sec_labels)}")
        print(f"Tertiary labels count: {len(ter_labels)}")

        # Setup config and store
        print("Creating config and store...")
        cfg = PipelineConfig(
            primary_group="State",
            model=ModelConfig(model_type="rf", weighting=True),
            classifiers_dir="Classifiers"
        )
        store = ClassifierStore(cfg.classifiers_dir, "default")
        classifier_store = store  # Store globally for counterfactual generation
        print("Config and store created successfully")

        # Run inference for all three levels (WITH explanations for SHAP, LIME, MDI)
        print("Running primary inference...")
        outputs_prim, shap_prim, lime_prim, paths_prim, mdi_prim = successive_binary_classifier(
            X, X_train, prim_labels, store=store, level="primary", compute_explanations=True
        )
        print(f"Primary predictions: {len(outputs_prim)}")

        print("Running secondary inference...")
        outputs_sec, shap_sec, lime_sec, paths_sec, mdi_sec = successive_binary_classifier(
            X, X_train, sec_labels, store=store, level="secondary", compute_explanations=True
        )
        print(f"Secondary predictions: {len(outputs_sec)}")

        print("Running tertiary inference...")
        outputs_ter, shap_ter, lime_ter, paths_ter, mdi_ter = successive_binary_classifier(
            X, X_train, ter_labels, store=store, level="tertiary", compute_explanations=True
        )
        print(f"Tertiary predictions: {len(outputs_ter)}")

        # Format predictions
        def format_predictions(outputs_dict):
            predictions = []
            for label, confidence in outputs_dict.items():
                predictions.append({
                    'label': label,
                    'confidence': float(confidence)
                })
            # Sort by confidence descending
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            return predictions

        # Store globally
        current_data = {
            'predictions': {
                'primary': format_predictions(outputs_prim),
                'secondary': format_predictions(outputs_sec),
                'tertiary': format_predictions(outputs_ter)
            },
            'explanations': {
                'primary': {'shap': shap_prim, 'lime': lime_prim, 'paths': paths_prim, 'mdi': mdi_prim},
                'secondary': {'shap': shap_sec, 'lime': lime_sec, 'paths': paths_sec, 'mdi': mdi_sec},
                'tertiary': {'shap': shap_ter, 'lime': lime_ter, 'paths': paths_ter, 'mdi': mdi_ter}
            },
            'mappings': {
                'prim_to_sec': prim_to_sec,
                'sec_to_ter': sec_to_ter
            }
        }

        print("Load complete!")
        print(f"Primary SHAP labels: {list(shap_prim.keys())}")
        print(f"Secondary SHAP labels count: {len(shap_sec)}")
        print(f"Tertiary SHAP labels count: {len(shap_ter)}")
        return jsonify({"status": "success"})

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in /load: {error_trace}")
        return jsonify({"status": "error", "message": str(e), "trace": error_trace}), 500

@app.route('/predictions', methods=['GET'])
def get_predictions():
    """Return all primary predictions"""
    if current_data is None:
        return jsonify({"error": "No data loaded. Call /load first"}), 400

    return jsonify({
        'primary': current_data['predictions']['primary']
    })

@app.route('/predictions/<primary_label>', methods=['GET'])
def get_secondary_for_primary(primary_label):
    """Return only secondaries that belong to this primary"""
    if current_data is None:
        return jsonify({"error": "No data loaded. Call /load first"}), 400

    prim_to_sec = current_data['mappings']['prim_to_sec']
    valid_secondaries = prim_to_sec.get(primary_label, [])

    # Filter secondary predictions
    filtered = [p for p in current_data['predictions']['secondary']
                if p['label'] in valid_secondaries]

    # Sort by confidence descending
    filtered.sort(key=lambda x: x['confidence'], reverse=True)

    return jsonify({'secondary': filtered})

@app.route('/predictions/<primary_label>/<secondary_label>', methods=['GET'])
def get_tertiary_for_secondary(primary_label, secondary_label):
    """Return only tertiaries that belong to this secondary"""
    if current_data is None:
        return jsonify({"error": "No data loaded. Call /load first"}), 400

    sec_to_ter = current_data['mappings']['sec_to_ter']
    valid_tertiaries = sec_to_ter.get(secondary_label, [])

    # Filter tertiary predictions (exclude nonspecific labels)
    filtered = [p for p in current_data['predictions']['tertiary']
                if p['label'] in valid_tertiaries and 'nonspecific' not in p['label'].lower()]

    # Sort by confidence descending
    filtered.sort(key=lambda x: x['confidence'], reverse=True)

    return jsonify({'tertiary': filtered})

@app.route('/explanations/<level>/<label>', methods=['GET'])
def get_explanations(level, label):
    """Return SHAP values for a specific label"""
    if current_data is None:
        return jsonify({"error": "No data loaded. Call /load first"}), 400

    try:
        # Get SHAP values for this label
        shap_dict = current_data['explanations'][level]['shap']

        print(f"Looking for label '{label}' in {level}")
        print(f"Available labels: {list(shap_dict.keys())}")

        if label not in shap_dict:
            return jsonify({
                "error": f"Label '{label}' not found in {level} explanations",
                "available_labels": list(shap_dict.keys())
            }), 404

        shap_values = shap_dict[label]

        # Convert to list if numpy array
        if hasattr(shap_values, 'tolist'):
            shap_values = shap_values.tolist()

        print(f"SHAP values length: {len(shap_values)}")
        print(f"Feature names length: {len(actual_feature_names)}")
        print(f"First 5 SHAP values: {shap_values[:5]}")

        # Handle binary classification SHAP values
        # If we have 2x the features, they're interleaved: [class0_feat0, class1_feat0, class0_feat1, class1_feat1, ...]
        if len(shap_values) == 2 * len(actual_feature_names):
            print("Binary SHAP detected - taking positive class (every other value starting at index 1)")
            # Take every other value starting from index 1 (class 1 values)
            shap_values = shap_values[1::2]
            print(f"After taking positive class, length: {len(shap_values)}")
            print(f"After taking positive class, first 5 values: {shap_values[:5]}")

        # SHAP values should now match feature count
        if len(shap_values) != len(actual_feature_names):
            print(f"ERROR: SHAP length still mismatched. Expected {len(actual_feature_names)}, got {len(shap_values)}")
            return jsonify({"error": f"SHAP values length ({len(shap_values)}) doesn't match features ({len(actual_feature_names)})"}), 500

        # Get LIME values for this label
        lime_dict = current_data['explanations'][level]['lime']
        lime_list = lime_dict.get(label, [])

        print(f"DEBUG LIME for {label}: lime_list length={len(lime_list)}, first 3={lime_list[:3] if lime_list else 'Empty'}")

        # LIME returns list of tuples: [(feature_name, value), ...]
        lime_values = {}
        for item in lime_list:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                lime_values[item[0]] = float(item[1])

        print(f"DEBUG LIME parsed: {len(lime_values)} features, sample values={list(lime_values.items())[:3]}")

        # Get MDI values for this label
        mdi_dict = current_data['explanations'][level]['mdi']
        mdi_array = mdi_dict.get(label, [])

        # Convert to list if numpy array
        if hasattr(mdi_array, 'tolist'):
            mdi_array = mdi_array.tolist()

        print(f"MDI values length: {len(mdi_array)}")
        print(f"First 5 MDI values: {mdi_array[:5] if len(mdi_array) > 0 else 'Empty'}")

        # Get MDI (feature importances) for this label
        mdi_dict = current_data['explanations'][level]['mdi']
        mdi_array = mdi_dict.get(label, [])

        # Convert to list if numpy array
        if hasattr(mdi_array, 'tolist'):
            mdi_array = mdi_array.tolist()

        # Get decision path for this label
        paths_dict = current_data['explanations'][level]['paths']
        path_list = paths_dict.get(label, [])

        print(f"Path data for {label}:")
        print(f"Number of steps: {len(path_list)}")
        if path_list:
            print(f"First step keys: {path_list[0].keys() if isinstance(path_list[0], dict) else 'Not a dict'}")
            print(f"First step data: {path_list[0]}")

        # Format path steps for display
        path_steps = []
        for step in path_list:
            if isinstance(step, dict):
                feature_name = step.get('feature_name', 'Unknown')
                threshold = step.get('avg_value', 0)  # avg_value IS the threshold
                actual_value = sample_feature_values.get(feature_name, 0)  # Get actual input value
                direction = 'left' if actual_value <= threshold else 'right'

                path_steps.append({
                    'feature_name': feature_name,
                    'threshold': threshold,
                    'actual_value': actual_value,
                    'direction': direction,
                    'gain': step.get('avg_gain', 0),
                    'support_pct': step.get('support_pct', 0),
                    'depth': step.get('depth', 0)
                })

        # Create feature-value pairs with SHAP, LIME, and MDI
        features = []
        for i, (feature_name, shap_val) in enumerate(zip(actual_feature_names, shap_values)):
            mdi_val = mdi_array[i] if i < len(mdi_array) else 0.0
            features.append({
                'name': feature_name,
                'shap': float(shap_val),
                'lime': lime_values.get(feature_name, 0.0),
                'mdi': float(mdi_val)
            })

        return jsonify({
            'label': label,
            'features': features,
            'path': path_steps
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in /explanations: {error_trace}")
        return jsonify({"error": str(e), "trace": error_trace}), 500

@app.route('/counterfactual/<level>/<label>', methods=['GET'])
def get_counterfactual(level, label):
    """Generate a counterfactual explanation for a given label"""
    if current_data is None or test_sample is None or classifier_store is None:
        return jsonify({"error": "No data loaded. Call /load first"}), 400

    try:
        print(f"Generating counterfactuals for {level}/{label}")

        # Use actual feature names from loaded data (not hardcoded FEATURE_NAMES)
        feature_cols = actual_feature_names

        # Prepare the dataframe for DiCE - keep only target level and features used for training
        df_local = df_global.copy()

        # Keep only the level column and feature columns (columns 5+)
        # First, get the target labels
        target_labels = df_local[level].values

        # Get only the feature columns (5+ for 23 features)
        features_df = df_local.iloc[:, 5:].copy()

        # Create new dataframe with features + binary target
        df_local = features_df.copy()
        df_local[level] = [1 if entry == label else 0 for entry in target_labels]

        print(f"DiCE dataset shape: {df_local.shape}")
        print(f"DiCE columns: {list(df_local.columns)}")
        print(f"Target distribution: {df_local[level].value_counts().to_dict()}")

        # Load the trained model for this level/label
        model = classifier_store.load(level, label)

        # Get actual model if wrapped in search CV
        actual_model = model
        if hasattr(model, 'best_estimator_'):
            actual_model = model.best_estimator_

        # Create DiCE data object
        d = dice_ml.Data(
            dataframe=df_local,
            continuous_features=feature_cols,
            outcome_name=level
        )

        # Create DiCE model object
        m = dice_ml.Model(model=actual_model, backend='sklearn')

        # Create DiCE explainer
        exp = dice_ml.Dice(d, m, method='random')

        # Prepare query instance (test sample as dataframe)
        query_df = pd.DataFrame(test_sample, columns=feature_cols)

        print(f"Query instance shape: {query_df.shape}")
        print(f"Query features: {list(query_df.columns)}")

        # Get current prediction for reference
        current_pred = model.predict(test_sample)[0]
        current_prob = model.predict_proba(test_sample)[0][1]  # Probability of class 1
        print(f"Current prediction: {current_pred}, Probability: {current_prob}")

        # Generate a single counterfactual - use "opposite" to flip the prediction
        dice_exp = exp.generate_counterfactuals(
            query_df,
            total_CFs=1,
            desired_class="opposite"
        )

        # Extract counterfactual example
        cf_examples = dice_exp.cf_examples_list[0].final_cfs_df

        print(cf_examples)

        if cf_examples is None or len(cf_examples) == 0:
            return jsonify({
                'label': label,
                'changes': [],
                'message': 'No counterfactuals found'
            })

        # Get the single counterfactual and calculate changes per feature
        original_values = query_df.iloc[0].to_dict()
        cf_row = cf_examples.iloc[0]

        changes = []
        for feature in feature_cols:
            original_val = original_values[feature]
            cf_val = cf_row[feature]

            # Calculate the raw change
            change_val = cf_val - original_val

            if abs(change_val) > 1e-6:  # Feature changed
                changes.append({
                    'feature': feature,
                    'original': float(original_val),
                    'counterfactual': float(cf_val),
                    'change': float(change_val)
                })

        # Sort by absolute change magnitude (most impactful first)
        changes.sort(key=lambda x: abs(x['change']), reverse=True)

        print(f"Found {len(changes)} feature changes")
        return jsonify({
            'label': label,
            'changes': changes,
            'current_prediction': int(current_pred),
            'current_probability': float(current_prob),
            'desired_prediction': 1 - int(current_pred)
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in /counterfactual: {error_trace}")
        return jsonify({"error": str(e), "trace": error_trace}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5001)  # Set debug=True only in development
