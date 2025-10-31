# DOE Grid Event Signature Library Signature Matching Tool

This repository contains a **Signature Matching Tool (SMT)** designed to process time-series PMU data, train a hierarchy of binary classifiers, and visualize predictions. It consists of two main components:

- **Training Pipeline** (`smt_lcn.py`) – Trains a hierarchy of binary classifiers for group, class, and subclass labels.
- **Prediction & Visualization** (`get_prediction.py`) – Loads trained models and uses them to classify new PMU signatures, outputting nested pie charts and feature importance visualizations.

---

## Features

- Hierarchical Classification (Primary → Secondary → Tertiary Labels)
- Per-label binary classifiers using Random Forests
- Normalization and statistical feature extraction for PMU signals
- Likelihood estimation for each label (probability-based)
- Visualizations: nested pie charts for prediction probabilities, bar charts for feature importances
- Sklearn-compatible pipelines with persistent pickle models

---

## Project Structure

```
.
├── smt_lcn.py                       # Training + Evaluation pipeline
├── get_prediction.py               # Prediction + Visualization pipeline
├── Inputs/
│   ├── pmu_transmission_features.csv     # Training dataset
│   ├── sigId-980.csv                     # Input signal for prediction
│   ├── pri_to_sec_dict_transmission.json
│   └── sec_to_tert_dict_transmission.json
├── Classifiers/                   # Pickled binary classifiers saved here
└── Outputs/
    ├── lcn_rf_test_results.csv         # Prediction outputs
    └── lcn_rf_metrics_results.csv      # Classification metrics
```

---

## How to Run the Tool

### 1. Training Binary Classifiers

To train new binary classifiers, run:

```bash
python smt_lcn.py
```

- **Input**:  
  - `Inputs/pmu_transmission_features.csv`: Pre-extracted PMU statistical features (includes `primary`, `secondary`, `tertiary` labels).
  - `Inputs/sec_to_tert_dict_transmission.json`: Mapping from secondary to tertiary labels.
- **Output**:  
  - Pickled models saved to `Classifiers/`.
  - Evaluation reports in `Outputs/`.

### 2. Run Predictions on a New Signature

To classify a new PMU signature and visualize results, run:

```bash
python get_prediction.py
```

- **Input**:  
  - `Inputs/sigId-980.csv`: Raw PMU time-series waveform data. Assumed format:
    ```
    Time, P001.V1, P001.V2, P001.V3, P001.I1, P001.I2, P001.I3, ...
    ```
  - `pri_to_sec_dict_transmission.json`: Group → Class mapping
  - `sec_to_tert_dict_transmission.json`: Class → Subclass mapping

- **Output**:  
  - Visualizations (pop-up matplotlib figures):
    - Pie charts showing predicted class/subclass probabilities
    - (Optional) Bar charts showing feature importances

---

## Input Requirements

### For Training (`pmu_transmission_features.csv`)

| Column Name | Description |
|-------------|-------------|
| `sigid`     | Unique ID for the signature |
| `primary`   | Group-level label |
| `secondary` | Class-level label |
| `tertiary`  | Subclass-level label |
| Feature Columns | PMU-derived statistical features (mean, std, skew, kurtosis) |

### For Prediction (`sigId-980.csv`)

- CSV file with PMU time-series waveform data
- **Assumes exactly 6 features per PMU** (e.g., `P001.df`, `P001.f`, `P001.ip_a`, `P001.ip_m`, `P001.vp_a`, `P001.vp_m`)
- First column should be `Time`
- Automatically computes statistical features per PMU instance

---

## Model Details

- **Classifier**: `RandomForestClassifier` from scikit-learn
- **Approach**: Local Classifier per Node (LCN)
- **Hierarchy**: Primary (Group) → Secondary (Class) → Tertiary (Subclass)
- **Evaluation Metrics**:
  - Accuracy
  - Hierarchical Precision, Recall, and F1 Score

---

## Example Output

- Nested pie chart showing probability distribution across class and subclass labels.
- Feature importance bar chart (optional toggle).
- CSV files with evaluation metrics and prediction results.

---

## Dependencies

Install required packages with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Branch Management Guidelines

### Main Branch (`main`)
- Stable, production-ready code only.
- No direct commits; merge via pull requests after review.
- Used for releases and deployments.

### Development Branch (`development`)
- Integrates ongoing work and new features.
- No direct commits; merge feature branches via pull requests.
- Regularly sync with `main` and prepare for releases.

### Branch Creation Rules

#### Naming Convention

- **Format:** `dev/organization/branch-name`
  - `dev`: Indicates a development branch.
  - `organization`: Your team, department, or project identifier.
  - `branch-name`: A short, descriptive name for the feature, fix, or task.

##### Example
- `dev/LLNL/reform-input`

#### Workflow Guidelines

- Always create feature branches from the latest `development` branch.
- Use clear, concise branch names that describe the branch’s purpose.
- Delete feature branches after merging to keep the repository clean.
## Contact

| Contact Name            | Email Address          | Organization   |
|-------------------------|------------------------|----------------|
| Jhi-Young Joo           | joo3@llnl.gov          | LLNL           |
| Chih-Che (Ryan) Sun     | sun31@llnl.gov         | LLNL           |
| Christabella Annalicia  | Annalicia1@llnl.gov    | LLNL           |
| Diego Gandara-Mendez    | gandaramende1@llnl.gov | UTEP, LLNL     |

## Release

The code of this site is released under the BSD 3 License. For more details, see the [LICENSE](LICENSE) File.

LLNL-CODE-2010456



