# Signature Matching Tool (SMT)

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



# LICENSE

Copyright (c) 2025, Lawrence Livermore National Security, LLC.  All rights reserved.  LLNL-CODE-2010456

May be exempt from public release under the Freedom of Information Act (5 U.S.C 552), exemption and category: Exemption 3 (ECI), [also include any additional exemptions and category names] Department of Energy review required before public release. Date: [Date of guidance determination] Name/org: [Name/org of employee determining status of doc] Guidance (if applicable): [“NA” if not applicable]

EXPORT CONTROLLED INFORMATION - Contains technical data whose export is restricted by statute. Violations may result in administrative, civil, or criminal penalties. Limit dissemination to authorized U.S. Department of Energy and major U.S. DOE contractor personnel. The cognizant program manager must approve other dissemination, in accordance with applicable regulations. This notice shall not be separated from the attached document.

## NOTICE
This work was produced at the Lawrence Livermore National Laboratory (LLNL) under contract no. DE-AC52-07NA27344 (Contract 44) between the U.S. Department of Energy (DOE) and Lawrence Livermore National Security, LLC (LLNS) for the operation of LLNL.  Copyright is reserved to Lawrence Livermore National Security, LLC for purposes of controlled dissemination, commercialization through formal licensing, or other disposition under terms of Contract 44; DOE policies, regulations and orders; and U.S. statutes. The rights of the Federal Government are reserved under Contract 44. 

## DISCLAIMER 
This work was prepared as an account of work sponsored by an agency of the United States Government. NEITHER THE UNITED STATES GOVERNMENT NOR LAWRENCE LIVERMORE NATIONAL SECURITY, LLC NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY-OWNED RIGHTS.  THIS SOFTWARE IS PROVIDED BY LAWRENCE LIVERMORE NATIONAL SECURITY, LLC "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

Reference herein to any specific commercial products, process, or service by trade name, trademark, manufacturer or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or Lawrence Livermore National Security, LLC. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or Lawrence Livermore National Security, LLC, and shall not be used for advertising or product endorsement purposes.

## NOTIFICATION OF COMMERCIAL USE

Commercialization of this product is prohibited without notifying the Department of Energy (DOE) or Lawrence Livermore National Security.



