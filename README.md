# Grid Event Signature Library's Signature Matching Tool (GESL SMT with XAI)

Hierarchical classification system for power grid event signatures with explainable AI (XAI) methods.

GESL is a multi-level classification framework that builds and deploys machine learning models for power grid event analysis. It works with PMU (Phasor Measurement Unit) data and provides explainable predictions through multiple XAI methods including SHAP, LIME, and feature importance visualization.

See the [QUICKSTART.md](QUICKSTART.md) for a quick guide to getting started.

## Project Structure

```
GESL/
├── src/
│   └── lcn/              # Core library for hierarchical classification
│       ├── config.py     # Configuration classes
│       ├── models.py     # Model building (RF, XGBoost)
│       ├── training.py   # Training pipeline
│       ├── inference.py  # Inference with XAI explanations
│       ├── augmentation.py  # Data resampling (SMOTE, ADASYN, GMM)
│       └── ...
├── server.py            # Flask server with file upload support
├── demo_server.py       # Demo server with pre-loaded data
├── viewer.html          # Interactive web viewer with XAI methods
├── Data/                # Training data
├── Classifiers/         # Trained model files
└── Uploads/             # User-uploaded files
```

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

#### Full Server (File Upload)
```bash
python server.py
```
Then open `viewer.html` in your browser and upload your CSV file.

#### Demo Server (Pre-loaded Data)
```bash
python demo_server.py
```
Then open `viewer.html` in your browser to view pre-loaded event classifications.

#### No Server

Look at the `no_app.py` for standalone usage examples.

## Data Format

Input CSV files should contain:
- Columns 0-4: Metadata (Sig_ID, primary, secondary, tertiary, etc.)
- Columns 5+: Feature columns (23 statistical features from PMU signals)

Expected features (excluding P001.df_mean):
```
P001.df_std, P001.df_skew, P001.df_kurtosis,
P001.f_mean, P001.f_std, P001.f_skew, P001.f_kurtosis,
P001.ip_a_mean, P001.ip_a_std, P001.ip_a_skew, P001.ip_a_kurtosis,
P001.ip_m_mean, P001.ip_m_std, P001.ip_m_skew, P001.ip_m_kurtosis,
P001.vp_a_mean, P001.vp_a_std, P001.vp_a_skew, P001.vp_a_kurtosis,
P001.vp_m_mean, P001.vp_m_std, P001.vp_m_skew, P001.vp_m_kurtosis
```

## Documentation

Full documentation is available in the following files:
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [TODO.md](TODO.md) - Development roadmap and known issues
- `.env.example` - Environment configuration template

For additional help, please reference the Authors section below or open an issue on GitHub.

## Community

GESL is an open source project. Questions, discussion, and contributions are welcome. Contributions can be anything from new features to bugfixes, documentation, or improvements to the classification models.

Resources:
- **Contact**: See Authors section for team contact information
- **Documentation**: See files listed in the Documentation section above

## Contributing

Contributing to GESL is relatively easy. Just send us a pull request.

Your PR must:
- Make `main` (or `develop`) the destination branch
- Pass all unit tests and validation checks
- Follow the existing code style and conventions
- Include appropriate documentation for new features
- Sign off all commits with `git commit --signoff`. Signoff says that you agree to the Developer Certificate of Origin.

For more details, see [CONTRIBUTING.md](community-health/CONTRIBUTING.md).

## Code of Conduct

Please note that GESL has a Code of Conduct. By participating in the GESL community, you agree to abide by its rules.

See [CODE_OF_CONDUCT.md](community-health/CODE_OF_CONDUCT.md) for details.

## Authors

| Contact Name | Email Address | Organization | Status |
|---|---|---|---|
| Jhi-Young Joo |  | LLNL | Left LLNL |
| Chih-Che (Ryan) Sun |  | LLNL | Left LLNL |
| Christabella Annalicia | Annalicia1@llnl.gov | LLNL | Current |
| Diego Gandara-Mendez | gandaramende1@llnl.gov | UTEP, LLNL | Current |
| Jean Tonday | tondayrodrig1@llnl.gov | FIU, LLNL | Current |

## License

GESL is distributed under the terms of BSD-3 license.

All new contributions must be made under the same license.

See LICENSE for details.

SPDX-License-Identifier: BSD-3-Clause

## Release

LLNL-CODE-2025335
