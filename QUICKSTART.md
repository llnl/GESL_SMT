# Quick Start Guide

This folder contains a clean, ready-to-push implementation of the GESL hierarchical classification system.

## Directory Structure

```
clean_implementation/
├── src/lcn/              # Core classification library (11 modules)
├── server.py             # Flask server with file upload
├── demo_server.py        # Demo server with pre-loaded data
├── viewer_v2.html           # Interactive web UI with 5 XAI methods
├── Data/                 # Training data
├── Classifiers/          # Pre-trained models
├── Uploads/              # User uploads directory
├── requirements.txt      # Python dependencies
├── README.md             # Full documentation
└── .gitignore            # Git ignore rules
```

## Installation

0. **Install Git LFS and download models**
   ```bash
   git lfs install
   git lfs pull
   ```

1. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment**
   ```bash
   # Windows:
   venv\Scripts\activate
   
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Option 1: Demo Mode (Quickest)
```bash
python demo_server.py
```
- Regenerates `xai.json` with predictions from the first training sample
- Open `viewer_v2.html` in your browser
- Click "LOAD DEMO" button to load the pre-computed results
- Explore the hierarchical classifications and XAI explanations

### Option 2: Upload Mode (Your Own Data)
```bash
python server.py
```
- Server starts on `http://localhost:5001`
- Open `viewer_v2.html` in your browser
- Upload your CSV file (must have 23 features)
- View predictions and explanations

## Features in the Viewer

### Classification Hierarchy
- **Primary** → **Secondary** → **Tertiary** levels
- Interactive selection with confidence scores
- Animated transitions

### Explainable AI (XAI) Methods
Click on any prediction to see:

1. **SHAP** - Game theory-based feature importance
2. **LIME** - Local model approximation
3. **MDI** - Random Forest feature importance
4. **Decision Path** - Tree traversal with information gain
5. **Counterfactual** - Minimum changes to flip prediction

Each method has an info button (?) that opens an interactive slideshow explaining how it works!

## What Changed from Original Code

### File Organization
- **Before**: Scattered across `GESL_DIEGO/lcn/` and `GESL_SMT-development/`
- **After**: Consolidated in `src/lcn/`

### Import Paths
- **Before**: `from GESL_DIEGO.lcn.config import ...`
- **After**: `from src.lcn.config import ...`

## Next Steps for GitHub

1. **Initialize Git** (if not already done)
   ```bash
   cd clean_implementation
   git init
   ```

2. **Add files**
   ```bash
   git add .
   ```

3. **Commit**
   ```bash
   git commit -m "Initial commit: Clean GESL implementation"
   ```

4. **Push to GitHub**
   ```bash
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

## Notes

- The `Classifiers/` folder contains pre-trained models (large .pickle files)
- Consider using Git LFS for large model files if pushing to GitHub
- The `Data/` folder has training data - ensure it's appropriate to share
- All imports are now relative to `src.lcn` module

## Support

See `README.md` for full documentation including:
- API endpoints
- Training new models
- Data format requirements
- Architecture details
