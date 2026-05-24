# NIDS

Network Intrusion Detection System using the NSL-KDD benchmark dataset.

This repository trains and evaluates machine learning models to classify network traffic as normal or malicious, supporting both binary and multiclass threat detection.

## Repository structure

- `nids/`
  - `columns.py`       — feature names, categorical columns, and label mappings
  - `preprocess.py`    — data loading, encoding, scaling, and preprocessing pipeline
  - `train.py`         — trains Random Forest, Gradient Boosting, and MLP models
  - `evaluate.py`      — evaluates models and saves plots
  - `simulate.py`      — simulates live inference from the test dataset
  - `requirements.txt` — Python dependencies for the project
  - `README.md`        — package-specific docs
- `venv/`              — local Python virtual environment (ignored)
- `models/`            — saved model artifacts (ignored)
- `outputs/`           — generated evaluation plots (ignored)
- `nids/data/`         — dataset files (ignored)

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Download the NSL-KDD dataset:

- Go to: https://www.unb.ca/cic/datasets/nsl.html
- Download `KDDTrain+.txt` and `KDDTest+.txt`
- Place them in `nids/data/`

## Usage

Train all models:

```powershell
python nids/train.py --mode multiclass
```

Evaluate and generate outputs:

```powershell
python nids/evaluate.py --mode multiclass
```

Simulate real-time inference:

```powershell
python nids/simulate.py --n 50 --delay 0.1 --mode multiclass
```

## Notes

- Use Python 3.9+ for compatibility with the required packages.
- The root repository ignores generated artifacts and the local `venv`.
