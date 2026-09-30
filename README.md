# Fashion ANN Pipeline

End-to-end ML versioning assignment: a fully-connected ANN classifying Fashion-MNIST
images, built with reproducible Git + DVC workflows.

## Problem
Classify Fashion-MNIST images (28x28 grayscale, 10 classes) using a Dense ANN.
Target: ≥85% test accuracy.

## Structure
- `src/prepare.py` — loads Fashion-MNIST, saves raw arrays to `data/raw/`
- `src/preprocess.py` — normalizes and splits data, saves to `data/processed/`
- `src/train.py` — trains the ANN, saves model to `models/model.h5`
- `src/evaluate.py` — evaluates on test set, writes `metrics.json`

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pipeline
Data and models are versioned with DVC (Google Drive remote).
Run the full pipeline with:
```bash
dvc repro
```

## Reproducing
```bash
git clone <repo-url>
dvc pull
dvc repro
```


word