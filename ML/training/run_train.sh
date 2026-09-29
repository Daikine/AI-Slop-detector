#!/usr/bin/env bash
# End-to-end local training: TF-IDF + LogisticRegression baseline
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "=== Train and evaluate TF-IDF + LogisticRegression ==="
python ML/training/train_baseline.py

echo "=== DONE. Artifacts: ML/models/baseline_model.pkl and baseline_vectorizer.pkl ==="