#!/usr/bin/env bash
# End-to-end ML training: baseline (tfidf+logreg) -> distilbert -> eval
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "=== [1/3] baseline: tfidf + logreg ==="
python ML/training/train_baseline.py

echo "=== [2/3] distilbert text detector ==="
python ML/training/train_text_model.py

echo "=== [3/3] evaluate on held-out test split ==="
python ML/training/evaluate.py ML/models/text_detector ML/data/processed/baseline_test.parquet

echo "=== DONE. Artifacts: ML/models/baseline_model.pkl, ML/models/text_detector/ ==="