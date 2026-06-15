#!/usr/bin/env bash
# Retrain ML models with library versions compatible with SageMaker sklearn 1.4-2 container.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="$ROOT_DIR/.venv-sagemaker"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.10 is required. Install with: brew install python@3.10"
  exit 1
fi

echo "Creating SageMaker-compatible training env with $PYTHON_BIN..."
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install -q --upgrade pip
pip install -q "numpy>=1.26,<2" "scikit-learn>=1.4,<1.5" "pandas" "joblib" "ucimlrepo"

echo "Training regression model..."
python training/train_regression.py

echo "Training classification model..."
python training/train_classification.py

python - <<'PY'
import joblib
import sklearn
import numpy as np
print("Compatible artifacts ready:")
print("  sklearn", sklearn.__version__)
print("  numpy", np.__version__)
print("  regression model:", "models/housing_regression_model.joblib")
print("  classification model:", "models/bank_classification_model.joblib")
PY

deactivate
echo "Done. Redeploy with deployment/deploy_regression.py"
