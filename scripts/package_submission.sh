#!/usr/bin/env bash
# Create submission zip for Google Drive (excludes secrets and virtualenvs).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_DIR="$(dirname "$ROOT_DIR")"
PROJECT_NAME="$(basename "$ROOT_DIR")"
OUTPUT="$PARENT_DIR/${PROJECT_NAME}-submission.zip"

cd "$PARENT_DIR"
zip -r "$OUTPUT" "$PROJECT_NAME" \
  -x "${PROJECT_NAME}/venv/*" \
  -x "${PROJECT_NAME}/.venv-sagemaker/*" \
  -x "${PROJECT_NAME}/.env" \
  -x "${PROJECT_NAME}/.git/*" \
  -x "${PROJECT_NAME}/deployment/artifacts/*" \
  -x "${PROJECT_NAME}/__pycache__/*" \
  -x "${PROJECT_NAME}/*/__pycache__/*" \
  -x "${PROJECT_NAME}/*/*/__pycache__/*" \
  -x "${PROJECT_NAME}/.DS_Store"

echo "Created: $OUTPUT"
