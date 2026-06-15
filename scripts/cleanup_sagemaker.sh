#!/usr/bin/env bash
# Remove failed SageMaker endpoint resources before redeploying.
set -euo pipefail

export AWS_REGION="${AWS_REGION:-us-east-1}"

ENDPOINTS=(
  "housing-regression-endpoint"
  "bank-classification-endpoint"
)

for name in "${ENDPOINTS[@]}"; do
  echo "Cleaning $name ..."
  aws sagemaker delete-endpoint --endpoint-name "$name" 2>/dev/null || true
  aws sagemaker delete-endpoint-config --endpoint-config-name "$name" 2>/dev/null || true
done

echo "Listing recent sklearn models to delete manually if needed:"
aws sagemaker list-models --sort-by CreationTime --sort-order Descending --max-results 5 --query "Models[].ModelName" --output table

echo "Cleanup attempted. If models remain, delete with:"
echo "  aws sagemaker delete-model --model-name MODEL_NAME"
