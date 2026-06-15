#!/usr/bin/env bash
# Configure Azure OpenAI in .env and verify summarization works.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Azure OpenAI setup for Press Release summarization"
echo "=================================================="
echo ""
echo "You need an Azure OpenAI resource first. If you don't have one yet:"
echo "  1. https://portal.azure.com → Create a resource → Azure OpenAI"
echo "  2. In the resource → Keys and Endpoint → copy KEY 1 and Endpoint"
echo "  3. In the resource → Model deployments → Deploy gpt-4o-mini (or gpt-35-turbo)"
echo ""

read -r -p "Azure OpenAI Endpoint (e.g. https://my-resource.openai.azure.com/): " ENDPOINT
read -r -p "Azure OpenAI API Key: " API_KEY
read -r -p "Deployment name [gpt-4o-mini]: " DEPLOYMENT
DEPLOYMENT="${DEPLOYMENT:-gpt-4o-mini}"

if [[ -z "$ENDPOINT" || -z "$API_KEY" ]]; then
  echo "Endpoint and API key are required."
  exit 1
fi

# Ensure trailing slash on endpoint
[[ "$ENDPOINT" != */ ]] && ENDPOINT="${ENDPOINT}/"

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

update_env() {
  local key="$1"
  local value="$2"
  if grep -q "^${key}=" .env; then
    sed -i '' "s|^${key}=.*|${key}=${value}|" .env
  else
    echo "${key}=${value}" >> .env
  fi
}

update_env "USE_AZURE_OPENAI" "true"
update_env "AZURE_OPENAI_ENDPOINT" "$ENDPOINT"
update_env "AZURE_OPENAI_API_KEY" "$API_KEY"
update_env "AZURE_OPENAI_DEPLOYMENT" "$DEPLOYMENT"

echo ""
echo "Testing Azure OpenAI summarization..."
"$ROOT_DIR/venv/bin/python" - <<'PY'
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(".env"), override=True)
from inference.azure_summary import summarize_with_azure_or_local

sample = (
    "Prologis reported strong quarterly results. Revenue grew 8 percent year over year. "
    "The company expanded its logistics portfolio in key markets."
)
result = summarize_with_azure_or_local(sample)
if result.startswith("Azure fallback used"):
    print("FAILED:", result)
    raise SystemExit(1)
print("SUCCESS:", result[:200])
PY

echo ""
echo "Azure OpenAI is configured. Restart Streamlit and open Press Releases."
