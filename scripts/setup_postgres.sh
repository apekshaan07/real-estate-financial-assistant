#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Starting PostgreSQL with Docker Compose..."
docker compose up -d

echo "Waiting for Postgres to become ready..."
for _ in {1..20}; do
  if docker compose exec -T postgres pg_isready -U postgres >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/real_estate_finance"

if [ ! -f .env ]; then
  cp .env.example .env
fi

if grep -q "^DATABASE_URL=" .env; then
  sed -i '' 's|^DATABASE_URL=.*|DATABASE_URL=postgresql://postgres:postgres@localhost:5432/real_estate_finance|' .env
else
  echo "DATABASE_URL=postgresql://postgres:postgres@localhost:5432/real_estate_finance" >> .env
fi

echo "Initializing database tables and seed data..."
"$ROOT_DIR/venv/bin/python" - <<'PY'
import os
os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/real_estate_finance"
import sys
sys.path.insert(0, "app")
from db import initialize_database, get_property_financials
initialize_database(force=True)
print(f"Loaded {len(get_property_financials())} property records into Postgres")
PY

echo "Postgres setup complete."
echo "DATABASE_URL is set in .env"
