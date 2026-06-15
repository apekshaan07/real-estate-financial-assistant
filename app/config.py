"""Environment loading and integration status helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


def _apply_streamlit_secrets() -> None:
    """Load Streamlit Community Cloud secrets into os.environ."""
    try:
        import streamlit as st

        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float, bool)):
                os.environ.setdefault(str(key), str(value))
    except Exception:
        pass


load_dotenv(ENV_PATH, override=False)
_apply_streamlit_secrets()


def get_database_backend() -> str:
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url.startswith("postgresql"):
        return "sqlite"
    try:
        from db import get_active_backend

        return get_active_backend()
    except ImportError:
        return "postgres"


def get_integration_status() -> dict[str, dict[str, str]]:
    backend = get_database_backend()
    postgres_ready = backend == "postgres"

    vertex_ready = (
        os.getenv("USE_VERTEX_ADK", "false").lower() == "true"
        and bool(os.getenv("GOOGLE_CLOUD_PROJECT"))
    )
    sagemaker_regression = bool(os.getenv("SAGEMAKER_REGRESSION_ENDPOINT"))
    sagemaker_classification = bool(os.getenv("SAGEMAKER_CLASSIFICATION_ENDPOINT"))
    bedrock_ready = os.getenv("USE_BEDROCK", "false").lower() == "true"
    azure_ready = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"
    sec_edgar_live = os.getenv("USE_SEC_EDGAR", "true").lower() == "true"

    return {
        "database": {
            "status": "ready" if postgres_ready else "fallback",
            "detail": "PostgreSQL" if postgres_ready else "SQLite (Postgres unavailable — start Docker)",
        },
        "vertex_adk": {
            "status": "ready" if vertex_ready else "fallback",
            "detail": "Vertex AI ADK enabled" if vertex_ready else "Local chat router",
        },
        "sagemaker_regression": {
            "status": "ready" if sagemaker_regression else "fallback",
            "detail": os.getenv("SAGEMAKER_REGRESSION_ENDPOINT", "Local .joblib model"),
        },
        "sagemaker_classification": {
            "status": "ready" if sagemaker_classification else "fallback",
            "detail": os.getenv("SAGEMAKER_CLASSIFICATION_ENDPOINT", "Local .joblib model"),
        },
        "bedrock": {
            "status": "ready" if bedrock_ready else "fallback",
            "detail": "AWS Bedrock summarization" if bedrock_ready else "Local summary fallback",
        },
        "azure_openai": {
            "status": "ready" if azure_ready else "fallback",
            "detail": "Azure OpenAI summarization" if azure_ready else "Local summary fallback",
        },
        "sec_edgar": {
            "status": "ready" if sec_edgar_live else "fallback",
            "detail": "Live SEC EDGAR fetch" if sec_edgar_live else "Mock SEC metrics",
        },
    }
