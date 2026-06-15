"""Check which assignment integrations are configured and reachable."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from config import get_database_backend, get_integration_status  # noqa: F401
from db import get_property_financials, initialize_database


def check_postgres() -> tuple[bool, str]:
    backend = get_database_backend()
    initialize_database()
    count = len(get_property_financials())
    return backend == "postgres", f"{backend} with {count} property rows"


def check_vertex() -> tuple[bool, str]:
    if os.getenv("USE_VERTEX_ADK", "false").lower() != "true":
        return False, "USE_VERTEX_ADK is not true"
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        return False, "GOOGLE_CLOUD_PROJECT is missing"
    return True, f"Configured for project {os.getenv('GOOGLE_CLOUD_PROJECT')}"


def check_sagemaker() -> tuple[bool, str]:
    regression = os.getenv("SAGEMAKER_REGRESSION_ENDPOINT")
    classification = os.getenv("SAGEMAKER_CLASSIFICATION_ENDPOINT")
    if regression and classification:
        return True, f"Regression={regression}, Classification={classification}"
    missing = []
    if not regression:
        missing.append("SAGEMAKER_REGRESSION_ENDPOINT")
    if not classification:
        missing.append("SAGEMAKER_CLASSIFICATION_ENDPOINT")
    return False, "Missing " + ", ".join(missing)


def check_azure() -> tuple[bool, str]:
    if os.getenv("USE_AZURE_OPENAI", "false").lower() != "true":
        return False, "USE_AZURE_OPENAI=false"
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    if not endpoint or "your-resource" in endpoint or not api_key or api_key == "your-key":
        return False, "Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY"
    try:
        from inference.azure_summary import summarize_with_azure_or_local

        result = summarize_with_azure_or_local("Test quarterly revenue update from Prologis.")
        if result.startswith("Azure fallback used"):
            return False, result[:120]
        return True, f"Deployment {os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o-mini')}"
    except Exception as exc:
        return False, str(exc)[:120]


def main() -> None:
    checks = [
        ("Postgres database", check_postgres()),
        ("Vertex AI ADK config", check_vertex()),
        ("SageMaker endpoints", check_sagemaker()),
        (
            "AWS Bedrock",
            (
                os.getenv("USE_BEDROCK", "false").lower() == "true",
                "USE_BEDROCK=true" if os.getenv("USE_BEDROCK", "false").lower() == "true" else "USE_BEDROCK=false",
            ),
        ),
        ("Azure OpenAI", check_azure()),
        (
            "SEC EDGAR live mode",
            (
                os.getenv("USE_SEC_EDGAR", "true").lower() == "true",
                "USE_SEC_EDGAR=true"
                if os.getenv("USE_SEC_EDGAR", "true").lower() == "true"
                else "USE_SEC_EDGAR=false",
            ),
        ),
    ]

    print("Integration verification\n" + "=" * 40)
    for name, (ok, detail) in checks:
        label = "PASS" if ok else "TODO"
        print(f"[{label}] {name}: {detail}")

    print("\nConfigured status from app:")
    for key, value in get_integration_status().items():
        print(f"- {key}: {value['status']} ({value['detail']})")


if __name__ == "__main__":
    main()
