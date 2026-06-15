"""Package trained joblib models into SageMaker model.tar.gz artifacts."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "deployment" / "artifacts"


def package_model(source_name: str, target_name: str) -> Path:
    source_path = MODELS_DIR / source_name
    if not source_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {source_path}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = ARTIFACTS_DIR / target_name

    with tarfile.open(target_path, "w:gz") as tar:
        tar.add(source_path, arcname=source_name)

    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SageMaker model.tar.gz files")
    parser.add_argument(
        "--target",
        choices=["regression", "classification", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.target in {"regression", "all"}:
        path = package_model(
            "housing_regression_model.joblib",
            "housing_regression_model.tar.gz",
        )
        print(f"Created regression artifact: {path}")

    if args.target in {"classification", "all"}:
        path = package_model(
            "bank_classification_model.joblib",
            "bank_classification_model.tar.gz",
        )
        print(f"Created classification artifact: {path}")


if __name__ == "__main__":
    main()
