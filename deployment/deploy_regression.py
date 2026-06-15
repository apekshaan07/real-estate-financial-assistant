"""Deploy the California Housing Random Forest model to Amazon SageMaker."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import sagemaker
from sagemaker.sklearn import SKLearnModel

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from deployment.sagemaker_utils import upload_artifact_to_s3
from scripts.package_sagemaker_models import package_model

DEPLOY_DIR = BASE_DIR / "deployment" / "regression"


def deploy(endpoint_name: str, instance_type: str = "ml.t2.medium") -> str:
    role = os.environ.get("SAGEMAKER_ROLE")
    if not role:
        raise EnvironmentError("Set SAGEMAKER_ROLE to your SageMaker execution role ARN.")

    artifact_path = package_model(
        "housing_regression_model.joblib",
        "housing_regression_model.tar.gz",
    )

    session = sagemaker.Session()
    model_data = upload_artifact_to_s3(
        artifact_path,
        key_prefix="models/housing-regression",
        session=session,
    )

    model = SKLearnModel(
        model_data=model_data,
        role=role,
        entry_point="inference.py",
        source_dir=str(DEPLOY_DIR),
        framework_version="1.4-2",
        sagemaker_session=session,
    )

    print(f"Deploying endpoint: {endpoint_name} (this may take 5-15 minutes)...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )
    return predictor.endpoint_name


def main():
    parser = argparse.ArgumentParser(description="Deploy housing regression model to SageMaker")
    parser.add_argument("--endpoint-name", default="housing-regression-endpoint")
    parser.add_argument("--instance-type", default="ml.t2.medium")
    args = parser.parse_args()

    endpoint = deploy(args.endpoint_name, args.instance_type)
    print(f"Deployed regression endpoint: {endpoint}")
    print(f"export SAGEMAKER_REGRESSION_ENDPOINT={endpoint}")


if __name__ == "__main__":
    main()
