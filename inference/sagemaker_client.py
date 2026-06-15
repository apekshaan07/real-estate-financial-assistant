import json
import os
from pathlib import Path

from inference.regression_inference import predict_housing_value
from inference.classification_inference import predict_subscription


def call_sagemaker_endpoint(endpoint_name, payload):
    import boto3

    region = os.getenv("AWS_REGION", "us-east-1")
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = response["Body"].read().decode("utf-8")
    parsed = json.loads(result)
    parsed["inference_source"] = f"sagemaker:{endpoint_name}"
    return parsed


BASE_DIR = Path(__file__).resolve().parent.parent
REGRESSION_MODEL_PATH = BASE_DIR / "models" / "housing_regression_model.joblib"


def _local_regression_available() -> bool:
    return REGRESSION_MODEL_PATH.exists()


def predict_housing_value_cloud_or_local(input_data):
    endpoint_name = os.getenv("SAGEMAKER_REGRESSION_ENDPOINT")

    if endpoint_name:
        try:
            return call_sagemaker_endpoint(endpoint_name, input_data)
        except Exception as exc:
            if _local_regression_available():
                local_result = predict_housing_value(input_data)
                local_result["inference_source"] = "local_fallback"
                local_result["warning"] = f"SageMaker regression call failed: {exc}"
                return local_result
            return {
                "predicted_median_house_value": None,
                "predicted_value_usd": None,
                "inference_source": "unavailable",
                "error": (
                    "SageMaker regression call failed and the local regression model is not "
                    "deployed on this host. Update AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                    "in Streamlit Secrets with valid IAM credentials."
                ),
                "warning": str(exc),
            }

    if _local_regression_available():
        result = predict_housing_value(input_data)
        result["inference_source"] = "local_model"
        return result

    return {
        "predicted_median_house_value": None,
        "predicted_value_usd": None,
        "inference_source": "unavailable",
        "error": "No SageMaker endpoint or local regression model configured.",
    }


def predict_subscription_cloud_or_local(input_data):
    endpoint_name = os.getenv("SAGEMAKER_CLASSIFICATION_ENDPOINT")

    if endpoint_name:
        try:
            return call_sagemaker_endpoint(endpoint_name, input_data)
        except Exception as exc:
            local_result = predict_subscription(input_data)
            local_result["inference_source"] = "local_fallback"
            local_result["warning"] = f"SageMaker classification call failed: {exc}"
            return local_result

    result = predict_subscription(input_data)
    result["inference_source"] = "local_model"
    return result
