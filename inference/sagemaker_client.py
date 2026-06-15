import json
import os

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


def predict_housing_value_cloud_or_local(input_data):
    endpoint_name = os.getenv("SAGEMAKER_REGRESSION_ENDPOINT")

    if endpoint_name:
        try:
            return call_sagemaker_endpoint(endpoint_name, input_data)
        except Exception as exc:
            local_result = predict_housing_value(input_data)
            local_result["inference_source"] = "local_fallback"
            local_result["warning"] = f"SageMaker regression call failed: {exc}"
            return local_result

    result = predict_housing_value(input_data)
    result["inference_source"] = "local_model"
    return result


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
