import json
import os

import joblib
import numpy as np
import pandas as pd


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "housing_regression_model.joblib"))


def input_fn(request_body, request_content_type="application/json"):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        feature_columns = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ]
        return pd.DataFrame([data], columns=feature_columns)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    prediction = float(model.predict(input_data)[0])
    return {
        "predicted_median_house_value": prediction,
        "predicted_value_usd": prediction * 100000,
    }


def output_fn(prediction, response_content_type="application/json"):
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}")
