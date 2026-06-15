import json
import os

import joblib
import pandas as pd


FEATURE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day_of_week",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "bank_classification_model.joblib"))


def input_fn(request_body, request_content_type="application/json"):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return pd.DataFrame([data], columns=FEATURE_COLUMNS)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_
    probability_map = {str(classes[i]): float(probabilities[i]) for i in range(len(classes))}
    return {
        "prediction": str(prediction),
        "probability_no": probability_map.get("no", 0.0),
        "probability_yes": probability_map.get("yes", 0.0),
    }


def output_fn(prediction, response_content_type="application/json"):
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}")
