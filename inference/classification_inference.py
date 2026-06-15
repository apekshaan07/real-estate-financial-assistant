from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "bank_classification_model.joblib"


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


def load_model():
    return joblib.load(MODEL_PATH)


def predict_subscription(input_data):
    """
    Returns bank marketing subscription prediction and probability.
    """

    model = load_model()

    # Compatibility fix for sklearn LogisticRegression pickle issue
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        classifier = model.named_steps["classifier"]
        if not hasattr(classifier, "multi_class"):
            classifier.multi_class = "auto"

    df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    classes = model.classes_
    probability_map = {
        str(classes[i]): float(probability[i])
        for i in range(len(classes))
    }

    return {
        "prediction": str(prediction),
        "probability_no": probability_map.get("no", 0.0),
        "probability_yes": probability_map.get("yes", 0.0),
    }