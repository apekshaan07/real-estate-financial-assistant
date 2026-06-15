from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "housing_regression_model.joblib"

FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def load_model():
    return joblib.load(MODEL_PATH)


def predict_housing_value(input_data):
    """
    input_data example:
    {
        "MedInc": 5.0,
        "HouseAge": 20,
        "AveRooms": 6.0,
        "AveBedrms": 1.1,
        "Population": 1200,
        "AveOccup": 3.0,
        "Latitude": 34.05,
        "Longitude": -118.25
    }
    """

    model = load_model()

    df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

    prediction = model.predict(df)[0]

    return {
        "predicted_median_house_value": float(prediction),
        "predicted_value_usd": float(prediction * 100000)
    }


if __name__ == "__main__":
    sample_input = {
        "MedInc": 5.0,
        "HouseAge": 20,
        "AveRooms": 6.0,
        "AveBedrms": 1.1,
        "Population": 1200,
        "AveOccup": 3.0,
        "Latitude": 34.05,
        "Longitude": -118.25
    }

    result = predict_housing_value(sample_input)
    print(result)