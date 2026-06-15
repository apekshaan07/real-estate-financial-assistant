from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "housing_regression_model.joblib"
METRICS_PATH = MODEL_DIR / "housing_regression_metrics.json"


def main():
    print("Loading California Housing dataset...")

    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    print("Dataset shape:", X.shape)
    print("Feature columns:", list(X.columns))

    print("\nBasic EDA:")
    print(X.describe())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    print("\nTraining Random Forest Regressor...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2),
        "features": list(X.columns)
    }

    pd.Series(metrics).to_json(METRICS_PATH, indent=2)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()