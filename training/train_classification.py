from pathlib import Path
import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "bank_classification_model.joblib"
METRICS_PATH = MODEL_DIR / "bank_classification_metrics.json"


def main():
    print("Loading Bank Marketing dataset from UCI...")

    bank_marketing = fetch_ucirepo(id=222)

    X = bank_marketing.data.features
    y = bank_marketing.data.targets

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    print("Dataset shape:", X.shape)
    print("Target distribution:")
    print(y.value_counts())

    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print("Categorical columns:", categorical_columns)
    print("Numeric columns:", numeric_columns)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training Logistic Regression classifier...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label="yes")
    recall = recall_score(y_test, predictions, pos_label="yes")
    f1 = f1_score(y_test, predictions, pos_label="yes")
    cm = confusion_matrix(y_test, predictions).tolist()

    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "target_values": sorted(y.unique().tolist()),
    }

    pd.Series(metrics).to_json(METRICS_PATH, indent=2)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()