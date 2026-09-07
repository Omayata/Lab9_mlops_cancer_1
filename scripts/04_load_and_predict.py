import mlflow
import mlflow.sklearn
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer


# =========================
# MLflow
# =========================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model_uri = "models:/cancer-classifier-prod@staging"

model = mlflow.sklearn.load_model(model_uri)


# =========================
# Load dataset
# =========================

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target


# =========================
# Load scaler
# =========================

scaler = joblib.load(
    "artifacts/scaler.joblib"
)

X_scaled = pd.DataFrame(
    scaler.transform(X),
    columns=data.feature_names
)


# =========================
# Find first sample
# of each class
# =========================

first_malignant = next(
    i for i, label in enumerate(y)
    if label == 0
)

first_benign = next(
    i for i, label in enumerate(y)
    if label == 1
)


indices = [
    first_malignant,
    first_benign
]


# =========================
# Predict
# =========================

for i in indices:

    sample = X_scaled.iloc[[i]]

    prediction = model.predict(sample)[0]

    actual_name = data.target_names[y[i]]
    predicted_name = data.target_names[prediction]

    correct = prediction == y[i]

    print(f"Row: {i}")
    print(f"Actual: {actual_name}")
    print(f"Predicted: {predicted_name}")
    print(f"Correct: {correct}")
    print("-" * 30)