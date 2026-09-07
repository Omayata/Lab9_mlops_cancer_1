import sys
import os

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


# =========================
# Arguments
# =========================

if len(sys.argv) != 3:
    print(
        "Usage: python scripts/03_train_evaluate_register.py "
        "<preprocessing_run_id> <C>"
    )
    sys.exit(1)

preprocessing_run_id = sys.argv[1]
C = float(sys.argv[2])


# =========================
# Load preprocessing data
# =========================

train_path = "artifacts/train.csv"
test_path = "artifacts/test.csv"

if not os.path.exists(train_path):
    print(f"ERROR: {train_path} not found")
    sys.exit(1)

if not os.path.exists(test_path):
    print(f"ERROR: {test_path} not found")
    sys.exit(1)


import pandas as pd

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]


# =========================
# MLflow
# =========================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

experiment_name = "Breast Cancer Classification"
mlflow.set_experiment(experiment_name)


with mlflow.start_run() as run:

    # =========================
    # Train
    # =========================

    model = LogisticRegression(
        C=C,
        max_iter=10000,
        random_state=42
    )

    model.fit(X_train, y_train)


    # =========================
    # Evaluate
    # =========================

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)


    # =========================
    # Log parameters / metrics
    # =========================

    mlflow.log_param("C", C)
    mlflow.log_param(
        "preprocessing_run_id",
        preprocessing_run_id
    )

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)


    # =========================
    # Quality Gate
    # =========================

    accuracy_pass = accuracy >= 0.95
    roc_auc_pass = roc_auc >= 0.98

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    if not (accuracy_pass and roc_auc_pass):
        print("QUALITY GATE: FAILED")
        print(
            f"Accuracy >= 0.95: {accuracy_pass}"
        )
        print(
            f"ROC-AUC >= 0.98: {roc_auc_pass}"
        )
        sys.exit(1)


    print("QUALITY GATE: PASSED")


    # =========================
    # Register model
    # =========================

    model_name = "cancer-classifier-prod"

    model_uri = f"runs:/{run.info.run_id}/model"

    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name=model_name
    )

    print(f"Registered model: {model_name}")


    # =========================
    # Set alias
    # =========================

    client = mlflow.tracking.MlflowClient()

    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    latest_version = max(
        int(v.version) for v in versions
    )

    client.set_registered_model_alias(
        model_name,
        "staging",
        latest_version
    )

    print(
        f"Registered as version {latest_version}"
    )

    print(
        f"Alias set: {model_name}@staging"
    )

    print(
        f"Training Run ID: {run.info.run_id}"
    )