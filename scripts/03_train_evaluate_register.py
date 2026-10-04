import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ACCURACY_THRESHOLD = 0.95
ROC_AUC_THRESHOLD = 0.98
MODEL_NAME = "cancer-classifier-prod"
MODEL_ALIAS = "staging"


def train_evaluate_register(preprocessing_run_id, C=1.0):
    """
    Loads preprocessed data from the preprocessing run, trains a model, evaluates it
    (accuracy + ROC-AUC), and registers it only if BOTH thresholds are met.
    """
    mlflow.set_experiment("Breast Cancer - Model Training")

    with mlflow.start_run(run_name=f"logistic_regression_C_{C}"):
        print(f"Starting training run with C={C}...")
        mlflow.set_tag("ml.step", "model_training_evaluation")
        mlflow.log_param("preprocessing_run_id", preprocessing_run_id)

        # 1. โหลดข้อมูลจาก Artifacts ของ Preprocessing Run (อ้างด้วย Run ID)
        try:
            local_artifact_path = download_artifacts(
                run_id=preprocessing_run_id, artifact_path="processed_data"
            )
            print(f"Artifacts downloaded to: {local_artifact_path}")
            train_df = pd.read_csv(os.path.join(local_artifact_path, "train.csv"))
            test_df = pd.read_csv(os.path.join(local_artifact_path, "test.csv"))
            print("Successfully loaded data from downloaded artifacts.")
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            print("Please ensure the preprocessing_run_id is correct.")
            sys.exit(1)

        X_train = train_df.drop("target", axis=1)
        y_train = train_df["target"]
        X_test = test_df.drop("target", axis=1)
        y_test = test_df["target"]

        # 2. Pipeline = Scaler + Model → ตอนใช้งานส่งข้อมูลดิบเข้าไปได้เลย
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=C, random_state=42, max_iter=10000)),
        ])
        pipeline.fit(X_train, y_train)

        # 3. ประเมินผล — binary จึงดู ROC-AUC (ใช้ความน่าจะเป็นของคลาส 1 = benign) ประกอบ accuracy
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

        # 4. Log Parameters, Metrics, และ Model (Pipeline)
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc_auc)
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="cancer_classifier_pipeline",
            input_example=X_train.head(5),
        )

        # 5. Quality gate — ต้องผ่าน "ทั้งสองเกณฑ์" จึงจะลงทะเบียน
        acc_pass = acc >= ACCURACY_THRESHOLD
        auc_pass = roc_auc >= ROC_AUC_THRESHOLD
        print(f"Accuracy >= {ACCURACY_THRESHOLD}: {acc_pass}")
        print(f"ROC-AUC  >= {ROC_AUC_THRESHOLD}: {auc_pass}")

        if acc_pass and auc_pass:
            print("QUALITY GATE: PASSED — Registering model...")
            registered_model = mlflow.register_model(model_info.model_uri, MODEL_NAME)
            print(f"Model registered as '{registered_model.name}' version {registered_model.version}")

            client = MlflowClient()
            client.set_registered_model_alias(
                name=MODEL_NAME, alias=MODEL_ALIAS, version=registered_model.version
            )
            print(f"Set alias '@{MODEL_ALIAS}' -> {MODEL_NAME} version {registered_model.version}")
        else:
            print("QUALITY GATE: FAILED — Model is NOT registered.")
        print("Training run finished.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/03_train_evaluate_register.py <preprocessing_run_id> [C_value]")
        sys.exit(1)

    run_id = sys.argv[1]
    c_value = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    train_evaluate_register(preprocessing_run_id=run_id, C=c_value)
