from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import joblib
import mlflow
import os


# =========================
# 1. Load dataset
# =========================

data = load_breast_cancer(as_frame=True)
df = data.frame

X = df.drop(columns=["target"])
y = df["target"]


# =========================
# 2. Train/Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)


# =========================
# 3. StandardScaler
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# 4. Create directories
# =========================

os.makedirs("artifacts", exist_ok=True)

train_path = "artifacts/train.csv"
test_path = "artifacts/test.csv"
scaler_path = "artifacts/scaler.joblib"


# =========================
# 5. Save train/test
# =========================

train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
train_df["target"] = y_train.values

test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
test_df["target"] = y_test.values

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

joblib.dump(scaler, scaler_path)


# =========================
# 6. MLflow
# =========================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Breast Cancer Classification")

with mlflow.start_run() as run:

    mlflow.log_param("test_size", 0.25)
    mlflow.log_param("stratify", True)
    mlflow.log_param("scaler", "StandardScaler")

    mlflow.log_metric("training_set_rows", len(X_train))
    mlflow.log_metric("test_set_rows", len(X_test))

    mlflow.log_artifact(train_path)
    mlflow.log_artifact(test_path)
    mlflow.log_artifact(scaler_path)

    print("Preprocessing Run ID:", run.info.run_id)
    print("training_set_rows =", len(X_train))
    print("test_set_rows =", len(X_test))
    print("Preprocessing completed successfully.")