import mlflow
from sklearn.datasets import load_breast_cancer

MODEL_NAME = "cancer-classifier-prod"
MODEL_ALIAS = "staging"


def load_and_predict():
    """
    Simulates a production scenario: load the model by alias from the Model Registry
    and predict the first sample of each class (malignant, benign).
    """
    print(f"Loading model '{MODEL_NAME}' with alias '@{MODEL_ALIAS}'...")
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(f"Please make sure a model version has the alias '@{MODEL_ALIAS}' in the MLflow UI.")
        return

    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    target_names = data.target_names  # ['malignant', 'benign']

    # รายแรกของแต่ละคลาส: malignant (0) และ benign (1)
    for cls in [0, 1]:
        idx = y[y == cls].index[0]
        sample = X.loc[[idx]]  # ข้อมูลดิบ — scaler อยู่ใน pipeline แล้ว ไม่ต้อง scale เอง
        prediction = int(model.predict(sample)[0])
        actual_name = target_names[cls]
        predicted_name = target_names[prediction]

        print("-" * 30)
        print(f"Row: {idx}")
        print(f"Actual: {actual_name}")
        print(f"Predicted: {predicted_name}")
        print(f"Correct: {prediction == cls}")
    print("-" * 30)


if __name__ == "__main__":
    load_and_predict()
