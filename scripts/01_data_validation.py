import mlflow
from sklearn.datasets import load_breast_cancer

EXPECTED_CLASSES = 2  # ข้อมูลชุดนี้เป็น binary (malignant / benign) — Wine เดิมเช็ค < 3 ซึ่งจะทำให้ชุดนี้ Failed
MIN_CLASS_BALANCE = 0.20  # คลาสน้อยสุดต้องไม่ต่ำกว่า 20% (Capture 2.2: ลองแก้เป็น 0.45 ให้ไม่ผ่าน)


def validate_data():
    """
    Loads the breast cancer dataset, performs basic validation checks,
    and logs the results to MLflow.
    """
    mlflow.set_experiment("Breast Cancer - Data Validation")

    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        # 1. Load data as a Pandas DataFrame
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        print("Data loaded successfully.")

        # 2. Perform validation checks
        num_rows, num_cols = df.shape
        num_classes = df["target"].nunique()
        missing_values = df.isnull().sum().sum()
        # สัดส่วนของ "คลาสที่น้อยที่สุด" (ดูทุกคลาสแล้วเอาค่าน้อยสุด ไม่ใช่ค่าของคลาสสุดท้าย)
        class_balance = df["target"].value_counts(normalize=True).min()

        print(f"Dataset shape: {num_rows} rows, {num_cols} columns")
        print(f"Number of classes: {num_classes}")
        print(f"Missing values: {missing_values}")
        print(f"Class balance (minority class): {class_balance:.4f}")

        # 3. Log validation results to MLflow
        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_metric("class_balance", class_balance)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("min_class_balance", MIN_CLASS_BALANCE)

        # Check if the data passes our defined criteria
        validation_status = "Success"
        if missing_values > 0:
            validation_status = "Failed"
            print("FAILED: dataset has missing values")
        if num_classes != EXPECTED_CLASSES:
            validation_status = "Failed"
            print(f"FAILED: expected {EXPECTED_CLASSES} classes, got {num_classes}")
        if class_balance < MIN_CLASS_BALANCE:
            validation_status = "Failed"
            print(f"FAILED: class balance {class_balance:.2%} is below {MIN_CLASS_BALANCE:.0%}")

        mlflow.log_param("validation_status", validation_status)
        print(f"Validation status: {validation_status}")

        # 4. คืน exit code ที่ไม่ใช่ 0 เมื่อข้อมูลไม่ผ่าน — ให้ CI จับได้และหยุด pipeline
        if validation_status == "Failed":
            raise SystemExit("Data validation failed — หยุด pipeline ไม่ให้ไปขั้นถัดไป")

        print("Data validation run finished.")


if __name__ == "__main__":
    validate_data()
