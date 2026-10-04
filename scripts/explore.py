"""ส่วนที่ 1 — สำรวจข้อมูล Breast Cancer (Capture 1.2)"""
from sklearn.datasets import load_breast_cancer

d = load_breast_cancer(as_frame=True)

print("Shape (rows, columns):", d.frame.shape)  # (569, 31) = 30 ฟีเจอร์ + target
print("Target names:", d.target_names)  # ['malignant' 'benign'] → 0 = malignant, 1 = benign
print("Number of classes:", d.frame["target"].nunique())
print("\nClass proportion:")
print(d.frame["target"].value_counts(normalize=True))  # 1 → 0.6274, 0 → 0.3726
