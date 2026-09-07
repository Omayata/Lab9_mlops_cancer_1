from sklearn.datasets import load_breast_cancer

d = load_breast_cancer()

print("Number of samples:", d.data.shape[0])
print("Number of features:", d.data.shape[1])
print("Target names:", d.target_names)

print("\nClass distribution:")
import numpy as np

unique, counts = np.unique(d.target, return_counts=True)

for label, count in zip(unique, counts):
    percentage = count / len(d.target) * 100
    print(f"{label} ({d.target_names[label]}): {count} ({percentage:.2f}%)")