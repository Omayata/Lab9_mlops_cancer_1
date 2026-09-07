from sklearn.datasets import load_breast_cancer
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Basic information
logging.info(f"Dataset shape: {X.shape}")
logging.info(f"Number of classes: {len(set(y))}")

# Class distribution
classes, counts = __import__("numpy").unique(y, return_counts=True)

for cls, count in zip(classes, counts):
    percentage = count / len(y) * 100
    logging.info(
        f"Class {cls} ({data.target_names[cls]}): "
        f"{count} samples ({percentage:.2f}%)"
    )

# Validation
if len(classes) != 2:
    logging.error("VALIDATION FAILED: Dataset must have exactly 2 classes.")
    sys.exit(1)

logging.info("VALIDATION PASSED")
sys.exit(0)