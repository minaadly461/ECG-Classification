
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    classification_report
)

# =========================
# 1. Load Validation Data
# =========================

data = np.load("data/processed/ecg_dataset.npz")

X_val = data["X_val"]
Y_val = data["Y_val"]

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

# =========================
# 2. Load Weighted Model
# =========================

model = tf.keras.models.load_model(
    "models/ecg_cnn_weighted.keras",
    compile=False
)

# =========================
# 3. Predictions
# =========================

probabilities = model.predict(
    X_val,
    batch_size=32,
    verbose=1
)

# =========================
# 4. Optimize Thresholds
# =========================

thresholds = []

for i, class_name in enumerate(CLASS_NAMES):

    precision, recall, threshold_values = precision_recall_curve(
        Y_val[:, i],
        probabilities[:, i]
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-8)
    )

    best_idx = np.argmax(f1_scores)

    best_threshold = threshold_values[best_idx]

    thresholds.append(best_threshold)

    print(f"\n{class_name}")
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Best F1: {f1_scores[best_idx]:.4f}")

thresholds = np.array(thresholds)

# =========================
# 5. Final Validation Metrics
# =========================

predictions = (
    probabilities >= thresholds.reshape(1, -1)
).astype(int)

macro_auc = roc_auc_score(
    Y_val,
    probabilities,
    average="macro"
)

print("\n" + "=" * 50)
print("WEIGHTED CNN VALIDATION RESULTS")
print("=" * 50)

print(f"Macro ROC-AUC: {macro_auc:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        Y_val,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4
    )
)

# =========================
# 6. Save Results
# =========================

np.save(
    "models/weighted_thresholds.npy",
    thresholds
)

np.savez_compressed(
    "models/weighted_validation_results.npz",
    probabilities=probabilities,
    predictions=predictions,
    thresholds=thresholds,
    y_true=Y_val,
    macro_auc=macro_auc
)

print("\nResults saved successfully!")
