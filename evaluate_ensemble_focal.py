
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    classification_report
)

# Load validation data
data = np.load("data/processed/ecg_dataset.npz")
X_val = data["X_val"]
Y_val = data["Y_val"]

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Load models
model_paths = [
    "models/ecg_cnn.keras",
    "models/ecg_cnn_gru.keras",
    "models/ecg_cnn_lstm.keras",
    "models/ecg_cnn_focal.keras"
]

models = [
    tf.keras.models.load_model(path, compile=False)
    for path in model_paths
]

# Predict probabilities
predictions = []

for model in models:
    predictions.append(model.predict(X_val, batch_size=32, verbose=1))

# Soft voting
ensemble_probs = np.mean(predictions, axis=0)

# Optimize thresholds on validation
thresholds = []
Y_pred = np.zeros_like(Y_val)

for i in range(len(CLASS_NAMES)):
    precision, recall, th = precision_recall_curve(
        Y_val[:, i],
        ensemble_probs[:, i]
    )

    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1[:-1])

    threshold = th[best_idx]
    thresholds.append(threshold)

    Y_pred[:, i] = (
        ensemble_probs[:, i] >= threshold
    ).astype(int)

    print(
        f"{CLASS_NAMES[i]} | "
        f"Threshold: {threshold:.4f} | "
        f"F1: {f1[best_idx]:.4f}"
    )

# Metrics
macro_auc = roc_auc_score(
    Y_val,
    ensemble_probs,
    average="macro"
)

print("\nMacro ROC-AUC:", round(macro_auc, 4))

print("\nClassification Report:")
print(
    classification_report(
        Y_val,
        Y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

print("Macro F1:", round(
    np.mean([
        2 * np.sum(Y_val[:, i] * Y_pred[:, i]) /
        (np.sum(Y_val[:, i]) + np.sum(Y_pred[:, i]) + 1e-8)
        for i in range(len(CLASS_NAMES))
    ]),
    4
))

# Save results
np.save("models/ensemble_focal_thresholds.npy", thresholds)

np.savez(
    "models/ensemble_focal_validation_results.npz",
    probabilities=ensemble_probs,
    predictions=Y_pred,
    thresholds=thresholds
)

print("\nResults saved successfully.")
