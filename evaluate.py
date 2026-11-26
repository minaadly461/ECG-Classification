
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score
)

# Class names
CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Load validation data
data = np.load("data/processed/ecg_dataset.npz")

X_val = data["X_val"]
Y_val = data["Y_val"]

# Load trained model
model = tf.keras.models.load_model("models/ecg_cnn_lstm.keras")

# Predict probabilities
probabilities = model.predict(
    X_val,
    batch_size=32,
    verbose=1
)

# Evaluate each class
thresholds = []

print("\nValidation Results")
print("=" * 60)

for i, class_name in enumerate(CLASS_NAMES):

    y_true = Y_val[:, i]
    y_prob = probabilities[:, i]

    # ROC-AUC
    auc = roc_auc_score(y_true, y_prob)

    # Find best threshold based on F1-score
    precision, recall, threshold_values = precision_recall_curve(
        y_true,
        y_prob
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-8)
    )

    best_idx = np.argmax(f1_scores)

    best_threshold = threshold_values[best_idx]
    best_f1 = f1_scores[best_idx]

    thresholds.append(best_threshold)

    print(f"\nClass: {class_name}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Best F1-score: {best_f1:.4f}")
    print(f"Precision: {precision[best_idx]:.4f}")
    print(f"Recall: {recall[best_idx]:.4f}")

# Save optimized thresholds
thresholds = np.array(thresholds)


np.save("models/cnn_lstm_thresholds.npy", thresholds)

print("\n" + "=" * 60)
print("Thresholds saved successfully!")
print(thresholds)
