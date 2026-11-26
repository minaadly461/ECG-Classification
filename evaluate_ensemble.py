
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve
)

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Load validation data
data = np.load("data/processed/ecg_dataset.npz")

X_val = data["X_val"]
Y_val = data["Y_val"]

# Load three models
cnn = tf.keras.models.load_model("models/ecg_cnn.keras")
gru = tf.keras.models.load_model("models/ecg_cnn_gru.keras")
lstm = tf.keras.models.load_model("models/ecg_cnn_lstm.keras")

# Get predictions
print("Predicting with CNN...")
p_cnn = cnn.predict(X_val, batch_size=32, verbose=1)

print("Predicting with CNN-GRU...")
p_gru = gru.predict(X_val, batch_size=32, verbose=1)

print("Predicting with CNN-LSTM...")
p_lstm = lstm.predict(X_val, batch_size=32, verbose=1)

# Soft Voting Ensemble
ensemble_probs = (p_cnn + p_gru + p_lstm) / 3

# Evaluate ensemble
thresholds = []

print("\nEnsemble Validation Results")
print("=" * 60)

for i, class_name in enumerate(CLASS_NAMES):

    y_true = Y_val[:, i]
    y_prob = ensemble_probs[:, i]

    auc = roc_auc_score(y_true, y_prob)

    precision, recall, threshold_values = precision_recall_curve(
        y_true, y_prob
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-8)
    )

    best_idx = np.argmax(f1_scores)
    threshold = threshold_values[best_idx]

    thresholds.append(threshold)

    print(f"\nClass: {class_name}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"F1-score: {f1_scores[best_idx]:.4f}")
    print(f"Precision: {precision[best_idx]:.4f}")
    print(f"Recall: {recall[best_idx]:.4f}")

# Save ensemble thresholds
np.save(
    "models/ensemble_thresholds.npy",
    np.array(thresholds)
)

print("\nEnsemble thresholds saved!")
