
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    multilabel_confusion_matrix
)

# =========================
# 1. Load Test Data
# =========================

data = np.load("data/processed/ecg_dataset.npz")

X_test = data["X_test"]
Y_test = data["Y_test"]

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

# =========================
# 2. Load Thresholds
# =========================

thresholds = np.load("models/ensemble_thresholds.npy")

# =========================
# 3. Load Trained Models
# =========================

cnn = tf.keras.models.load_model("models/ecg_cnn.keras")
gru = tf.keras.models.load_model("models/ecg_cnn_gru.keras")
lstm = tf.keras.models.load_model("models/ecg_cnn_lstm.keras")

# =========================
# 4. Generate Predictions
# =========================

print("Predicting with CNN...")
p_cnn = cnn.predict(X_test, batch_size=32, verbose=1)

print("Predicting with GRU...")
p_gru = gru.predict(X_test, batch_size=32, verbose=1)

print("Predicting with LSTM...")
p_lstm = lstm.predict(X_test, batch_size=32, verbose=1)

# =========================
# 5. Ensemble Predictions
# =========================

ensemble_probs = (p_cnn + p_gru + p_lstm) / 3

predictions = (
    ensemble_probs >= thresholds.reshape(1, -1)
).astype(int)

# =========================
# 6. ROC-AUC Evaluation
# =========================

macro_auc = roc_auc_score(
    Y_test,
    ensemble_probs,
    average="macro"
)

print("\n" + "=" * 50)
print("FINAL TEST EVALUATION")
print("=" * 50)

print(f"Macro ROC-AUC: {macro_auc:.4f}")

# =========================
# 7. Classification Report
# =========================

print("\nClassification Report:")

print(
    classification_report(
        Y_test,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4
    )
)

# =========================
# 8. Confusion Matrices
# =========================

cms = multilabel_confusion_matrix(
    Y_test,
    predictions
)

for i, class_name in enumerate(CLASS_NAMES):

    print(f"\n{class_name} Confusion Matrix:")
    print(cms[i])

# =========================
# 9. Save Results
# =========================

np.savez_compressed(
    "models/ensemble_test_results.npz",
    probabilities=ensemble_probs,
    predictions=predictions,
    thresholds=thresholds,
    y_true=Y_test,
    macro_auc=macro_auc,
    confusion_matrices=cms
)

print("\nResults saved to:")
print("models/ensemble_test_results.npz")
print("\nEvaluation completed!")
