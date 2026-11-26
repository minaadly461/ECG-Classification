
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow

from sklearn.utils.class_weight import compute_class_weight
from src.model import build_cnn

# =========================
# 1. Load Data
# =========================

data = np.load("data/processed/ecg_dataset.npz")

X_train = data["X_train"]
Y_train = data["Y_train"]

X_val = data["X_val"]
Y_val = data["Y_val"]

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

# =========================
# 2. Calculate Class Weights
# =========================

class_weights = []

for i in range(Y_train.shape[1]):

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=Y_train[:, i]
    )

    class_weights.append(weights[1])

class_weights = np.array(class_weights)

print("Positive Class Weights:")

for name, weight in zip(CLASS_NAMES, class_weights):
    print(f"{name}: {weight:.4f}")

# =========================
# 3. Custom Weighted Loss
# =========================

weights_tensor = tf.constant(
    class_weights,
    dtype=tf.float32
)

def weighted_binary_crossentropy(y_true, y_pred):

    epsilon = tf.keras.backend.epsilon()

    y_pred = tf.clip_by_value(
        y_pred,
        epsilon,
        1 - epsilon
    )

    loss = -(
        weights_tensor * y_true * tf.math.log(y_pred)
        +
        (1 - y_true) * tf.math.log(1 - y_pred)
    )

    return tf.reduce_mean(loss, axis=-1)

# =========================
# 4. Build Model
# =========================

model = build_cnn()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss=weighted_binary_crossentropy,
    metrics=[
        tf.keras.metrics.AUC(
            multi_label=True,
            num_labels=5,
            name="auc"
        )
    ]
)

# =========================
# 5. MLflow Experiment
# =========================

mlflow.set_experiment("PTB-XL-ECG-Classification")

with mlflow.start_run(run_name="CNN-Class-Weighted"):

    mlflow.log_params({
        "architecture": "CNN",
        "loss": "weighted_binary_crossentropy",
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 0.001
    })

    for name, weight in zip(CLASS_NAMES, class_weights):
        mlflow.log_param(f"{name}_positive_weight", float(weight))

    # =========================
    # 6. Train
    # =========================

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # =========================
    # 7. Save Model
    # =========================

    model.save("models/ecg_cnn_weighted.keras")

    mlflow.log_metric(
        "best_val_loss",
        float(min(history.history["val_loss"]))
    )

    mlflow.log_metric(
        "best_val_auc",
        float(max(history.history["val_auc"]))
    )

    mlflow.log_artifact(
        "models/ecg_cnn_weighted.keras"
    )

print("\nWeighted CNN training completed!")
