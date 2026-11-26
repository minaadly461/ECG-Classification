
import numpy as np
import tensorflow as tf
import mlflow

from src.model import build_cnn

# =========================
# 1. Load Data
# =========================

data = np.load("data/processed/ecg_dataset.npz")

X_train = data["X_train"]
Y_train = data["Y_train"]

X_val = data["X_val"]
Y_val = data["Y_val"]

# =========================
# 2. Focal Loss
# =========================

ALPHA = 0.25
GAMMA = 2.0

def binary_focal_loss(y_true, y_pred):

    epsilon = tf.keras.backend.epsilon()

    y_pred = tf.clip_by_value(
        y_pred,
        epsilon,
        1.0 - epsilon
    )

    p_t = (
        y_true * y_pred
        + (1 - y_true) * (1 - y_pred)
    )

    alpha_t = (
        y_true * ALPHA
        + (1 - y_true) * (1 - ALPHA)
    )

    focal_weight = alpha_t * tf.pow(
        1.0 - p_t,
        GAMMA
    )

    loss = -focal_weight * tf.math.log(p_t)

    return tf.reduce_mean(loss, axis=-1)

# =========================
# 3. Build Model
# =========================

model = build_cnn()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss=binary_focal_loss,
    metrics=[
        tf.keras.metrics.AUC(
            multi_label=True,
            num_labels=5,
            name="auc"
        )
    ]
)

# =========================
# 4. MLflow
# =========================

mlflow.set_experiment("PTB-XL-ECG-Classification")

with mlflow.start_run(run_name="CNN-Focal-Loss"):

    mlflow.log_params({
        "architecture": "CNN",
        "loss": "binary_focal_loss",
        "alpha": ALPHA,
        "gamma": GAMMA,
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 0.001
    })

    # =========================
    # 5. Train
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
    # 6. Save Model
    # =========================

    model.save("models/ecg_cnn_focal.keras")

    mlflow.log_metric(
        "best_val_loss",
        float(min(history.history["val_loss"]))
    )

    mlflow.log_metric(
        "best_val_auc",
        float(max(history.history["val_auc"]))
    )

    mlflow.log_artifact(
        "models/ecg_cnn_focal.keras"
    )

print("\nFocal Loss training completed!")
