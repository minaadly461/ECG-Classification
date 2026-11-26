
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf

from src.model import build_cnn_lstm

data = np.load("data/processed/ecg_dataset.npz")

X_train = data["X_train"]
Y_train = data["Y_train"]
X_val = data["X_val"]
Y_val = data["Y_val"]

EPOCHS = 30
BATCH_SIZE = 32

mlflow.set_experiment("PTB-XL-ECG-Classification")

with mlflow.start_run(run_name="CNN-LSTM"):

    model = build_cnn_lstm()

    mlflow.log_params({
        "architecture": "CNN-LSTM",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "loss": "binary_crossentropy"
    })

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
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    mlflow.log_metric(
        "best_val_loss",
        min(history.history["val_loss"])
    )

    mlflow.log_metric(
        "best_val_auc",
        max(history.history["val_auc"])
    )

    model.save("models/ecg_cnn_lstm.keras")
    mlflow.log_artifact("models/ecg_cnn_lstm.keras")

    print("CNN-LSTM training completed!")
