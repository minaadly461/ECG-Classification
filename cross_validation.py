
import numpy as np
import tensorflow as tf

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

from src.model import build_cnn

# Load training data only
data = np.load("data/processed/ecg_dataset.npz")

X = data["X_train"]
Y = data["Y_train"]

# Cross-validation settings
N_SPLITS = 5
BATCH_SIZE = 32
EPOCHS = 30

skf = MultilabelStratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y), 1):

    print(f"\n========== Fold {fold}/{N_SPLITS} ==========")

    X_fold_train = X[train_idx]
    Y_fold_train = Y[train_idx]

    X_fold_val = X[val_idx]
    Y_fold_val = Y[val_idx]

    # Build a fresh model
    tf.keras.backend.clear_session()
    model = build_cnn()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_fold_train,
        Y_fold_train,
        validation_data=(X_fold_val, Y_fold_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    # Predictions
    probabilities = model.predict(
        X_fold_val,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # Metrics
    auc = roc_auc_score(
        Y_fold_val,
        probabilities,
        average="macro"
    )

    predictions = (probabilities >= 0.5).astype(int)

    f1 = f1_score(
        Y_fold_val,
        predictions,
        average="macro",
        zero_division=0
    )

    fold_results.append([auc, f1])

    print(f"Fold {fold} AUC: {auc:.4f}")
    print(f"Fold {fold} F1:  {f1:.4f}")


# Final results
fold_results = np.array(fold_results)

print("\n========== Cross-Validation Results ==========")

print(f"Mean AUC: {fold_results[:, 0].mean():.4f}")
print(f"Std AUC:  {fold_results[:, 0].std():.4f}")

print(f"Mean F1:  {fold_results[:, 1].mean():.4f}")
print(f"Std F1:   {fold_results[:, 1].std():.4f}")

np.save("models/cnn_cv_results.npy", fold_results)
