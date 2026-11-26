import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(input_shape=(1000, 12), num_classes=5):

    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv1D(64, kernel_size=7, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(128, kernel_size=5, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(256, kernel_size=3, activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),

        layers.Dense(num_classes, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(
                multi_label=True,
                num_labels=num_classes,
                name="auc"
            )
        ]
    )

    return model


def build_cnn_gru(input_shape=(1000, 12), num_classes=5):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # CNN feature extraction
        layers.Conv1D(64, kernel_size=7, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(128, kernel_size=5, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        # GRU temporal learning
        layers.GRU(128, return_sequences=False),

        # Classification
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(
                multi_label=True,
                num_labels=num_classes,
                name="auc"
            )
        ]
    )

    return model


def build_cnn_lstm(input_shape=(1000, 12), num_classes=5):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # CNN feature extraction
        layers.Conv1D(64, kernel_size=7, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(128, kernel_size=5, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        # LSTM temporal learning
        layers.LSTM(128, return_sequences=False),

        # Classification
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(
                multi_label=True,
                num_labels=num_classes,
                name="auc"
            )
        ]
    )

    return model
