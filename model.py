import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

MODEL_PATH = "saved_model/ann_stock_model.h5"


def build_model(input_shape):
    """Build and compile the ANN model."""
    model = Sequential([
        Flatten(input_shape=input_shape),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dropout(0.2),

        Dense(32, activation='relu'),

        Dense(1, activation='sigmoid')   # Binary: Up / Down
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(X_train, y_train):
    """Train the ANN and save to disk."""
    model = build_model((X_train.shape[1], X_train.shape[2]))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    os.makedirs("saved_model", exist_ok=True)
    model.save(MODEL_PATH)
    return model, history


def load_model():
    """Load a previously saved model."""
    return tf.keras.models.load_model(MODEL_PATH)


def model_exists():
    return os.path.exists(MODEL_PATH)
