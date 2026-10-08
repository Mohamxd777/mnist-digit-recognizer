"""
train_cnn.py
============
Author: Student 3 (CNN & Model Evaluation Specialist)
Assignment 1: Training Models on MNIST + Interactive Drawing App

Responsibilities:
- Implement a simple introductory Convolutional Neural Network (CNN) using Keras / TensorFlow.
- ARCHITECTURE CONSTRAINT: Introductory architecture (Conv2D + MaxPooling2D + Dense).
  NO transfer learning, NO pre-trained weights, and NO overly deep networks (ResNet/VGG).
- Perform hyperparameter tuning (filter counts, kernel sizes, dropout rates, learning rates).
- Track validation metrics and save the best trained model to `saved_models/cnn_model.keras`.
- Export benchmark evaluation metrics to `saved_models/cnn_metrics.json`.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Ensure root directory is on Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_mnist


def build_introductory_cnn(
    num_filters_1: int = 32,
    num_filters_2: int = 64,
    kernel_size: int = 3,
    dense_units: int = 128,
    dropout_rate: float = 0.25,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Builds a clean, introductory 2D Convolutional Neural Network.

    Parameters
    ----------
    num_filters_1 : int
        Number of filters in first Conv2D layer.
    num_filters_2 : int
        Number of filters in second Conv2D layer.
    kernel_size : int
        Convolution spatial window size (e.g., 3 for 3x3).
    dense_units : int
        Number of units in the post-convolution dense hidden layer.
    dropout_rate : float
        Dropout probability before output layer.
    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    keras.Model
        Compiled introductory CNN model.
    """
    model = keras.Sequential(
        [
            # Input Layer: 28x28 grayscale image with 1 channel
            layers.Input(shape=(28, 28, 1), name="input_image_tensor"),
            
            # Conv Block 1
            layers.Conv2D(
                filters=num_filters_1,
                kernel_size=(kernel_size, kernel_size),
                activation="relu",
                padding="same",
                name="conv2d_1",
            ),
            layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),
            
            # Conv Block 2
            layers.Conv2D(
                filters=num_filters_2,
                kernel_size=(kernel_size, kernel_size),
                activation="relu",
                padding="same",
                name="conv2d_2",
            ),
            layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2"),
            
            # Flatten to 1D vector
            layers.Flatten(name="flatten"),
            
            # Fully-Connected Dense Block
            layers.Dense(dense_units, activation="relu", name="dense_features"),
            layers.Dropout(dropout_rate, name="dropout"),
            
            # Final 10-class Softmax layer
            layers.Dense(10, activation="softmax", name="output_digit_probabilities"),
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn_model(
    num_filters_1: int = 32,
    num_filters_2: int = 64,
    dense_units: int = 128,
    dropout_rate: float = 0.25,
    epochs: int = 15,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    output_dir: str = "saved_models",
) -> Dict[str, Any]:
    """
    Trains and validates the introductory CNN model on MNIST.

    Parameters
    ----------
    num_filters_1 : int
        First Conv layer filter count.
    num_filters_2 : int
        Second Conv layer filter count.
    dense_units : int
        Dense layer unit count.
    dropout_rate : float
        Dropout rate.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Batch size.
    learning_rate : float
        Optimizer learning rate.
    output_dir : str
        Directory to save models and metrics.

    Returns
    -------
    dict
        Evaluation metrics dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n==================================================")
    print(f"  Training Introductory CNN Model (Student 3)")
    print(f"==================================================")

    # 1. Load data with shape (N, 28, 28, 1)
    print("[1/4] Loading MNIST spatial channel data (N, 28, 28, 1)...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist(
        flatten=False, normalize=True, val_split=0.1
    )
    print(f"      Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # 2. Build introductory CNN
    print("[2/4] Building Introductory CNN architecture...")
    model = build_introductory_cnn(
        num_filters_1=num_filters_1,
        num_filters_2=num_filters_2,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    # 3. Train Model
    print(f"[3/4] Training CNN for up to {epochs} epochs (batch_size={batch_size})...")
    start_time = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    training_time = time.time() - start_time
    print(f"      Training completed in {training_time:.2f}s")

    # 4. Evaluate on Test Set
    print("[4/4] Evaluating CNN on test set...")
    t0_eval = time.time()
    y_probs = model.predict(X_test, batch_size=batch_size, verbose=0)
    eval_latency_total = time.time() - t0_eval
    avg_inference_latency_ms = (eval_latency_total / len(X_test)) * 1000.0

    y_pred = np.argmax(y_probs, axis=1)

    test_accuracy = float(accuracy_score(y_test, y_pred))
    test_precision = float(precision_score(y_test, y_pred, average="macro"))
    test_recall = float(recall_score(y_test, y_pred, average="macro"))
    test_f1 = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n--- CNN Evaluation Summary ---")
    print(f"Test Accuracy:         {test_accuracy:.4f}")
    print(f"Test Macro F1:         {test_f1:.4f}")
    print(f"Test Macro Precision:  {test_precision:.4f}")
    print(f"Test Macro Recall:     {test_recall:.4f}")
    print(f"Training Time:         {training_time:.2f} s")
    print(f"Avg Inference Latency: {avg_inference_latency_ms:.4f} ms/sample")
    print(f"Total Parameters:      {model.count_params():,}")

    # Save artifacts
    model_path = os.path.join(output_dir, "cnn_model.keras")
    metrics_path = os.path.join(output_dir, "cnn_metrics.json")

    model.save(model_path)
    print(f"\n[Saved] Model saved to:   {model_path}")

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    metrics_payload = {
        "model_name": "Convolutional_Neural_Network_CNN",
        "architecture": {
            "filters_1": num_filters_1,
            "filters_2": num_filters_2,
            "dense_units": dense_units,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "total_parameters": int(model.count_params()),
            "trainable_parameters": int(
                np.sum([keras.backend.count_params(w) for w in model.trainable_weights])
            ),
        },
        "test_accuracy": test_accuracy,
        "test_precision_macro": test_precision,
        "test_recall_macro": test_recall,
        "test_f1_macro": test_f1,
        "training_time_sec": float(training_time),
        "inference_latency_ms": float(avg_inference_latency_ms),
        "epochs_trained": len(history.history["loss"]),
        "confusion_matrix": cm,
        "history": history_dict,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"[Saved] Metrics saved to: {metrics_path}")

    return metrics_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Introductory CNN Model on MNIST")
    parser.add_argument(
        "--filters-1",
        type=int,
        default=32,
        help="Number of filters for 1st Conv2D layer",
    )
    parser.add_argument(
        "--filters-2",
        type=int,
        default=64,
        help="Number of filters for 2nd Conv2D layer",
    )
    parser.add_argument(
        "--dense-units",
        type=int,
        default=128,
        help="Dense layer hidden units",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.25,
        help="Dropout rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    args = parser.parse_args()

    train_cnn_model(
        num_filters_1=args.filters_1,
        num_filters_2=args.filters_2,
        dense_units=args.dense_units,
        dropout_rate=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
