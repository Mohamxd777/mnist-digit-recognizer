"""
train_mlp.py
============
Author: Student 2 (Deep Learning & MLP Specialist)
Assignment 1: Training Models on MNIST + Interactive Drawing App

Responsibilities:
- Implement a Multi-Layer Perceptron (MLP) using Keras / TensorFlow.
- STRICT CONSTRAINT: Use Dense layers ONLY (NO convolutional or pooling layers).
- Perform systematic hyperparameter search (e.g., hidden layers, neurons, activation functions, dropout, learning rates).
- Track training history, validation curves, and early stopping.
- Save the best trained model to `saved_models/mlp_model.keras`.
- Save evaluation metrics to `saved_models/mlp_metrics.json`.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

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


def build_mlp_model(
    hidden_units: List[int] = [512, 256],
    activation: str = "relu",
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Constructs an MLP with strictly Dense, Dropout, and BatchNormalization layers.

    Parameters
    ----------
    hidden_units : list of int
        Number of neurons per hidden layer (e.g., [512, 256] or [256, 128, 64]).
    activation : str
        Activation function ('relu', 'leaky_relu', 'elu', etc.).
    dropout_rate : float
        Dropout probability for regularization.
    learning_rate : float
        Adam optimizer initial learning rate.

    Returns
    -------
    keras.Model
        Compiled Keras sequential model.
    """
    model = keras.Sequential()
    # Explicit Input layer for 784-dimensional flattened MNIST vectors
    model.add(layers.Input(shape=(784,), name="input_flattened_pixels"))

    for i, units in enumerate(hidden_units):
        model.add(layers.Dense(units, activation=activation, name=f"dense_{i+1}"))
        model.add(layers.BatchNormalization(name=f"batch_norm_{i+1}"))
        if dropout_rate > 0.0:
            model.add(layers.Dropout(dropout_rate, name=f"dropout_{i+1}"))

    # Final Softmax classification layer for 10 digit classes (0-9)
    model.add(layers.Dense(10, activation="softmax", name="output_digits"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_mlp_model(
    hidden_units: List[int] = [512, 256],
    dropout_rate: float = 0.2,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    output_dir: str = "saved_models",
) -> Dict[str, Any]:
    """
    Trains and validates the MLP model, logging convergence and saving artifacts.

    Parameters
    ----------
    hidden_units : list of int
        Neurons per hidden dense layer.
    dropout_rate : float
        Dropout probability.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Batch size for gradient descent.
    learning_rate : float
        Initial learning rate.
    output_dir : str
        Directory where artifacts are saved.

    Returns
    -------
    dict
        Evaluation metrics dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n==================================================")
    print(f"  Training MLP Model (Dense Layers ONLY) (Student 2)")
    print(f"==================================================")

    # 1. Load data
    print("[1/4] Loading flattened MNIST data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist(
        flatten=True, normalize=True, val_split=0.1
    )
    print(f"      Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # 2. Build model
    print(f"[2/4] Building MLP architecture with layers {hidden_units}...")
    model = build_mlp_model(
        hidden_units=hidden_units,
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
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    # 3. Train Model
    print(f"[3/4] Training MLP for up to {epochs} epochs (batch_size={batch_size})...")
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

    # 4. Evaluate Model on Test Set
    print("[4/4] Evaluating MLP on test set...")
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

    print(f"\n--- MLP Evaluation Summary ---")
    print(f"Test Accuracy:         {test_accuracy:.4f}")
    print(f"Test Macro F1:         {test_f1:.4f}")
    print(f"Test Macro Precision:  {test_precision:.4f}")
    print(f"Test Macro Recall:     {test_recall:.4f}")
    print(f"Training Time:         {training_time:.2f} s")
    print(f"Avg Inference Latency: {avg_inference_latency_ms:.4f} ms/sample")
    print(f"Total Parameters:      {model.count_params():,}")

    # Save artifacts
    model_path = os.path.join(output_dir, "mlp_model.keras")
    metrics_path = os.path.join(output_dir, "mlp_metrics.json")

    model.save(model_path)
    print(f"\n[Saved] Model saved to:   {model_path}")

    # Convert history floats for JSON serialization
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    metrics_payload = {
        "model_name": "Multilayer_Perceptron_MLP",
        "architecture": {
            "hidden_units": hidden_units,
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


def run_hyperparameter_grid(output_dir: str = "saved_models"):
    """
    TODO [Student 2]: Run systematic search over architectures
    e.g., hidden layers: [256], [512, 256], [512, 256, 128]
    and dropout rates: [0.0, 0.2, 0.4] to include comparative tables in REPORT_TEMPLATE.md
    """
    print("[Student 2 TODO] Implement automated grid sweep over multiple MLP topologies.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP (Dense-Only) Model for MNIST")
    parser.add_argument(
        "--hidden-units",
        nargs="+",
        type=int,
        default=[512, 256],
        help="List of neuron counts for hidden layers (e.g. 512 256)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability (0.0 to 0.5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate",
    )
    args = parser.parse_args()

    train_mlp_model(
        hidden_units=args.hidden_units,
        dropout_rate=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
