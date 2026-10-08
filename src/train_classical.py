"""
train_classical.py
==================
Author: Student 1 (Data & Classical ML Specialist)
Assignment 1: Training Models on MNIST + Interactive Drawing App

Responsibilities:
- Implement Classical Machine Learning baseline using Scikit-Learn (SVM and/or Random Forest).
- Perform systematic hyperparameter search using GridSearchCV.
- Measure training time, validation performance, and test metrics.
- Save best model artifact to `saved_models/classical_model.joblib`.
- Export benchmark evaluation metrics to `saved_models/classical_metrics.json`.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Ensure root directory is on Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_mnist


def train_classical_model(
    model_type: str = "svm",
    sample_size: int = 10000,
    cv_folds: int = 3,
    output_dir: str = "saved_models",
) -> Dict[str, Any]:
    """
    Trains and tunes a classical ML classifier on MNIST.

    Parameters
    ----------
    model_type : str
        'svm' for Support Vector Classifier or 'rf' for RandomForestClassifier.
    sample_size : int
        Subsample size for hyperparameter search (SVM scales O(N^2) to O(N^3)).
    cv_folds : int
        Number of cross-validation folds in GridSearchCV.
    output_dir : str
        Directory where artifacts and metrics will be saved.

    Returns
    -------
    dict
        Evaluation summary metrics and hyperparameter tuning results.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n==================================================")
    print(f"  Training Classical Model [{model_type.upper()}] (Student 1)")
    print(f"==================================================")

    # 1. Load flattened data
    print(f"[1/4] Loading MNIST data (sample_size={sample_size} for training)...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist(
        flatten=True, normalize=True, subset_size=sample_size
    )
    print(f"      Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    # 2. Configure Model and Hyperparameter Search Grid
    print(f"[2/4] Configuring GridSearchCV for {model_type.upper()}...")
    if model_type.lower() == "svm":
        base_estimator = SVC(probability=True, random_state=42)
        # TODO [Student 1]: Experiment with additional C and gamma values for the final report
        param_grid = {
            "C": [1.0, 10.0],
            "gamma": ["scale", 0.01],
            "kernel": ["rbf"],
        }
    elif model_type.lower() == "rf":
        base_estimator = RandomForestClassifier(random_state=42)
        # TODO [Student 1]: Fine-tune trees, max_depth, and min_samples_split
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [15, 25, None],
            "min_samples_split": [2, 5],
        }
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Choose 'svm' or 'rf'.")

    grid_search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
    )

    # 3. Fit Model & Benchmark Training Time
    print(f"[3/4] Running GridSearchCV optimization ({cv_folds} folds)...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"      Optimization completed in {training_time:.2f}s")
    print(f"      Best Parameters: {grid_search.best_params_}")
    print(f"      Best CV Score:   {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    # 4. Evaluate on Test Set
    print("[4/4] Evaluating best model on test set...")
    t0_eval = time.time()
    y_pred = best_model.predict(X_test)
    eval_latency_total = time.time() - t0_eval
    avg_inference_latency_ms = (eval_latency_total / len(X_test)) * 1000.0

    test_accuracy = float(accuracy_score(y_test, y_pred))
    test_precision = float(precision_score(y_test, y_pred, average="macro"))
    test_recall = float(recall_score(y_test, y_pred, average="macro"))
    test_f1 = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n--- Classical Model Evaluation Summary ---")
    print(f"Test Accuracy:         {test_accuracy:.4f}")
    print(f"Test Macro F1:         {test_f1:.4f}")
    print(f"Test Macro Precision:  {test_precision:.4f}")
    print(f"Test Macro Recall:     {test_recall:.4f}")
    print(f"Training Time:         {training_time:.2f} s")
    print(f"Avg Inference Latency: {avg_inference_latency_ms:.4f} ms/sample")

    # Save artifacts
    model_path = os.path.join(output_dir, "classical_model.joblib")
    metrics_path = os.path.join(output_dir, "classical_metrics.json")

    joblib.dump(best_model, model_path)
    print(f"\n[Saved] Model saved to:   {model_path}")

    metrics_payload = {
        "model_name": f"Classical_{model_type.upper()}",
        "model_type": model_type,
        "best_params": grid_search.best_params_,
        "best_cv_score": float(grid_search.best_score_),
        "test_accuracy": test_accuracy,
        "test_precision_macro": test_precision,
        "test_recall_macro": test_recall,
        "test_f1_macro": test_f1,
        "training_time_sec": float(training_time),
        "inference_latency_ms": float(avg_inference_latency_ms),
        "confusion_matrix": cm,
        "training_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"[Saved] Metrics saved to: {metrics_path}")

    return metrics_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Classical ML Model for MNIST")
    parser.add_argument(
        "--model-type",
        type=str,
        default="svm",
        choices=["svm", "rf"],
        help="Type of classical model to train ('svm' or 'rf')",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Training sample size (for fast SVM training and hyperparameter search)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Cross-validation folds for GridSearchCV",
    )
    args = parser.parse_args()

    train_classical_model(
        model_type=args.model_type,
        sample_size=args.sample_size,
        cv_folds=args.cv_folds,
    )
