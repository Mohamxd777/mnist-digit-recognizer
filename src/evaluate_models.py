"""
evaluate_models.py
==================
Author: Student 3 (CNN & Model Evaluation Specialist)
Assignment 1: Training Models on MNIST + Interactive Drawing App

Responsibilities:
- Load all three saved models:
    1. Classical ML (`saved_models/classical_model.joblib`)
    2. Multilayer Perceptron (`saved_models/mlp_model.keras`)
    3. Convolutional Neural Network (`saved_models/cnn_model.keras`)
- Compute uniform test metrics across models:
    - Test Accuracy (%)
    - Macro F1-Score
    - Training Duration (from logged metrics JSONs)
    - Empirical Inference Latency (ms / sample on test batch)
    - Model Complexity (Trainable parameters count & Disk file size)
- Generate a comprehensive comparison table (Markdown & CSV).
- Generate comparison visualization plots for the final report (`REPORT_TEMPLATE.md`).
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf

# Ensure root directory is on Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_mnist


def get_file_size_mb(filepath: str) -> float:
    """Returns file size in megabytes."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0


def benchmark_model_inference(
    predict_fn,
    X_samples: np.ndarray,
    n_iterations: int = 5,
) -> float:
    """
    Measures average inference latency in milliseconds per sample.
    """
    # Warmup pass
    _ = predict_fn(X_samples[:10])

    latencies = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = predict_fn(X_samples)
        t_elapsed = time.perf_counter() - t0
        latencies.append((t_elapsed / len(X_samples)) * 1000.0)

    return float(np.mean(latencies))


def evaluate_all_models(models_dir: str = "saved_models", test_split: Optional[float] = 0.30) -> pd.DataFrame:
    """
    Loads saved models and metrics, calculates uniform benchmarks,
    and returns a comparison summary DataFrame.
    """
    print(f"\n==================================================")
    print(f"  Cross-Model Benchmark & Evaluation (Student 3)")
    if test_split is not None:
        print(f"  Evaluation Split: {int((1 - test_split)*100)}% Train / {int(test_split*100)}% Test")
    print(f"==================================================")

    # 1. Load MNIST Test Set in both modalities
    print("[1/4] Loading test datasets...")
    (_, _), (_, _), (X_test_flat, y_test) = load_mnist(flatten=True, normalize=True, test_split=test_split)
    (_, _), (_, _), (X_test_cnn, _) = load_mnist(flatten=False, normalize=True, test_split=test_split)
    print(f"      Evaluated on {len(y_test):,} MNIST test samples.")

    # Benchmark subset for latency calculation (1,000 samples for high-precision timing)
    latency_subset_size = 1000
    sub_flat = X_test_flat[:latency_subset_size]
    sub_cnn = X_test_cnn[:latency_subset_size]

    records = []

    # 2. Benchmark Classical Model
    classical_model_path = os.path.join(models_dir, "classical_model.joblib")
    classical_metrics_path = os.path.join(models_dir, "classical_metrics.json")
    if os.path.exists(classical_model_path):
        print("\n[2/4] Evaluating Classical Model...")
        classical_model = joblib.load(classical_model_path)
        y_pred = classical_model.predict(X_test_flat)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        lat_ms = benchmark_model_inference(classical_model.predict, sub_flat)
        size_mb = get_file_size_mb(classical_model_path)

        train_time = None
        if os.path.exists(classical_metrics_path):
            with open(classical_metrics_path, "r") as f:
                c_meta = json.load(f)
                train_time = c_meta.get("training_time_sec")

        records.append({
            "Model Name": "Classical ML (SVM/RF)",
            "Test Accuracy (%)": round(acc * 100, 2),
            "Macro F1": round(f1, 4),
            "Training Time (s)": round(train_time, 2) if train_time else "N/A",
            "Inference Latency (ms/sample)": round(lat_ms, 4),
            "Parameters": "N/A (Non-parametric/Trees)",
            "Disk Size (MB)": round(size_mb, 2),
        })
    else:
        print(f"[Notice] Classical model not found at {classical_model_path}. (Run train_classical.py first)")

    # 3. Benchmark MLP Model
    mlp_model_path = os.path.join(models_dir, "mlp_model.keras")
    mlp_metrics_path = os.path.join(models_dir, "mlp_metrics.json")
    if os.path.exists(mlp_model_path):
        print("\n[3/4] Evaluating MLP Model (Dense Only)...")
        mlp_model = tf.keras.models.load_model(mlp_model_path)
        y_prob = mlp_model.predict(X_test_flat, batch_size=256, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        lat_ms = benchmark_model_inference(
            lambda x: mlp_model.predict(x, batch_size=256, verbose=0), sub_flat
        )
        size_mb = get_file_size_mb(mlp_model_path)

        train_time = None
        if os.path.exists(mlp_metrics_path):
            with open(mlp_metrics_path, "r") as f:
                m_meta = json.load(f)
                train_time = m_meta.get("training_time_sec")

        records.append({
            "Model Name": "Multilayer Perceptron (MLP)",
            "Test Accuracy (%)": round(acc * 100, 2),
            "Macro F1": round(f1, 4),
            "Training Time (s)": round(train_time, 2) if train_time else "N/A",
            "Inference Latency (ms/sample)": round(lat_ms, 4),
            "Parameters": f"{mlp_model.count_params():,}",
            "Disk Size (MB)": round(size_mb, 2),
        })
    else:
        print(f"[Notice] MLP model not found at {mlp_model_path}. (Run train_mlp.py first)")

    # 4. Benchmark CNN Model
    cnn_model_path = os.path.join(models_dir, "cnn_model.keras")
    cnn_metrics_path = os.path.join(models_dir, "cnn_metrics.json")
    if os.path.exists(cnn_model_path):
        print("\n[4/4] Evaluating Introductory CNN Model...")
        cnn_model = tf.keras.models.load_model(cnn_model_path)
        y_prob = cnn_model.predict(X_test_cnn, batch_size=256, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        lat_ms = benchmark_model_inference(
            lambda x: cnn_model.predict(x, batch_size=256, verbose=0), sub_cnn
        )
        size_mb = get_file_size_mb(cnn_model_path)

        train_time = None
        if os.path.exists(cnn_metrics_path):
            with open(cnn_metrics_path, "r") as f:
                c_meta = json.load(f)
                train_time = c_meta.get("training_time_sec")

        records.append({
            "Model Name": "Convolutional Neural Net (CNN)",
            "Test Accuracy (%)": round(acc * 100, 2),
            "Macro F1": round(f1, 4),
            "Training Time (s)": round(train_time, 2) if train_time else "N/A",
            "Inference Latency (ms/sample)": round(lat_ms, 4),
            "Parameters": f"{cnn_model.count_params():,}",
            "Disk Size (MB)": round(size_mb, 2),
        })
    else:
        print(f"[Notice] CNN model not found at {cnn_model_path}. (Run train_cnn.py first)")

    if not records:
        print("\n[Warning] No trained models found in saved_models/. Please train at least one model.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print("\n" + "=" * 80)
    print("                     CROSS-MODEL COMPARISON MATRIX")
    print("=" * 80)
    print(df.to_markdown(index=False))
    print("=" * 80)

    # Save CSV
    csv_path = os.path.join(models_dir, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] Comparison table saved to: {csv_path}")

    # Generate visual comparison plot if multiple models exist
    if len(df) >= 2:
        generate_comparison_chart(df, output_path=os.path.join(models_dir, "model_comparison.png"))

    return df


def generate_comparison_chart(df: pd.DataFrame, output_path: str):
    """Generates comparison bar chart for Accuracy and Latency."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy plot
        axes[0].bar(df["Model Name"], df["Test Accuracy (%)"], color=["#4A90E2", "#50E3C2", "#B8E986"][:len(df)])
        axes[0].set_title("Test Accuracy Comparison (%)")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_ylim(min(df["Test Accuracy (%)"]) - 2, 100)
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)

        # Latency plot
        axes[1].bar(df["Model Name"], df["Inference Latency (ms/sample)"], color=["#F5A623", "#D0021B", "#9013FE"][:len(df)])
        axes[1].set_title("Inference Latency (ms / sample)")
        axes[1].set_ylabel("Latency (ms)")
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[Saved] Comparison plot saved to:  {output_path}")
    except Exception as e:
        print(f"[Warning] Could not generate comparison plot: {e}")


if __name__ == "__main__":
    evaluate_all_models()
