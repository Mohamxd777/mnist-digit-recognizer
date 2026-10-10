"""
data_loader.py
==============
Author: Student 1 (Data & Classical ML Specialist)
Assignment 1: Training Models on MNIST + Interactive Drawing App

Responsibilities:
- Load the standard MNIST benchmark dataset (60,000 train, 10,000 test).
- Normalize pixel intensities from [0, 255] integer values to [0.0, 1.0] float32.
- Support stratified train-validation splitting.
- Provide two tensor shape representations:
    1. Flattened 1D feature vectors (N, 784) for Classical ML (SVM/RF) and MLP.
    2. 2D spatial channel tensors (N, 28, 28, 1) for Convolutional Neural Networks (CNN).
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split


def load_mnist(
    flatten: bool = False,
    normalize: bool = True,
    val_split: float = 0.1,
    test_split: Optional[float] = None,
    random_state: int = 42,
    subset_size: Optional[int] = None,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Loads, preprocesses, and partitions the MNIST dataset.

    Parameters
    ----------
    flatten : bool, default=False
        If True, returns images reshaped as (N, 784).
        If False, returns images shaped as (N, 28, 28, 1).
    normalize : bool, default=True
        If True, scales pixel values from [0, 255] to [0.0, 1.0] as float32.
    val_split : float, default=0.1
        Fraction of the training set to reserve for validation (e.g., 0.1 = 10%).
    test_split : Optional[float], default=None
        If specified (e.g. 0.3 for 70/30 split), combines all 70,000 MNIST images
        and splits into (1.0 - test_split) train pool and test_split test set.
    random_state : int, default=42
        Random seed for reproducible train-validation splitting.
    subset_size : Optional[int], default=None
        If specified, trims the training set to this size for rapid prototyping
        and quick hyperparameter tuning runs.

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
        Tuple of training, validation, and test subsets with respective features and labels.
    """
    # Attempt loading via TensorFlow/Keras datasets
    try:
        from tensorflow.keras.datasets import mnist

        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()
    except ImportError:
        # Fallback to OpenML if TensorFlow is unavailable
        from sklearn.datasets import fetch_openml

        print("[DataLoader] TensorFlow not found. Fetching MNIST via OpenML...")
        mnist_openml = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist_openml.data.reshape(-1, 28, 28).astype(np.uint8)
        y = mnist_openml.target.astype(np.uint8)
        x_train_raw, x_test_raw = X[:60000], X[60000:]
        y_train_raw, y_test_raw = y[:60000], y[60000:]

    # Custom train/test split if requested (e.g. 70/30 split across all 70k samples)
    if test_split is not None:
        X_all = np.concatenate([x_train_raw, x_test_raw], axis=0)
        y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)
        x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_all,
            y_all,
            test_size=test_split,
            random_state=random_state,
            stratify=y_all,
        )

    # Optional subset truncation for fast hyperparameter grid sweeps
    if subset_size is not None and subset_size < len(x_train_raw):
        indices = np.random.RandomState(random_state).permutation(len(x_train_raw))[:subset_size]
        x_train_raw = x_train_raw[indices]
        y_train_raw = y_train_raw[indices]

    # Convert data type to float32
    X_train_full = x_train_raw.astype(np.float32)
    X_test = x_test_raw.astype(np.float32)
    y_train_full = y_train_raw.astype(np.int64)
    y_test = y_test_raw.astype(np.int64)

    # Normalization: scale pixel intensities from [0, 255] to [0.0, 1.0]
    if normalize:
        X_train_full = X_train_full / 255.0
        X_test = X_test / 255.0

    # Reshaping based on model modality
    if flatten:
        # (N, 784) for Classical ML models & MLP Dense inputs
        X_train_full = X_train_full.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)
    else:
        # (N, 28, 28, 1) for CNN Conv2D channel inputs
        X_train_full = np.expand_dims(X_train_full, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

    # Train / Validation Stratified Split
    if val_split > 0.0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=val_split,
            random_state=random_state,
            stratify=y_train_full,
        )
    else:
        X_train, y_train = X_train_full, y_train_full
        X_val, y_val = np.empty((0, *X_train.shape[1:])), np.empty((0,), dtype=np.int64)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    print("=== Testing data_loader.py (Student 1) ===")
    
    # 1. Test Flattened Mode (Classical / MLP)
    (X_tr_flat, y_tr_flat), (X_va_flat, y_va_flat), (X_te_flat, y_te_flat) = load_mnist(flatten=True)
    print(f"[Flattened Mode]")
    print(f"  Train: X={X_tr_flat.shape}, y={y_tr_flat.shape}")
    print(f"  Val:   X={X_va_flat.shape}, y={y_va_flat.shape}")
    print(f"  Test:  X={X_te_flat.shape}, y={y_te_flat.shape}")
    print(f"  Pixel Range: [{X_tr_flat.min():.2f}, {X_tr_flat.max():.2f}]")

    # 2. Test Image Channel Mode (CNN)
    (X_tr_cnn, y_tr_cnn), (X_va_cnn, y_va_cnn), (X_te_cnn, y_te_cnn) = load_mnist(flatten=False)
    print(f"\n[CNN Channel Mode]")
    print(f"  Train: X={X_tr_cnn.shape}, y={y_tr_cnn.shape}")
    print(f"  Val:   X={X_va_cnn.shape}, y={y_va_cnn.shape}")
    print(f"  Test:  X={X_te_cnn.shape}, y={y_te_cnn.shape}")
    print(f"  Pixel Range: [{X_tr_cnn.min():.2f}, {X_tr_cnn.max():.2f}]")
    print("\nDataLoader self-check completed successfully!")
