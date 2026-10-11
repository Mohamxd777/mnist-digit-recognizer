<div align="center">

# ✏️ MNIST Digit Recognizer & Interactive Drawing App

[![Live App](https://img.shields.io/badge/Streamlit_App-Live_Online-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://mnistdigitrecognizerr.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Mohamxd777%2Fmnist--digit--recognizer-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Mohamxd777/mnist-digit-recognizer)

<br/>

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3.x-D00000.svg?logo=keras&logoColor=white)](https://keras.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e.svg)](LICENSE)

<p align="center">
  <b>An end-to-end Machine Learning and Deep Learning system for handwritten digit recognition on the MNIST benchmark dataset.</b><br/>
  Features three distinct model paradigms (Classical ML, Multilayer Perceptron, and Convolutional Neural Networks) evaluated under identical conditions and served via a modern, interactive Streamlit web application.
</p>

[**🚀 Open Live Web App**](https://mnistdigitrecognizerr.streamlit.app/) • [**📊 View Benchmarks**](#-cross-model-benchmark-matrix) • [**⚡ Quick Start**](#-quick-start-guide) • [**👥 Team Tasks**](#-team-contributions--ownership)

---

</div>

## 🌐 Live Web Application

The interactive web application is deployed and live on **Streamlit Community Cloud**:

👉 **[https://mnistdigitrecognizerr.streamlit.app/](https://mnistdigitrecognizerr.streamlit.app/)**

### Key App Capabilities:
* **Interactive Freehand Canvas:** Draw any digit ($0$–$9$) directly in your browser with customizable brush stroke thickness.
* **Simultaneous 3-Model Inference:** A single click on **✦ Predict** queries the **CNN**, **MLP**, and **Classical SVM** models concurrently on the exact same drawing.
* **Model Consensus Detection:** Automatically calculates and displays ensemble agreement across all models (*Full Consensus*, *Majority Agreement*, or *Split Decision*).
* **Deep Probability Distribution:** Dynamic interactive bar chart showing predicted probability percentages across all classes $0$ through $9$.
* **Canonical Preprocessing Inspector:** Real-time expander revealing the centered, normalized $28 \times 28$ image matrix and bounding box metadata sent to the models.
* **Zero-Rerun State Caching:** Switch the highlighted model in the dropdown to instantly view its confidence breakdown without redrawing or recomputing predictions.

---

## 📊 Cross-Model Benchmark Matrix

All models were evaluated using the uniform benchmarking suite ([`src/evaluate_models.py`](src/evaluate_models.py)) across **21,000 real MNIST test samples** (30% stratified test split from 70,000 images):

| Model Paradigm | Architecture / Algorithm | Test Accuracy | Macro F1 | Inference Latency | Trainable Parameters | Disk Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Classical ML** | Support Vector Classifier (`SVC`, RBF Kernel, $C=10.0$) | **97.02%** | **0.9700** | 2.4627 ms/sample | *Non-parametric* | 21.33 MB |
| **Multilayer Perceptron (MLP)** | Deep Feedforward (`Dense`: 512 → 256 → 128 → 10) + BatchNorm + Dropout | **99.59%** | **0.9959** | **0.1188 ms/sample** | 571,018 | 6.57 MB |
| **Convolutional Neural Net (CNN)** | Introductory 2D CNN (`Conv2D(32)` → `MaxPool` → `Conv2D(64)` → `MaxPool` → `Dense(128)`) | **99.09%** | **0.9909** | 0.1587 ms/sample | **421,642** | **4.86 MB** |

### Key Trade-Off Insights:
1. **Accuracy & Generalization:** The **MLP** achieved the highest test accuracy (**99.59%**) followed closely by the **CNN** (**99.09%**). Both deep learning models decisively out-performed the classical RBF SVM baseline (**97.02%**).
2. **Inference Latency:** Both neural network architectures demonstrated rapid throughput (~**0.12–0.16 ms per sample** on CPU), making them over **15× to 20× faster** at inference time than the Support Vector Classifier (**2.46 ms per sample**), whose kernel evaluations scale with the number of support vectors.
3. **Storage Footprint:** The **CNN** achieved the most compact model artifact (**4.86 MB**), requiring fewer parameters than the dense MLP while capturing 2D spatial features and translation invariance.

---

## 🧠 Architectural Overview

```text
                                        ┌────────────────────────┐
                                        │  User Canvas Drawing   │
                                        └───────────┬────────────┘
                                                    │
                                                    ▼
                                    ┌────────────────────────────────┐
                                    │      src/image_utils.py        │
                                    │  1. Extract Grayscale Intensity│
                                    │  2. Crop Bounding Box          │
                                    │  3. Aspect-Preserve to 20x20   │
                                    │  4. Center-of-Mass into 28x28  │
                                    │  5. Normalize to [0.0, 1.0]    │
                                    └───────┬────────────────┬───────┘
                                            │                │
                        Flattened (1, 784)  │                │ Spatial (1, 28, 28, 1)
                                            ▼                ▼
                     ┌──────────────────────┴────────┐  ┌─────────────────────────┐
                     │                               │  │       src/train_cnn.py  │
                     ▼                               ▼  │  Conv2D (32, 3x3)       │
           ┌─────────────────────┐       ┌─────────────────────┐│  MaxPooling2D (2x2)     │
           │src/train_classical.py│      │  src/train_mlp.py   ││  Conv2D (64, 3x3)       │
           │  Scikit-Learn SVC   │       │  Dense (512) + BN   ││  MaxPooling2D (2x2)     │
           │  RBF Kernel, C=10.0 │       │  Dense (256) + BN   ││  Flatten + Dense(128)   │
           │  GridSearchCV Tuned │       │  Dense (128) + BN   ││  Dense(10, Softmax)     │
           │  Probabilities      │       │  Dense(10, Softmax) │└───────────┬─────────────┘
           └──────────┬──────────┘       └──────────┬──────────┘            │
                      │                             │                       │
                      └──────────────────────┬──────┴───────────────────────┘
                                             │
                                             ▼
                                ┌─────────────────────────┐
                                │   app.py (Streamlit UI) │
                                │  • Top Predicted Digit  │
                                │  • Model Consensus Pill │
                                │  • 10-Class Probability │
                                │  • Multi-Model Compare  │
                                └─────────────────────────┘
```

---

## 📂 Repository Structure

```text
mnist-digit-recognizer/
├── .gitignore                          # Excludes caches, venv, checkpoints, and temporary logs
├── requirements.txt                    # Verified Python library dependencies
├── README.md                           # Comprehensive documentation & setup instructions
├── TEAM_TASKS.md                       # Role distributions, code ownership & workflow protocols
├── REPORT_TEMPLATE.md                  # Academic report scaffold for Deliverable 3
├── app.py                              # Interactive Streamlit drawing web application
├── saved_models/                       # Trained model weights, evaluation metrics, and plots
│   ├── classical_model.joblib          # Trained Scikit-Learn SVC artifact (21.3 MB)
│   ├── classical_metrics.json          # Classical ML training and evaluation metadata
│   ├── mlp_model.keras                 # Trained MLP Keras model artifact (6.6 MB)
│   ├── mlp_metrics.json                # MLP training curves and test evaluation metadata
│   ├── cnn_model.keras                 # Trained CNN Keras model artifact (4.9 MB)
│   ├── cnn_metrics.json                # CNN training and evaluation metadata
│   ├── comparison_table.csv            # Uniform comparative metrics table across models
│   └── model_comparison.png            # Visual bar charts comparing accuracy and latency
└── src/
    ├── __init__.py                     # Modular package initialization
    ├── data_loader.py                  # Dual-modality MNIST loader (N, 784) and (N, 28, 28, 1)
    ├── image_utils.py                  # RGBA canvas stroke extraction and MNIST canonical centering
    ├── train_classical.py              # Classical ML modeling (SVM / Random Forest) + GridSearchCV
    ├── train_mlp.py                    # Dense-only Multilayer Perceptron with hyperparameter tuning
    ├── train_cnn.py                    # Introductory 2D CNN with regularization and callbacks
    └── evaluate_models.py              # Cross-model uniform benchmarking framework
```

---

## ⚡ Quick Start Guide

### 1. Prerequisites & Environment Setup

Clone the repository and initialize a dedicated Python virtual environment (Python 3.9 – 3.13):

```bash
# Clone the repository
git clone https://github.com/Mohamxd777/mnist-digit-recognizer.git
cd mnist-digit-recognizer

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Windows Command Prompt:
.\venv\Scripts\activate.bat
# On macOS / Linux:
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Launch the Interactive App Locally

To start the Streamlit web application on your local machine:

```bash
python -m streamlit run app.py
```

The application will start immediately and can be accessed at:
👉 **`http://localhost:8501`**

---

### 3. Training the Models Independently

Each model includes a self-contained, modular training script with configurable CLI arguments:

#### A. Train Classical Machine Learning (Student 1)
```bash
# Train Support Vector Classifier (RBF Kernel with GridSearchCV):
python src/train_classical.py --model-type svm --sample-size 10000 --cv-folds 3

# Or train Random Forest:
python src/train_classical.py --model-type rf --sample-size 20000
```
*Outputs: `saved_models/classical_model.joblib` and `saved_models/classical_metrics.json`.*

#### B. Train Multilayer Perceptron (Student 2)
```bash
# Train feedforward Dense-only neural network:
python src/train_mlp.py --hidden-units 512 256 128 --dropout 0.4 --epochs 25 --batch-size 128 --lr 0.0005
```
*Outputs: `saved_models/mlp_model.keras` and `saved_models/mlp_metrics.json`.*

#### C. Train Convolutional Neural Network (Student 3)
```bash
# Train introductory 2D CNN:
python src/train_cnn.py --filters-1 32 --filters-2 64 --dense-units 128 --dropout 0.25 --epochs 15
```
*Outputs: `saved_models/cnn_model.keras` and `saved_models/cnn_metrics.json`.*

---

### 4. Run the Cross-Model Benchmark Suite

Execute the uniform evaluation script to benchmark all 3 saved models against the MNIST test set:

```bash
python src/evaluate_models.py
```

This outputs a Markdown comparison matrix directly to the console, regenerates [`saved_models/comparison_table.csv`](saved_models/comparison_table.csv), and saves the visualization chart to [`saved_models/model_comparison.png`](saved_models/model_comparison.png).

---

## 👥 Team Contributions & Ownership

Developed as **Assignment 1** by a team of 4 students with modular file ownership:

| Student | GitHub Handle | Primary Code Ownership | Responsibilities |
| :--- | :--- | :--- | :--- |
| **Student 1** | [@abuelmagd2oo5-svg](https://github.com/abuelmagd2oo5-svg) | `src/data_loader.py`<br/>`src/train_classical.py` | Data normalization, stratified splitting, Classical ML modeling (SVM / RF), hyperparameter optimization via `GridSearchCV`. |
| **Student 2** | [@M0L0TFY](https://github.com/M0L0TFY) | `src/train_mlp.py` | Deep Multilayer Perceptron architecture (strictly Dense layers), Batch Normalization, Dropout regularization, learning rate scheduling. |
| **Student 3** | [@Mohamxd777](https://github.com/Mohamxd777) | `src/train_cnn.py`<br/>`src/evaluate_models.py` | Introductory 2D CNN design, uniform cross-model evaluation framework (Accuracy, F1, Latency, Complexity), benchmark matrix generation. |
| **Student 4** | [@saraasawy](https://github.com/saraasawy) | `src/image_utils.py`<br/>`app.py` | Canvas stroke extraction, aspect-preserving 20×20 bounding box centering, Streamlit UI/UX design, simultaneous multi-model integration. |

For detailed branch guidelines and task breakdowns, refer to [TEAM_TASKS.md](TEAM_TASKS.md).  
For the academic report scaffold, refer to [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — see the LICENSE file for details.
