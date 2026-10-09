# ✏️ MNIST Digit Recognizer & Interactive Drawing App

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning and Deep Learning system for handwritten digit recognition on the MNIST benchmark dataset. Features three distinct model paradigms (Classical ML, Multilayer Perceptron, and Convolutional Neural Networks) and an interactive web application built with Streamlit and `streamlit-drawable-canvas`.

Developed as **Assignment 1** by a team of 4 students.

---

## 📂 Repository Structure

```text
mnist-digit-recognizer/
├── .gitignore                 # Excludes venv, cache, OS files, and datasets
├── requirements.txt           # Python library dependencies
├── README.md                  # Comprehensive setup, training, and execution guide
├── TEAM_TASKS.md              # 4-student role distribution, ownership boundaries & Git workflow
├── REPORT_TEMPLATE.md         # Technical report scaffold for Deliverable 3
├── saved_models/              # Directory housing saved model weights and metrics JSONs
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # [Student 1] Dual-modality MNIST loader (N, 784) and (N, 28, 28, 1)
│   ├── train_classical.py     # [Student 1] Classical ML (SVM/RF) + GridSearchCV tuning
│   ├── train_mlp.py           # [Student 2] Multilayer Perceptron (Dense-only) + hyperparameter search
│   ├── train_cnn.py           # [Student 3] Introductory CNN (Conv2D + MaxPool + Dense) + tuning
│   ├── evaluate_models.py     # [Student 3] Uniform cross-model benchmarking (Acc, F1, Latency, Params)
│   └── image_utils.py         # [Student 4] Canvas RGBA -> Crop -> 20x20 scale -> 28x28 Center-of-Mass
└── app.py                     # [Student 4] Interactive Streamlit drawing app & live probability charts
```

---

## ⚡ Quick Start Guide

### 1. Prerequisites & Environment Setup
Clone the repository and configure a dedicated Python virtual environment:

```bash
# Clone the repository
git clone https://github.com/your-org/mnist-digit-recognizer.git
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

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Model Training Workflow

Each model can be trained independently using its modular training script. Trained weights and performance metrics are automatically saved into `saved_models/`.

### 1. Train Classical ML (Student 1)
Trains a Support Vector Classifier (`SVC`) or Random Forest with hyperparameter grid search:
```bash
# Train SVM (default, subsample for fast tuning):
python src/train_classical.py --model-type svm --sample-size 10000 --cv-folds 3

# Or train Random Forest:
python src/train_classical.py --model-type rf --sample-size 20000
```
- **Output:** Saves `saved_models/classical_model.joblib` and `saved_models/classical_metrics.json`.

### 2. Train Multilayer Perceptron (MLP) (Student 2)
Trains a feedforward deep neural network using **Dense layers only**:
```bash
python src/train_mlp.py --hidden-units 512 256 --dropout 0.2 --epochs 20 --batch-size 128 --lr 0.001
```
- **Output:** Saves `saved_models/mlp_model.keras` and `saved_models/mlp_metrics.json`.

### 3. Train Introductory CNN (Student 3)
Trains an introductory Convolutional Neural Network (`Conv2D` + `MaxPooling2D` + `Dense`):
```bash
python src/train_cnn.py --filters-1 32 --filters-2 64 --dense-units 128 --dropout 0.25 --epochs 15
```
- **Output:** Saves `saved_models/cnn_model.keras` and `saved_models/cnn_metrics.json`.

---

## 📊 Cross-Model Evaluation & Benchmarks (Student 3)

After training the models, execute the evaluation suite to calculate uniform comparative metrics on the 10,000-image MNIST test set:

```bash
python src/evaluate_models.py
```

This generates:
1. Console Markdown table displaying:
   - Test Accuracy (%)
   - Macro F1-Score
   - Training Time (seconds)
   - Average Inference Latency (ms / sample)
   - Model Complexity (Parameter count & Disk size)
2. `saved_models/comparison_table.csv`
3. `saved_models/model_comparison.png` (Comparative bar charts)

---

## 🎨 Interactive Drawing Web App (Deliverable 5 - Student 4)

Launch the interactive web application to draw handwritten digits and perform real-time model inference:

```bash
streamlit run app.py
```

### Features:
- **Interactive Canvas:** Draw freehand digits with customizable brush width.
- **Model Selector:** Instantly toggle between Classical ML, MLP, and CNN.
- **MNIST Preprocessing Visualization:** View the 28x28 normalized, center-of-mass aligned image the model receives.
- **Confidence Metrics & Probability Charts:** Bar chart visualizing probability distribution across digits $0$ through $9$.
- **Clear Canvas:** One-click reset for rapid testing.

---

## 🌿 Team Collaboration & Git Feature Branches

To ensure zero merge conflicts, 4 isolated branches were created off `main` for each team member:
1. `abuelmagd2oo5-svg-ml` (@abuelmagd2oo5-svg: `data_loader.py`, `train_classical.py`)
2. `M0L0TFY-nn` (@M0L0TFY: `train_mlp.py`)
3. `Mohamxd777-dl` (@Mohamxd777: `train_cnn.py`, `evaluate_models.py`)
4. `saraasawy-frontend` (@saraasawy: `image_utils.py`, `app.py`)

For branch guidelines, pull request protocols, and report distribution, consult [TEAM_TASKS.md](TEAM_TASKS.md).  
For the report deliverable structure, refer to [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md).
