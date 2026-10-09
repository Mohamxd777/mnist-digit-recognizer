# 👥 Team Task Distribution & Git Collaboration Guide

**Assignment 1:** Training Models on the MNIST Dataset + Interactive Drawing App  
**Team Composition:** 4 Students  
**Branch Strategy:** Isolated Feature Branches (Zero Merge Conflicts)

---

## 🎯 Executive Overview & Architecture Boundary

To prevent merge conflicts, each student owns dedicated files and functions within the codebase. The table below outlines ownership boundaries:

| Member & GitHub Handle | Assigned Git Branch | Primary Code Ownership | Key Deliverables & Report Ownership |
| :--- | :--- | :--- | :--- |
| **@abuelmagd2oo5-svg** (ML) | `student-1-data-classical` | `src/data_loader.py`<br>`src/train_classical.py` | Deliverable 1 (Data Preprocessing pipeline) + Classical ML modeling (SVM / Random Forest) + Deliverable 3 Section 1 & 2.1 |
| **@M0L0TFY** (NN) | `student-2-mlp` | `src/train_mlp.py` | Multilayer Perceptron (MLP) Architecture (Dense layers only) + Systematic Hyperparameter Exploration + Deliverable 3 Section 2.2 |
| **@Mohamxd777** (DL) | `student-3-cnn-eval` | `src/train_cnn.py`<br>`src/evaluate_models.py` | Simple Introductory CNN + Comparative Evaluation Framework (Accuracy, F1, Latency, Complexity) + Deliverable 3 Section 2.3 & 3 |
| **@saraasawy** (Frontend & Integration) | `student-4-streamlit-ui` | `src/image_utils.py`<br>`app.py` | Canvas Image Preprocessing (bounding box, centering, 28x28) + Streamlit Web UI + Joining Front and Back together + Deliverable 4 & 5 |

---

## 📌 Student-by-Student Detailed Breakdown

### 🧑‍💻 @abuelmagd2oo5-svg: Data Preprocessing & Classical ML Specialist (ML)
- **Assigned Branch:** `student-1-data-classical`
- **Files Owned:**
  - `src/data_loader.py`
  - `src/train_classical.py`
- **Key Tasks:**
  1. **Data Loading & Preprocessing:**
     - Implement `load_mnist()` supporting both flattened representations `(N, 784)` and 2D spatial channel representations `(N, 28, 28, 1)`.
     - Normalize all pixel values from $[0, 255] \rightarrow [0.0, 1.0]$ float32.
     - Partition training data into stratified train/validation sets (e.g., 90% train, 10% val).
  2. **Classical Machine Learning:**
     - Train Scikit-Learn classifiers (Support Vector Classifier `SVC(probability=True)` and/or `RandomForestClassifier`).
     - Conduct hyperparameter search using `GridSearchCV` (e.g., test $C \in [0.1, 1, 10]$, $\gamma \in ['scale', 0.01]$, or tree depths).
     - Save the best trained model to `saved_models/classical_model.joblib`.
     - Export metrics (accuracy, macro F1, confusion matrix, training time) to `saved_models/classical_metrics.json`.
  3. **Report Deliverable Sections:**
     - Fill in Section 1 (Data Preprocessing & Normalization) in `REPORT_TEMPLATE.md`.
     - Fill in Section 2.1 (Classical Model Architecture & Hyperparameter Tuning) in `REPORT_TEMPLATE.md`.

---

### 🧑‍💻 @M0L0TFY: Multilayer Perceptron (MLP) Specialist (NN)
- **Assigned Branch:** `student-2-mlp`
- **Files Owned:**
  - `src/train_mlp.py`
- **Key Tasks:**
  1. **MLP Architecture Design:**
     - Build a feedforward neural network using TensorFlow/Keras with **strictly Dense layers** (NO convolutional or pooling layers).
     - Incorporate Batch Normalization, Dropout (e.g., rate 0.2–0.4), and activation functions (ReLU / LeakyReLU).
  2. **Systematic Hyperparameter Search:**
     - Systematically evaluate variations in:
       - Network depth & layer capacities (e.g., `[256, 128]`, `[512, 256]`, `[512, 256, 128]`).
       - Dropout rates (`0.0`, `0.2`, `0.4`).
       - Learning rates (`1e-2`, `1e-3`, `1e-4`) and optimizers (`Adam`, `RMSprop`).
     - Utilize `EarlyStopping` and `ReduceLROnPlateau` callbacks.
  3. **Model Artifacts & Metrics:**
     - Save the best MLP model to `saved_models/mlp_model.keras`.
     - Export training history curves and test metrics to `saved_models/mlp_metrics.json`.
  4. **Report Deliverable Sections:**
     - Fill in Section 2.2 (MLP Architecture, Tuning Grid, and Overfitting Prevention) in `REPORT_TEMPLATE.md`.

---

### 🧑‍💻 @Mohamxd777: CNN & Evaluation Framework Specialist (DL)
- **Assigned Branch:** `student-3-cnn-eval`
- **Files Owned:**
  - `src/train_cnn.py`
  - `src/evaluate_models.py`
- **Key Tasks:**
  1. **Introductory Convolutional Neural Network:**
     - Design a clean, introductory CNN architecture (`Conv2D` + `MaxPooling2D` + `Dense`).
     - **Constraint:** Strictly NO pre-trained weights or excessively deep models (ResNet, VGG, etc.). Keep it introductory.
     - Tune filter counts (32, 64), kernel sizes (3x3), dense units, and dropout.
     - Save trained model to `saved_models/cnn_model.keras` and metrics to `saved_models/cnn_metrics.json`.
  2. **Cross-Model Benchmark Suite (`src/evaluate_models.py`):**
     - Load all three saved models (`classical_model.joblib`, `mlp_model.keras`, `cnn_model.keras`).
     - Calculate uniform test metrics on all 10,000 MNIST test samples:
       - Test Accuracy (%)
       - Macro F1-Score
       - Training Time (seconds)
       - Average Inference Latency (ms per sample over 1,000 iterations)
       - Parameter Count and Disk Size (MB)
     - Generate a Markdown comparison matrix and export to `saved_models/comparison_table.csv` & `saved_models/model_comparison.png`.
  3. **Report Deliverable Sections:**
     - Fill in Section 2.3 (CNN Architecture & Hyperparameter Tuning) in `REPORT_TEMPLATE.md`.
     - Fill in Section 3 (Comprehensive Cross-Model Comparison & Trade-off Analysis) and Section 4 (Best Model Justification) in `REPORT_TEMPLATE.md`.

---

### 🧑‍💻 @saraasawy: Frontend, UI & Full-Stack Integration Specialist
- **Assigned Branch:** `student-4-streamlit-ui`
- **Files Owned:**
  - `src/image_utils.py`
  - `app.py`
- **Key Tasks:**
  1. **Canvas Drawing Preprocessor (`src/image_utils.py`):**
     - Transform raw RGBA arrays from `st_canvas` into standard MNIST inputs:
       - Convert RGBA to single-channel grayscale with black background and white strokes.
       - Crop bounding box around drawn strokes.
       - Scale bounding box to fit inside a 20x20 pixel area (preserving aspect ratio).
       - Center the 20x20 image inside a 28x28 bounding frame (using center-of-mass shift).
       - Normalize pixel intensities to $[0.0, 1.0]$.
       - Return formatted tensors for both `(1, 784)` and `(1, 28, 28, 1)`.
  2. **Streamlit Web Application & Full-Stack Integration (`app.py`):**
     - Join frontend canvas drawing with backend model inference.
     - Interactive drawing board using `streamlit-drawable-canvas` (`st_canvas`).
     - Controls: Brush stroke width slider, "Clear Canvas" button.
     - Dropdown selector to choose between the 3 models (Classical ML, MLP, CNN).
     - Graceful error handling: Notify the user with exact commands if a model file is missing.
     - "Predict Digit" button showing predicted digit, confidence percentage, 28x28 preprocessed preview, and full class probability bar chart (0-9).
  3. **Deliverables & Report Sections:**
     - Prepare Streamlit interactive demo, screenshots, or screencast for Deliverable 5.
     - Fill in Section 1.2 (Interactive Preprocessing & Canvas Alignment) in `REPORT_TEMPLATE.md`.

---

## 🌿 Git Collaborative Workflow (Zero Merge Conflicts)

Because each student modifies distinct files, merging back into `main` will be clean and conflict-free when following these rules:

### Step 1: Switch to Your Isolated Feature Branch
```bash
# Student 1:
git checkout student-1-data-classical

# Student 2:
git checkout student-2-mlp

# Student 3:
git checkout student-3-cnn-eval

# Student 4:
git checkout student-4-streamlit-ui
```

### Step 2: Work, Stage, and Commit Locally
Make small, frequent commits within your owned files:
```bash
git add src/train_classical.py
git commit -m "feat(classical): complete GridSearchCV tuning for SVM"
```

### Step 3: Keep Branch Updated with `main`
Before opening a pull request or merging, sync any shared updates from `main`:
```bash
git fetch origin
git rebase main   # Or: git merge main
```

### Step 4: Merge Order & Integration Plan
1. **Milestone 1:** Student 1 finishes `src/data_loader.py` and merges into `main`. All other students pull `main` to access `load_mnist()`.
2. **Milestone 2:** Students 1, 2, and 3 finish their respective training scripts (`train_classical.py`, `train_mlp.py`, `train_cnn.py`) and train baseline models into `saved_models/`.
3. **Milestone 3:** Student 3 runs `evaluate_models.py` once all 3 models are saved in `saved_models/`.
4. **Milestone 4:** Student 4 verifies real-time inference on `app.py` against all three models.
5. **Milestone 5:** Final review and joint compilation of `REPORT_TEMPLATE.md`.

---

## 📋 Assignment Deliverables Checklist

- [ ] **Deliverable 1:** Data loader with normalization and dual tensor format (`data_loader.py`).
- [ ] **Deliverable 2:** 3 trained models saved in `saved_models/`:
  - `classical_model.joblib`
  - `mlp_model.keras`
  - `cnn_model.keras`
- [ ] **Deliverable 3:** Complete written report following `REPORT_TEMPLATE.md` with comparison table and charts.
- [ ] **Deliverable 4:** Image preprocessing pipeline matching MNIST center-of-mass standard (`image_utils.py`).
- [ ] **Deliverable 5:** Interactive Streamlit web app (`app.py`).
