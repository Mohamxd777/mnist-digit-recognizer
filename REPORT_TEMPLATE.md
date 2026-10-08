# 📄 Assignment 1 Technical Report: MNIST Digit Recognizer & Interactive Drawing App

**Course:** Machine Learning & Deep Learning  
**Assignment:** Assignment 1 — Training Models on MNIST + Interactive Drawing App  
**Team Members:**  
- Student 1: `[Name / Student ID]` — Data Loading & Classical Machine Learning  
- Student 2: `[Name / Student ID]` — Multilayer Perceptron (MLP) Modeling  
- Student 3: `[Name / Student ID]` — Convolutional Neural Network (CNN) & Model Evaluation  
- Student 4: `[Name / Student ID]` — Image Preprocessing & Streamlit UI Application  
**Date:** `[Submission Date]`  

---

## 1. Data Preprocessing & Input Pipelines

### 1.1 MNIST Dataset Characteristics
- **Total Samples:** 70,000 images (60,000 train / validation, 10,000 test).
- **Resolution & Classes:** $28 \times 28$ grayscale pixels; 10 balanced digit classes ($0-9$).
- **Stratified Split:** 90% training (54,000 samples) and 10% validation (6,000 samples).

### 1.2 Normalization and Tensor Formatting
Explain how pixel values were preprocessed for each model family:
- **Scaling:** Scaled raw unsigned integers $[0, 255]$ into floating point $[0.0, 1.0]$ via $X_{\text{norm}} = X / 255.0$.
- **Classical ML & MLP Format:** Flattened 2D images into 1D vectors with shape $(N, 784)$.
- **CNN Format:** Retained 2D spatial dimensions with an explicit channel dimension: shape $(N, 28, 28, 1)$.

### 1.3 Canvas Drawing Preprocessing Pipeline (Deliverable 4)
Discuss how the interactive drawing canvas in Streamlit transforms freehand user drawings to match the distribution of the MNIST dataset:
1. **Grayscale Extraction:** Convert raw RGBA canvas buffer into a single intensity channel.
2. **Bounding Box Cropping:** Detect non-zero strokes to remove arbitrary spatial translation offsets.
3. **Aspect-Ratio Preserving Scaling:** Scale the cropped digit so its largest dimension spans exactly 20 pixels.
4. **Padding & Center-of-Mass Alignment:** Embed the $20 \times 20$ digit into a $28 \times 28$ blank frame and apply sub-pixel shifts using center of mass $(\bar{x}, \bar{y})$ alignment, replicating the original NIST/LeCun preprocessing protocol.

---

## 2. Model Architectures & Hyperparameter Tuning

### 2.1 Classical Machine Learning (Student 1)
- **Algorithm Selected:** Support Vector Classifier (`SVC(kernel='rbf')`) / `RandomForestClassifier`.
- **Hyperparameter Search Configuration (`GridSearchCV`):**
  - Parameter Grid Tested:
    - $C \in [0.1, 1.0, 10.0]$
    - $\gamma \in ['\text{scale}', 0.01, 0.001]$
    - Cross-validation: 3-fold stratified cross-validation.
- **Optimal Hyperparameters Found:**
  - `C`: `[Insert best C]`
  - `gamma`: `[Insert best gamma]`
  - Best CV Accuracy: `[Insert CV score %]`
- **Observations:** Discuss how training time scales with dataset size for non-linear SVMs and how the decision boundary performs on ambiguous digits.

---

### 2.2 Multilayer Perceptron (MLP) (Student 2)
- **Constraint Compliance:** Dense layers ONLY. No convolutional or pooling layers.
- **Model Topology:**
  - Input Layer: 784 features.
  - Hidden Layer 1: `Dense(512, activation='relu')` + `BatchNormalization()` + `Dropout(0.2)`
  - Hidden Layer 2: `Dense(256, activation='relu')` + `BatchNormalization()` + `Dropout(0.2)`
  - Output Layer: `Dense(10, activation='softmax')`
- **Hyperparameter Grid Sweeps:**
  
| Configuration | Hidden Layers | Dropout Rate | Learning Rate | Val Accuracy (%) | Test Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Config A** | [256, 128] | 0.0 | 0.001 | `XX.X%` | `XX.X%` |
| **Config B** | [512, 256] | 0.2 | 0.001 | `XX.X%` | `XX.X%` |
| **Config C** | [512, 256, 128]| 0.4 | 0.0005 | `XX.X%` | `XX.X%` |

- **Training Dynamics:**
  - Loss Function: `sparse_categorical_crossentropy`
  - Optimizer: `Adam`
  - Early stopping convergence epoch: `[Insert epoch]`

---

### 2.3 Introductory Convolutional Neural Network (CNN) (Student 3)
- **Constraint Compliance:** Introductory architecture (Conv2D + MaxPool2D + Dense). No transfer learning or deep pre-trained nets (ResNet/VGG).
- **Model Architecture:**
  - Layer 1: `Conv2D(32, (3, 3), activation='relu', padding='same')` $\rightarrow$ `MaxPooling2D((2, 2))`
  - Layer 2: `Conv2D(64, (3, 3), activation='relu', padding='same')` $\rightarrow$ `MaxPooling2D((2, 2))`
  - Layer 3: `Flatten()` $\rightarrow$ `Dense(128, activation='relu')` $\rightarrow$ `Dropout(0.25)`
  - Layer 4: `Dense(10, activation='softmax')`
- **Hyperparameter Tuning:**
  - Kernel sizes evaluated ($3 \times 3$ vs $5 \times 5$).
  - Filter capacities (16/32 vs 32/64).
  - Regularization via Dropout (0.25).
- **Observations:** Analyze spatial feature extraction, receptive field growth, and translation invariance compared to the MLP.

---

## 3. Quantitative Cross-Model Comparison & Evaluation

*(Populate this table by running `python src/evaluate_models.py`)*

| Evaluation Metric | Classical ML (SVM / RF) | Multilayer Perceptron (MLP) | Convolutional Neural Network (CNN) |
| :--- | :--- | :--- | :--- |
| **Test Accuracy (%)** | `XX.XX%` | `XX.XX%` | `XX.XX%` |
| **Macro F1-Score** | `0.XXXX` | `0.XXXX` | `0.XXXX` |
| **Macro Precision** | `0.XXXX` | `0.XXXX` | `0.XXXX` |
| **Macro Recall** | `0.XXXX` | `0.XXXX` | `0.XXXX` |
| **Training Duration (s)**| `XX.X s` | `XX.X s` | `XX.X s` |
| **Inference Latency (ms/sample)** | `X.XX ms` | `X.XX ms` | `X.XX ms` |
| **Trainable Parameters** | `N/A` | `XXX,XXX` | `XXX,XXX` |
| **Model Disk Size (MB)** | `XX.X MB` | `X.X MB` | `X.X MB` |

### 3.1 Confusion Matrix & Error Analysis
- **Most Common Misclassifications:** Discuss which digits had highest confusion (e.g., $4 \leftrightarrow 9$, $3 \leftrightarrow 5$, $7 \leftrightarrow 1$).
- **Qualitative Failure Modes:** Show failure cases from user canvas drawings (e.g. European-style crossed 7 vs standard 7).

---

## 4. Best Model Selection & Trade-Off Justification

### 4.1 Trade-Off Analysis
Evaluate the models across competing engineering dimensions:
1. **Accuracy vs. Computational Complexity:**
   - CNN achieves the highest generalization accuracy due to spatial parameter sharing.
2. **Inference Latency & Production Viability:**
   - How does MLP latency compare to CNN latency for real-time edge or browser-based inference?
3. **Training Scalability:**
   - Compare the $O(N^2)$ to $O(N^3)$ computational scaling of kernel SVMs against the linear mini-batch gradient descent of neural networks.

### 4.2 Final Recommendation for Interactive UI
- **Selected Best Model:** `[State CNN or MLP]`
- **Engineering Justification:**
  - Summarize why this model represents the optimal Pareto frontier balancing classification accuracy, inference speed, robustness to off-center brush strokes, and binary storage footprint.

---

## 5. Team Contribution Breakdown

| Team Member | Sub-branch | Primary Contributions | Percentage Effort |
| :--- | :--- | :--- | :--- |
| **Student 1** | `student-1-data-classical` | Implemented data loader, normalization, classical ML (SVM/RF), GridSearchCV | 25% |
| **Student 2** | `student-2-mlp` | Implemented Dense-only MLP, hyperparameter grid search, validation curves | 25% |
| **Student 3** | `student-3-cnn-eval` | Implemented introductory CNN, unified evaluation suite, metrics exporter | 25% |
| **Student 4** | `student-4-streamlit-ui` | Implemented canvas preprocessing, center-of-mass centering, Streamlit UI app | 25% |
