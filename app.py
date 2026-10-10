"""
app.py
======
Author: Student 4 (Streamlit UI & Preprocessing Specialist)
Assignment 1: Training Models on MNIST + Interactive Drawing App (Deliverable 5)

An interactive web application allowing users to draw a digit in real-time,
choose between trained models (Classical ML, MLP, CNN), and visualize
model prediction confidences across classes 0-9.
"""

import inspect
import json
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

# Ensure root directory is on Python path
sys.path.append(str(Path(__file__).resolve().parent))
from src.image_utils import preprocess_canvas_image

# --- Page Configuration ---
st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAVED_MODELS_DIR = Path(__file__).resolve().parent / "saved_models"
CLASSICAL_PATH = SAVED_MODELS_DIR / "classical_model.joblib"
MLP_PATH = SAVED_MODELS_DIR / "mlp_model.keras"
CNN_PATH = SAVED_MODELS_DIR / "cnn_model.keras"


# --- Model Loading Helpers with Caching ---
@st.cache_resource
def load_classical_model():
    if CLASSICAL_PATH.exists():
        return joblib.load(CLASSICAL_PATH)
    return None


@st.cache_resource
def load_mlp_model():
    if MLP_PATH.exists():
        return tf.keras.models.load_model(MLP_PATH)
    return None


@st.cache_resource
def load_cnn_model():
    if CNN_PATH.exists():
        return tf.keras.models.load_model(CNN_PATH)
    return None


@st.cache_data
def get_model_accuracy(model_choice_name: str) -> str:
    """Retrieves benchmark test accuracy for the selected model if available."""
    metrics_map = {
        "Classical ML (SVM / RF)": SAVED_MODELS_DIR / "classical_metrics.json",
        "Multilayer Perceptron (MLP)": SAVED_MODELS_DIR / "mlp_metrics.json",
        "Convolutional Neural Network (CNN)": SAVED_MODELS_DIR / "cnn_metrics.json",
        "Convolutional Neural Net (CNN)": SAVED_MODELS_DIR / "cnn_metrics.json",
        "CNN": SAVED_MODELS_DIR / "cnn_metrics.json",
    }
    path = metrics_map.get(model_choice_name)
    if path and path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
                acc = data.get("test_accuracy")
                if acc is not None:
                    return f"{acc * 100:.2f}%"
        except Exception:
            pass
    return "N/A"


# --- Custom CSS Styling ---
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .pred-box {
        background-color: rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .pred-label {
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
    }
    .pred-digit {
        font-size: 5rem;
        font-weight: 800;
        color: #38bdf8;
        line-height: 1;
        margin: 10px 0;
    }
    .metrics-row {
        display: flex;
        justify-content: space-around;
        align-items: center;
        gap: 10px;
        margin-top: 14px;
        padding-top: 14px;
        border-top: 1px solid rgba(255, 255, 255, 0.12);
    }
    .metric-item {
        flex: 1;
        text-align: center;
    }
    .metric-sublabel {
        font-size: 0.72rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value-conf {
        font-size: 1.25rem;
        font-weight: 700;
        color: #22c55e;
    }
    .metric-value-acc {
        font-size: 1.25rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 80%;
        font-weight: 700;
        border-radius: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- App Header ---
st.markdown('<div class="main-title">✏️ MNIST Handwritten Digit Recognizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Assignment 1 • Team Project • Interactive Drawing App & Model Inference</div>',
    unsafe_allow_html=True,
)

# --- Sidebar Controls & Status ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model Selector
    model_choice = st.selectbox(
        "Select Classification Model:",
        options=[
            "Convolutional Neural Network (CNN)",
            "Multilayer Perceptron (MLP)",
            "Classical ML (SVM / RF)",
        ],
        index=0,
    )

    st.markdown("---")
    st.header("🎨 Drawing Canvas Settings")
    stroke_width = st.slider("Brush Stroke Width:", min_value=10, max_value=30, value=20, step=2)
    
    st.markdown("---")
    st.header("📦 Saved Model Status")
    
    # Status badges
    def render_status(label: str, path: Path):
        exists = path.exists()
        badge = "🟢 Ready" if exists else "🔴 Not Found"
        st.write(f"**{label}:** {badge}")
        if not exists:
            st.caption(f"Missing `{path.name}`")

    render_status("Classical ML", CLASSICAL_PATH)
    render_status("MLP (Dense)", MLP_PATH)
    render_status("CNN", CNN_PATH)
    
    st.markdown("---")
    st.caption("Developed by Team of 4 • Assignment 1")


# --- Main Layout: Drawing on Left, Prediction on Right ---
col_canvas, col_results = st.columns([1.1, 1], gap="large")

with col_canvas:
    st.subheader("Draw a Digit (0 - 9)")
    st.write("Draw in the center of the canvas below. Click **Predict** when ready.")

    # Key state to support clear button
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    col_btn_clear, col_btn_predict = st.columns([1.2, 1.8])
    with col_btn_clear:
        if st.button("🗑️ Clear Canvas", use_container_width=True):
            st.session_state.canvas_key += 1
            st.rerun()

    # Drawable Canvas Component
    canvas_kwargs = {
        "fill_color": "rgba(0, 0, 0, 0)",
        "stroke_width": stroke_width,
        "stroke_color": "#FFFFFF",
        "background_color": "#000000",
        "height": 280,
        "width": 280,
        "drawing_mode": "freedraw",
        "key": f"canvas_{st.session_state.canvas_key}",
    }
    try:
        if "return_image_data" in inspect.signature(st_canvas).parameters:
            canvas_kwargs["return_image_data"] = True
    except Exception:
        pass

    canvas_result = st_canvas(**canvas_kwargs)

with col_results:
    st.subheader("Prediction & Confidence")
    
    with col_btn_predict:
        predict_clicked = st.button("🚀 Predict Digit", type="primary", use_container_width=True)

    if predict_clicked:
        raw_image = None
        if canvas_result is not None:
            try:
                raw_image = canvas_result.image_data
            except Exception:
                raw_image = None

        if raw_image is None:
            st.warning("Canvas is empty. Please draw a digit first.")
        else:
            # Preprocess the canvas drawing using Student 4 pipeline
            flat_tensor, cnn_tensor, img2d, meta = preprocess_canvas_image(
                raw_image
            )

            if meta.get("status") in ["blank_canvas", "no_contour", "empty_input"]:
                st.warning("⚠️ Canvas appears to be blank. Please draw a digit clearly!")
            elif meta.get("status") == "too_small":
                st.warning("⚠️ That looks like just a dot or a speck! Please draw a full digit (0 – 9).")
            else:
                # 1. Load the active model
                model = None
                input_tensor = None

                if model_choice == "Classical ML (SVM / RF)":
                    model = load_classical_model()
                    input_tensor = flat_tensor
                    train_script = "python src/train_classical.py"
                elif model_choice == "Multilayer Perceptron (MLP)":
                    model = load_mlp_model()
                    input_tensor = flat_tensor
                    train_script = "python src/train_mlp.py"
                else:  # CNN
                    model = load_cnn_model()
                    input_tensor = cnn_tensor
                    train_script = "python src/train_cnn.py"

                # Check if model exists
                if model is None:
                    st.error(
                        f"🚨 Model `{model_choice}` has not been trained yet!\n\n"
                        f"Please train it first by running in your terminal:\n"
                        f"```bash\n{train_script}\n```"
                    )
                else:
                    # 2. Run Inference
                    with st.spinner("Analyzing stroke pattern..."):
                        if model_choice == "Classical ML (SVM / RF)":
                            if hasattr(model, "predict_proba"):
                                probabilities = model.predict_proba(input_tensor)[0]
                            else:
                                pred = model.predict(input_tensor)[0]
                                probabilities = np.zeros(10)
                                probabilities[pred] = 1.0
                        else:
                            probs = model.predict(input_tensor, verbose=0)
                            probabilities = probs[0]

                        pred_digit = int(np.argmax(probabilities))
                        confidence = float(probabilities[pred_digit])
                        model_acc_str = get_model_accuracy(model_choice)

                    # 3. Display High-Level Result Card
                    res_col1, res_col2 = st.columns([1, 1])
                    with res_col1:
                        st.markdown(
                            f"""
                            <div class="pred-box">
                                <div class="pred-label">PREDICTED DIGIT</div>
                                <div class="pred-digit">{pred_digit}</div>
                                <div class="metrics-row">
                                    <div class="metric-item">
                                        <div class="metric-sublabel">Confidence</div>
                                        <div class="metric-value-conf">{confidence * 100:.1f}%</div>
                                    </div>
                                    <div class="metric-item" style="border-left: 1px solid rgba(255, 255, 255, 0.15);">
                                        <div class="metric-sublabel">Model Accuracy</div>
                                        <div class="metric-value-acc">{model_acc_str}</div>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.caption("ℹ️ **Confidence:** Certainty on this stroke • **Accuracy:** Test benchmark score.")

                    with res_col2:
                        st.caption("Normalized 28x28 Model Input:")
                        # Display 28x28 normalized image as viewed by model
                        st.image(
                            img2d,
                            caption="MNIST Centered 28x28",
                            width=140,
                            clamp=True,
                        )

                    # 4. Class Probability Distribution Bar Chart
                    st.markdown("#### Class Probabilities (0 - 9)")
                    prob_df = pd.DataFrame(
                        {
                            "Digit": [str(d) for d in range(10)],
                            "Probability (%)": [p * 100.0 for p in probabilities],
                        }
                    )
                    
                    st.bar_chart(
                        prob_df.set_index("Digit"),
                        y="Probability (%)",
                        color="#1f77b4",
                    )
                    
                    # Top 3 summary table
                    top3_indices = np.argsort(probabilities)[::-1][:3]
                    top3_text = " | ".join(
                        [f"**#{i+1}: Digit {idx}** ({probabilities[idx]*100:.1f}%)" for i, idx in enumerate(top3_indices)]
                    )
                    st.info(f"Top Candidate Predictions: {top3_text}")
    else:
        st.info("👈 Draw a digit on the canvas and click **'🚀 Predict Digit'** to see classification results.")

# --- Architecture Details Expander ---
with st.expander("ℹ️ About the Models & Pipeline"):
    st.markdown(
        """
        - **Preprocessing (Student 4):**
          - Raw canvas stroke is extracted, bounded, aspect-ratio preserved to 20x20 pixels, and centered inside a 28x28 frame using center-of-mass alignment, mirroring the original MNIST database construction.
        - **Classical ML (Student 1):**
          - Scikit-Learn Support Vector Classifier (`SVC(kernel='rbf')`) / Random Forest trained on 784-dimensional flattened vectors with hyperparameter tuning via `GridSearchCV`.
        - **Multilayer Perceptron (Student 2):**
          - Feedforward Neural Network using **Dense layers only** (`Dense(512) -> Dense(256) -> Dense(10)`), with Batch Normalization and Dropout.
        - **Convolutional Neural Network (Student 3):**
          - Introductory 2D CNN architecture (`Conv2D -> MaxPool2D -> Conv2D -> MaxPool2D -> Dense(128) -> Dense(10)`), capturing local 2D spatial features and translation invariance.
        """
    )
