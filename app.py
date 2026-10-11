"""MNIST Digit Recognizer — Refined Streamlit Interface.

An end-to-end Machine Learning and Deep Learning system for handwritten digit recognition
on the MNIST benchmark dataset. Supports real-time multi-model simultaneous inference across
Classical ML (SVM), Multilayer Perceptron (MLP), and Convolutional Neural Networks (CNN).
"""

import inspect
import json
import os
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.image_utils import preprocess_canvas_image

st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SAVED_MODELS_DIR = ROOT / "saved_models"
CLASSICAL_PATH = SAVED_MODELS_DIR / "classical_model.joblib"
MLP_PATH = SAVED_MODELS_DIR / "mlp_model.keras"
CNN_PATH = SAVED_MODELS_DIR / "cnn_model.keras"
COMPARISON_CSV_PATH = SAVED_MODELS_DIR / "comparison_table.csv"
COMPARISON_PNG_PATH = SAVED_MODELS_DIR / "model_comparison.png"

MODEL_NAMES = {
    "CNN": "Convolutional Neural Network (CNN)",
    "MLP": "Multilayer Perceptron (MLP)",
    "Classical ML": "Classical ML (SVM / RF)",
}
SELECT_OPTIONS = list(MODEL_NAMES.values())


# --- Model Loading Helpers with Caching ---
@st.cache_resource(show_spinner=False)
def load_classical_model():
    return joblib.load(CLASSICAL_PATH) if CLASSICAL_PATH.exists() else None


@st.cache_resource(show_spinner=False)
def load_mlp_model():
    return tf.keras.models.load_model(MLP_PATH, compile=False) if MLP_PATH.exists() else None


@st.cache_resource(show_spinner=False)
def load_cnn_model():
    return tf.keras.models.load_model(CNN_PATH, compile=False) if CNN_PATH.exists() else None


@st.cache_data(show_spinner=False)
def read_accuracy(filename: str):
    path = SAVED_MODELS_DIR / filename
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f).get("test_accuracy")
        return float(value) * 100 if value is not None else None
    except (OSError, ValueError, TypeError):
        return None


def safe_load(loader):
    try:
        return loader(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def classical_probabilities(model, flat_tensor):
    """Return ten probabilities, respecting a scikit-learn estimator's class order."""
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(flat_tensor)[0], dtype=float)
        classes = getattr(model, "classes_", np.arange(len(raw)))
        if hasattr(model, "named_steps") and not hasattr(model, "classes_"):
            final_estimator = list(model.named_steps.values())[-1]
            classes = getattr(final_estimator, "classes_", np.arange(len(raw)))
        probs = np.zeros(10, dtype=float)
        for i, cls in enumerate(classes):
            try:
                digit = int(cls)
                if 0 <= digit <= 9 and i < len(raw):
                    probs[digit] = raw[i]
            except (ValueError, TypeError):
                continue
        total = probs.sum()
        return probs / total if total > 0 else probs

    prediction = int(np.asarray(model.predict(flat_tensor)).reshape(-1)[0])
    probs = np.zeros(10, dtype=float)
    if 0 <= prediction <= 9:
        probs[prediction] = 1.0
    return probs


def run_prediction(key, models, flat_tensor, cnn_tensor):
    model = models[key]
    if model is None:
        return {"available": False, "error": "Model file is missing or could not be loaded."}
    try:
        if key == "Classical ML":
            probs = classical_probabilities(model, flat_tensor)
        else:
            model_input = cnn_tensor if key == "CNN" else flat_tensor
            probs = np.asarray(model.predict(model_input, verbose=0)[0], dtype=float).reshape(-1)
            if probs.size != 10:
                return {"available": False, "error": f"Expected 10 class scores, received {probs.size}."}
            probs = np.clip(probs, 0, None)
            if probs.sum() > 0:
                probs = probs / probs.sum()
        pred = int(np.argmax(probs))
        return {
            "available": True,
            "digit": pred,
            "confidence": float(probs[pred]),
            "probabilities": probs,
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def accuracy_text(key):
    file_map = {
        "CNN": "cnn_metrics.json",
        "MLP": "mlp_metrics.json",
        "Classical ML": "classical_metrics.json",
    }
    accuracy = read_accuracy(file_map[key])
    return f"{accuracy:.2f}% test accuracy" if accuracy is not None else "Test accuracy unavailable"


# --- Styling ---
st.markdown(
    """
    <style>
    :root {
      --ink: #172554;
      --muted: #60719b;
      --blue: #3478f6;
      --purple: #6848ed;
      --card-bg: rgba(255, 255, 255, 0.88);
      --card-border: #e2e8f8;
    }
    .stApp {
      background: linear-gradient(135deg, #f7f8ff 0%, #edf1fd 50%, #f6f5ff 100%);
      color: var(--ink);
    }
    [data-testid="stHeader"] { display: none; }
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stAppViewContainer"] > .main { padding-top: 0; }
    .block-container {
      max-width: 1480px;
      padding-top: 1.1rem;
      padding-bottom: 2.5rem;
    }

    /* Canvas styling */
    [data-testid="stCanvas"] {
      border: 1px solid #dbe3fb;
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 4px 16px rgba(54, 72, 145, 0.06);
      margin: 0 auto;
    }

    /* Selectbox styling */
    [data-testid="stSelectbox"] div[data-baseweb="select"],
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stSelectbox"] div[data-baseweb="select"] [role="combobox"] {
      background: #ffffff !important;
      color: #172554 !important;
      border-color: #dbe2fb !important;
      border-radius: 12px !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] * {
      color: #172554 !important;
      -webkit-text-fill-color: #172554 !important;
    }
    body div[data-baseweb="popover"],
    body div[data-baseweb="popover"] > div,
    body div[data-baseweb="menu"],
    body div[data-baseweb="menu"] ul {
      background: #ffffff !important;
      color: #172554 !important;
    }
    body div[data-baseweb="popover"] [role="option"]:hover,
    body div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
      background: #efedff !important;
    }

    /* Buttons */
    div.stButton > button {
      min-height: 44px;
      font-weight: 650;
      border-radius: 12px;
      transition: all 0.2s ease;
    }
    div.stButton > button:not([kind="primary"]) {
      background: #ffffff;
      color: #33436e;
      border: 1px solid #dbe2fb;
    }
    div.stButton > button:not([kind="primary"]):hover {
      background: #f7f9ff;
      border-color: #c7d2fe;
    }
    div.stButton > button[kind="primary"] {
      border: 0;
      border-radius: 12px;
      background: linear-gradient(100deg, #4c67f2, #7a4ce9);
      box-shadow: 0 6px 18px rgba(107, 88, 232, 0.28);
      font-weight: 700;
      color: #ffffff;
    }
    div.stButton > button[kind="primary"]:hover {
      box-shadow: 0 8px 24px rgba(107, 88, 232, 0.38);
      transform: translateY(-1px);
    }

    /* Typography */
    h1, h2, h3 { color: #111b45 !important; letter-spacing: -0.025em; }

    /* Hero section */
    .hero {
      display: flex;
      align-items: center;
      gap: 15px;
      margin-bottom: 0.5rem;
    }
    .hero-icon {
      width: 52px;
      height: 52px;
      border-radius: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-size: 26px;
      background: linear-gradient(135deg, #397bff, #8050ee);
      box-shadow: 0 8px 22px rgba(110, 104, 237, 0.25);
    }
    .hero-title {
      font-size: 2.1rem;
      line-height: 1.1;
      font-weight: 800;
      color: #172554;
    }
    .hero-title span { color: #6551ec; }
    .hero-sub {
      color: #61719a;
      font-size: 0.96rem;
      margin-top: 5px;
    }

    /* Stat badges */
    .stat {
      height: 72px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 10px 14px;
      display: flex;
      align-items: center;
      gap: 12px;
      box-shadow: 0 4px 14px rgba(54, 72, 145, 0.03);
    }
    .stat-icon { font-size: 22px; color: #6551ec; }
    .stat-main { font-weight: 750; color: #172554; font-size: 0.95rem; }
    .stat-sub { font-size: 0.76rem; color: #61719a; margin-top: 2px; }

    /* Border wrappers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      box-shadow: 0 7px 25px rgba(54, 72, 145, 0.04);
      padding: 14px 18px;
    }

    .section-head {
      display: flex;
      align-items: center;
      gap: 11px;
      margin-bottom: 12px;
    }
    .section-icon {
      width: 40px;
      height: 40px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 12px;
      background: linear-gradient(135deg, #e8edff, #f0eaff);
      font-size: 20px;
    }
    .section-title { font-size: 1.2rem; font-weight: 780; color: #172554; }
    .section-sub { font-size: 0.85rem; color: #66769b; margin-top: 2px; }

    /* Consensus Pill */
    .consensus-box {
      border-radius: 14px;
      padding: 11px 16px;
      margin-bottom: 14px;
      font-size: 0.92rem;
      display: flex;
      align-items: center;
      gap: 9px;
      font-weight: 600;
    }
    .consensus-full {
      background: linear-gradient(90deg, #ecfdf5, #f0fdf4);
      border: 1px solid #a7f3d0;
      color: #065f46;
    }
    .consensus-majority {
      background: linear-gradient(90deg, #eff6ff, #f5f3ff);
      border: 1px solid #c7d2fe;
      color: #3730a3;
    }
    .consensus-split {
      background: linear-gradient(90deg, #fffbeb, #fefce8);
      border: 1px solid #fde68a;
      color: #92400e;
    }

    /* Result cards */
    .result-selected {
      border: 2px solid #7460f3;
      border-radius: 18px;
      padding: 18px 18px;
      background: linear-gradient(145deg, #f1f4ff, #f8f5ff);
      box-shadow: 0 8px 24px rgba(103, 87, 239, 0.12);
      min-height: 220px;
      margin-bottom: 12px;
    }
    .result-other {
      border: 1px solid #e3e8f8;
      border-radius: 16px;
      padding: 14px 14px;
      background: rgba(255, 255, 255, 0.95);
      min-height: 154px;
      box-shadow: 0 4px 14px rgba(54, 72, 145, 0.03);
    }
    .result-label {
      font-weight: 750;
      font-size: 0.9rem;
      color: #253461;
      display: flex;
      justify-content: space-between;
      gap: 6px;
      align-items: center;
    }
    .selected-pill {
      font-size: 0.68rem;
      background: #e2ddff;
      color: #5c43dc;
      padding: 4px 9px;
      border-radius: 20px;
      font-weight: 700;
      letter-spacing: 0.04em;
    }
    .result-digit {
      font-size: 4.6rem;
      line-height: 1.05;
      text-align: center;
      font-weight: 850;
      color: #4361ee;
      margin: 6px 0 2px;
    }
    .result-digit-small {
      font-size: 2.6rem;
      line-height: 1.05;
      text-align: center;
      font-weight: 850;
      color: #704ee7;
      margin: 6px 0 2px;
    }
    .result-confidence {
      text-align: center;
      color: #4351d8;
      font-weight: 750;
      font-size: 0.95rem;
      margin-bottom: 8px;
    }
    .result-foot {
      font-size: 0.76rem;
      color: #65749a;
      text-align: center;
      margin-top: 6px;
    }

    /* Top candidate badges */
    .candidates-row {
      display: flex;
      justify-content: center;
      gap: 6px;
      flex-wrap: wrap;
      margin: 10px 0 6px;
      padding-top: 8px;
      border-top: 1px solid rgba(116, 96, 243, 0.15);
    }
    .cand-badge {
      font-size: 0.74rem;
      padding: 3px 8px;
      border-radius: 8px;
      font-weight: 650;
    }
    .cand-top {
      background: rgba(37, 184, 90, 0.15);
      border: 1px solid #25b85a;
      color: #15803d;
    }
    .cand-other {
      background: rgba(100, 116, 155, 0.1);
      border: 1px solid rgba(100, 116, 155, 0.2);
      color: #475569;
    }

    /* Model status rows */
    .model-status-title {
      font-size: 0.76rem;
      font-weight: 750;
      color: #65749a;
      margin: 14px 0 8px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .model-status-row {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 12px;
      margin: 0 0 6px;
      background: rgba(255, 255, 255, 0.85);
      border: 1px solid #e3e9fb;
      border-radius: 12px;
      color: #27365f;
      font-size: 0.86rem;
    }
    .dot-ok {
      height: 8px;
      width: 8px;
      flex: 0 0 8px;
      border-radius: 50%;
      background: #25b85a;
      display: inline-block;
    }
    .dot-bad {
      height: 8px;
      width: 8px;
      flex: 0 0 8px;
      border-radius: 50%;
      background: #ef4444;
      display: inline-block;
    }
    .model-status-value {
      margin-left: auto;
      font-weight: 650;
      font-size: 0.8rem;
    }

    /* Drawing tip */
    .drawing-tip {
      font-size: 0.82rem;
      color: #60719b;
      margin-top: 8px;
      line-height: 1.4;
      background: rgba(239, 243, 255, 0.6);
      padding: 8px 12px;
      border-radius: 10px;
      border: 1px solid #e2e8fa;
    }

    /* Empty state */
    .empty-state-box {
      border: 2px dashed #dbe2fb;
      border-radius: 18px;
      padding: 44px 20px;
      text-align: center;
      background: rgba(255, 255, 255, 0.55);
      margin-top: 4px;
    }
    .empty-state-icon {
      font-size: 2.8rem;
      margin-bottom: 10px;
      color: #6848ed;
    }
    .empty-state-title {
      font-weight: 750;
      color: #253461;
      font-size: 1.05rem;
    }
    .empty-state-desc {
      font-size: 0.86rem;
      color: #65749a;
      margin-top: 8px;
      line-height: 1.5;
      max-width: 380px;
      margin-left: auto;
      margin-right: auto;
    }

    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label {
      font-weight: 650;
      color: #253461;
    }
    div[data-testid="stAlert"] { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load all 3 models safely
classical_model, classical_error = safe_load(load_classical_model)
mlp_model, mlp_error = safe_load(load_mlp_model)
cnn_model, cnn_error = safe_load(load_cnn_model)
models = {"Classical ML": classical_model, "MLP": mlp_model, "CNN": cnn_model}
load_errors = {"Classical ML": classical_error, "MLP": mlp_error, "CNN": cnn_error}

# --- Header Section ---
header_col, stat1, stat2, stat3, stat4 = st.columns([2.6, 1, 1, 1, 1.1])
with header_col:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-icon">✏️</div>
            <div>
                <div class="hero-title">MNIST <span>Digit Recognizer</span></div>
                <div class="hero-sub">Interactive canvas with real-time multi-model benchmarking.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with stat1:
    st.markdown(
        '<div class="stat"><div class="stat-icon">▱</div><div><div class="stat-main">3 Models</div><div class="stat-sub">CNN · MLP · SVM</div></div></div>',
        unsafe_allow_html=True,
    )
with stat2:
    st.markdown(
        '<div class="stat"><div class="stat-icon">⚡</div><div><div class="stat-main">Real-time</div><div class="stat-sub">1-Click Ensemble</div></div></div>',
        unsafe_allow_html=True,
    )
with stat3:
    st.markdown(
        '<div class="stat"><div class="stat-icon">🎯</div><div><div class="stat-main">99.59% Acc</div><div class="stat-sub">MNIST Benchmark</div></div></div>',
        unsafe_allow_html=True,
    )
with stat4:
    st.markdown(
        '<div class="stat"><div class="stat-icon">☁️</div><div><div class="stat-main">Cloud Live</div><div class="stat-sub"><a href="https://github.com/Mohamxd777/mnist-digit-recognizer" target="_blank" style="color:#6551ec;text-decoration:none;font-weight:600;">GitHub Repo ↗</a></div></div></div>',
        unsafe_allow_html=True,
    )

st.write("")

# --- Main Interface ---
col_left, col_right = st.columns([1, 1.08], gap="large")

with col_left:
    with st.container(border=True):
        st.markdown(
            '<div class="section-head"><div class="section-icon">✎</div><div><div class="section-title">1. Draw a Digit</div><div class="section-sub">Draw any handwritten digit (0–9) on the canvas.</div></div></div>',
            unsafe_allow_html=True,
        )

        if "canvas_key" not in st.session_state:
            st.session_state.canvas_key = 0

        key_for_label = {v: k for k, v in MODEL_NAMES.items()}
        control_col, predict_btn_col, clear_btn_col = st.columns(
            [2.4, 1.25, 0.9], gap="small", vertical_alignment="bottom"
        )
        with control_col:
            selected_model_label = st.selectbox(
                "Model to highlight:",
                SELECT_OPTIONS,
                index=0,
                key="selected_model_control",
            )
        selected_key = key_for_label[selected_model_label]

        with predict_btn_col:
            predict_clicked = st.button("✦ Predict", type="primary", use_container_width=True)

        with clear_btn_col:
            clear_clicked = st.button("↻ Clear", use_container_width=True)

        if clear_clicked:
            st.session_state.canvas_key += 1
            st.session_state.last_results = None
            st.rerun()

        stroke_width = st.slider(
            "Brush Stroke Width:",
            min_value=12,
            max_value=30,
            value=22,
            step=2,
            help="Adjust thickness to mimic real handwriting strokes.",
        )

        canvas_kwargs = {
            "fill_color": "rgba(0,0,0,0)",
            "stroke_width": stroke_width,
            "stroke_color": "#FFFFFF",
            "background_color": "#000000",
            "height": 330,
            "width": 430,
            "drawing_mode": "freedraw",
            "key": f"canvas_{st.session_state.canvas_key}",
        }
        try:
            if "return_image_data" in inspect.signature(st_canvas).parameters:
                canvas_kwargs["return_image_data"] = True
        except (TypeError, ValueError):
            pass

        canvas_result = st_canvas(**canvas_kwargs)

        st.markdown(
            """
            <div class="drawing-tip">
                💡 <b>Drawing Tip:</b> Draw a single, centered digit with clear strokes. 
                All 3 models will classify the drawing simultaneously upon clicking <b>✦ Predict</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Model Status Tray
        st.markdown('<div class="model-status-title">Saved Model Status</div>', unsafe_allow_html=True)
        status_items = [
            ("CNN", CNN_PATH, cnn_model, "99.09%"),
            ("MLP", MLP_PATH, mlp_model, "99.59%"),
            ("Classical ML", CLASSICAL_PATH, classical_model, "97.02%"),
        ]
        for key, path, model, acc_val in status_items:
            if model is not None:
                dot_class, label, color = "dot-ok", f"Loaded ({acc_val})", "#16964a"
            else:
                dot_class, label, color = "dot-bad", "Unavailable", "#dc3838"
            st.markdown(
                f'<div class="model-status-row"><span class="{dot_class}"></span><strong>{key}</strong><span class="model-status-value" style="color:{color}">{label}</span></div>',
                unsafe_allow_html=True,
            )

with col_right:
    with st.container(border=True):
        st.markdown(
            '<div class="section-head"><div class="section-icon">▥</div><div><div class="section-title">2. Prediction Results</div><div class="section-sub">Cross-model comparison on the exact same stroke pattern.</div></div></div>',
            unsafe_allow_html=True,
        )

        if predict_clicked:
            raw_image = getattr(canvas_result, "image_data", None) if canvas_result is not None else None
            if raw_image is None:
                st.session_state.last_results = None
                st.warning("⚠️ The canvas is empty. Please draw a digit first, then click Predict.")
            else:
                flat_tensor, cnn_tensor, img2d, meta = preprocess_canvas_image(raw_image)
                if meta.get("status") != "success":
                    st.session_state.last_results = None
                    if meta.get("status") == "too_small":
                        st.warning("⚠️ That drawing is too small or faint. Please draw a complete digit.")
                    else:
                        st.warning("⚠️ The canvas appears blank. Please draw a clear digit and try again.")
                else:
                    with st.spinner("Classifying stroke pattern with all 3 models…"):
                        results = {key: run_prediction(key, models, flat_tensor, cnn_tensor) for key in models}
                    st.session_state.last_results = {
                        "results": results,
                        "img2d": img2d,
                        "meta": meta,
                    }

        saved = st.session_state.get("last_results")
        if saved:
            results = saved["results"]
            selected_result = results[selected_key]

            # 1. Consensus Banner Across Models
            available_preds = [
                results[k]["digit"] for k in ("CNN", "MLP", "Classical ML") if results[k].get("available")
            ]
            if len(available_preds) == 3 and available_preds[0] == available_preds[1] == available_preds[2]:
                consensus_banner = (
                    f'<div class="consensus-box consensus-full">'
                    f'<span>🎯</span> <div><b>Full Consensus:</b> All 3 models unanimously predict <b>Digit {available_preds[0]}</b></div>'
                    f'</div>'
                )
            elif len(available_preds) >= 2 and len(set(available_preds)) < len(available_preds):
                majority_digit = Counter(available_preds).most_common(1)[0][0]
                consensus_banner = (
                    f'<div class="consensus-box consensus-majority">'
                    f'<span>⚡</span> <div><b>Majority Agreement:</b> 2 of 3 models predict <b>Digit {majority_digit}</b></div>'
                    f'</div>'
                )
            else:
                consensus_banner = (
                    f'<div class="consensus-box consensus-split">'
                    f'<span>⚖️</span> <div><b>Split Decision:</b> Models evaluated different subtle stroke features</div>'
                    f'</div>'
                )
            st.markdown(consensus_banner, unsafe_allow_html=True)

            # 2. Selected Model Hero Card
            if selected_result.get("available"):
                probs = selected_result["probabilities"]
                top3_idx = np.argsort(probs)[::-1][:3]
                cand_html = "".join(
                    [
                        f'<span class="cand-badge {"cand-top" if i == 0 else "cand-other"}">#{i+1}: Digit {idx} ({probs[idx]*100:.1f}%)</span>'
                        for i, idx in enumerate(top3_idx)
                    ]
                )

                st.markdown(
                    f"""
                    <div class="result-selected">
                        <div class="result-label">
                            <span>✦ {selected_key} Prediction</span>
                            <span class="selected-pill">FEATURED MODEL</span>
                        </div>
                        <div class="result-digit">{selected_result["digit"]}</div>
                        <div class="result-confidence">{selected_result["confidence"] * 100:.1f}% Confidence</div>
                        <div class="candidates-row">
                            {cand_html}
                        </div>
                        <div class="result-foot">{accuracy_text(selected_key)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"Selected model unavailable: {selected_result.get('error', 'Unknown error')}")

            # 3. Other Models Comparison Cards
            other_keys = [k for k in ("CNN", "MLP", "Classical ML") if k != selected_key]
            col_other1, col_other2 = st.columns(2)
            for col_target, other_k in zip((col_other1, col_other2), other_keys):
                other_res = results[other_k]
                with col_target:
                    if other_res.get("available"):
                        st.markdown(
                            f"""
                            <div class="result-other">
                                <div class="result-label">{other_k}</div>
                                <div class="result-digit-small">{other_res["digit"]}</div>
                                <div class="result-confidence">{other_res["confidence"] * 100:.1f}% Confidence</div>
                                <div class="result-foot">{accuracy_text(other_k)}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning(f"{other_k} unavailable")
                        if load_errors.get(other_k):
                            st.caption(load_errors[other_k])
                        else:
                            st.caption(other_res.get("error", "Model file missing."))

            # 4. Class Probability Distribution Bar Chart
            st.markdown("#### Class Probabilities — Selected Model")
            if selected_result.get("available"):
                prob_df = pd.DataFrame(
                    {
                        "Digit": [str(i) for i in range(10)],
                        "Probability (%)": selected_result["probabilities"] * 100,
                    }
                )
                st.bar_chart(
                    prob_df.set_index("Digit"),
                    y="Probability (%)",
                    color="#6b5be8",
                    height=200,
                )

                # Preprocessing Preview Expander
                with st.expander("🔍 View Preprocessed 28 × 28 MNIST Image & Metadata"):
                    img_col, meta_col = st.columns([1, 1.4])
                    with img_col:
                        st.image(
                            saved["img2d"],
                            caption="Canonical 28×28 Grayscale",
                            width=140,
                            clamp=True,
                        )
                    with meta_col:
                        meta_info = saved.get("meta", {})
                        bbox = meta_info.get("bounding_box", {})
                        st.write("**Preprocessing Pipeline:**")
                        st.caption(
                            f"• **Bounding Box:** {bbox.get('w', 0)}px × {bbox.get('h', 0)}px\n"
                            f"• **Scaled Frame:** 20 × 20 pixels (aspect preserved)\n"
                            f"• **Centering:** Center-of-Mass translated to 28 × 28 canvas\n"
                            f"• **Pixel Range:** [0.0, 1.0] float32 normalized"
                        )
        else:
            st.markdown(
                """
                <div class="empty-state-box">
                    <div class="empty-state-icon">✧</div>
                    <div class="empty-state-title">Awaiting Your Drawing</div>
                    <div class="empty-state-desc">
                        1. Draw any single digit (0–9) on the canvas to the left.<br>
                        2. Click <b>✦ Predict</b> to evaluate across CNN, MLP, and SVM.<br>
                        3. Switch the highlighted model anytime to compare confidence distributions.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# --- Information Expander & Benchmark Matrix ---
with st.expander("ℹ️ About the Models, Architecture & Cross-Model Benchmarks", expanded=False):
    tab_benchmarks, tab_arch = st.tabs(["📊 Cross-Model Benchmark Matrix", "🧠 Model Architecture & Pipeline"])

    with tab_benchmarks:
        st.markdown("#### Empirical MNIST Test Set Benchmarks ($N=21,000$ Samples)")
        if COMPARISON_CSV_PATH.exists():
            df_comp = pd.read_csv(COMPARISON_CSV_PATH)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
        else:
            st.info("Run `python src/evaluate_models.py` to regenerate the comparative benchmark table.")

        if COMPARISON_PNG_PATH.exists():
            st.image(str(COMPARISON_PNG_PATH), caption="Accuracy and Inference Latency Comparison", use_container_width=True)

    with tab_arch:
        st.markdown(
            """
            - **Canonical Preprocessing (`src/image_utils.py`):**
              - Raw canvas RGBA strokes are bounded, aspect-ratio preserved to a 20×20 bounding box, and centered into a standard 28×28 frame using Center-of-Mass alignment, matching the canonical MNIST database design.
            - **Classical ML (`src/train_classical.py`):**
              - Scikit-Learn Support Vector Classifier (`SVC(kernel='rbf', C=10.0)`), evaluated on 784-dimensional flattened vectors with hyperparameter optimization via `GridSearchCV`.
            - **Multilayer Perceptron (`src/train_mlp.py`):**
              - Deep feedforward neural network utilizing **strictly Dense layers** (`Dense(512) -> Dense(256) -> Dense(128) -> Dense(10)`), with Batch Normalization and Dropout regularization.
            - **Convolutional Neural Network (`src/train_cnn.py`):**
              - Introductory 2D CNN architecture (`Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Dense(128) -> Dense(10)`), learning translation-invariant spatial stroke features.
            """
        )

# Footer
st.markdown(
    """
    <div style="text-align:center;color:#8592b5;font-size:0.8rem;margin-top:25px;padding-top:15px;border-top:1px solid #e2e8f8;">
        MNIST Digit Recognizer • Assignment 1 Team Project • 
        <a href="https://github.com/Mohamxd777/mnist-digit-recognizer" target="_blank" style="color:#6848ed;text-decoration:none;font-weight:600;">GitHub Repository</a> • 
        <a href="https://mnistdigitrecognizerr.streamlit.app/" target="_blank" style="color:#6848ed;text-decoration:none;font-weight:600;">Live Web App</a>
    </div>
    """,
    unsafe_allow_html=True,
)
