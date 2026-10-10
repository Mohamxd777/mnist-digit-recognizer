"""MNIST Digit Recognizer — refreshed Streamlit interface.

The existing preprocessing function and saved model artifacts are kept unchanged.
"""

import inspect
import json
import sys
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

MODEL_NAMES = {
    "CNN": "Convolutional Neural Network (CNN)",
    "MLP": "Multilayer Perceptron (MLP)",
    "Classical ML": "Classical ML (SVM / RF)",
}
SELECT_OPTIONS = list(MODEL_NAMES.values())

# --- Load models once; errors are shown in the status area rather than crashing the app. ---
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
    """Return ten probabilities, respecting a sklearn estimator's class order."""
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
            # Softmax models should already sum to one; normalize defensively for display.
            probs = np.clip(probs, 0, None)
            if probs.sum() > 0:
                probs = probs / probs.sum()
        pred = int(np.argmax(probs))
        return {"available": True, "digit": pred, "confidence": float(probs[pred]), "probabilities": probs}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def accuracy_text(key):
    file_map = {"CNN": "cnn_metrics.json", "MLP": "mlp_metrics.json", "Classical ML": "classical_metrics.json"}
    accuracy = read_accuracy(file_map[key])
    return f"{accuracy:.2f}% test accuracy" if accuracy is not None else "Test accuracy unavailable"


# --- Visual styling: presentation only; inference and preprocessing are unchanged. ---
st.markdown(
    """
    <style>
    :root { --ink:#172554; --muted:#60719b; --blue:#3478f6; --purple:#6848ed; }
    .stApp { background: linear-gradient(135deg,#f7f8ff 0%,#eef2ff 55%,#f8f7ff 100%); color:var(--ink); }
    /* Hide Streamlit's deploy/menu bar so the app feels like a standalone product. */
    [data-testid="stHeader"] { display:none; }
    #MainMenu, footer { visibility:hidden; }
    [data-testid="stToolbar"] { display:none !important; }
    [data-testid="stAppViewContainer"] > .main { padding-top:0; }
    .block-container { max-width:1500px; padding-top:1.15rem; padding-bottom:2rem; }
    [data-testid="stCanvas"] { border:1px solid #dce4fb; border-radius:14px; overflow:hidden; box-shadow:0 4px 14px rgba(54,72,145,.05); }
    /* Force the entire BaseWeb select control to use readable dark text on white. */
    [data-testid="stSelectbox"] div[data-baseweb="select"],
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stSelectbox"] div[data-baseweb="select"] [role="combobox"] {
      background:#ffffff !important; color:#172554 !important; border-color:#dbe2fb !important; opacity:1 !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] *,
    [data-testid="stSelectbox"] div[data-baseweb="select"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] div[data-baseweb="select"] div {
      color:#172554 !important; -webkit-text-fill-color:#172554 !important; opacity:1 !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] { border-radius:12px !important; }
    /* The opened menu is rendered outside the widget, so style its portal and every option too. */
    body div[data-baseweb="popover"], body div[data-baseweb="popover"] > div,
    body div[data-baseweb="popover"] ul,
    body div[data-baseweb="popover"] [role="listbox"],
    body div[data-baseweb="menu"], body div[data-baseweb="menu"] ul {
      background:#ffffff !important; color:#172554 !important;
    }
    body div[data-baseweb="popover"] [role="option"],
    body div[data-baseweb="popover"] [role="option"] *,
    body div[data-baseweb="menu"] li, body div[data-baseweb="menu"] li * {
      background-color:transparent !important; color:#172554 !important;
      -webkit-text-fill-color:#172554 !important; opacity:1 !important;
    }
    body div[data-baseweb="popover"] [role="option"][aria-selected="true"],
    body div[data-baseweb="popover"] [role="option"]:hover {
      background:#efedff !important;
    }
    div.stButton > button { min-height:44px; font-weight:650; }
    div.stButton > button:not([kind="primary"]) { background:#fff; color:#33436e; }
    h1,h2,h3 { color:#111b45 !important; letter-spacing:-.025em; }
    .hero { display:flex; align-items:center; gap:15px; margin-bottom:1.25rem; }
    .hero-icon { width:54px;height:54px;border-radius:17px;display:flex;align-items:center;justify-content:center;
      color:white;font-size:27px;background:linear-gradient(135deg,#397bff,#8050ee);box-shadow:0 8px 22px #6e68ed33; }
    .hero-title { font-size:2.1rem;line-height:1.1;font-weight:800;color:#172554; }
    .hero-title span { color:#6551ec; }
    .hero-sub { color:#61719a;font-size:1rem;margin-top:7px; }
    .stat { height:76px;background:rgba(255,255,255,.78);border:1px solid #e1e7ff;border-radius:18px;padding:13px 15px;display:flex;align-items:center;gap:12px; }
    .stat-icon {font-size:23px;color:#6551ec}.stat-main {font-weight:750;color:#172554}.stat-sub {font-size:.78rem;color:#61719a;margin-top:2px}
    div[data-testid="stVerticalBlockBorderWrapper"] { background:rgba(255,255,255,.83);border:1px solid #e3e9fb;border-radius:22px;box-shadow:0 7px 25px rgba(54,72,145,.045); }
    .section-head {display:flex;align-items:center;gap:11px;margin-bottom:3px;}
    .section-icon {width:42px;height:42px;display:flex;align-items:center;justify-content:center;border-radius:13px;background:linear-gradient(135deg,#e8edff,#f0eaff);font-size:21px;}
    .section-title {font-size:1.22rem;font-weight:780;color:#172554}.section-sub {font-size:.88rem;color:#66769b;margin-top:3px}
    .result-selected {border:2px solid #7460f3;border-radius:20px;padding:18px 16px;background:linear-gradient(145deg,#f0f3ff,#f7f3ff);box-shadow:0 8px 24px #6757ef18;min-height:220px;}
    .result-other {border:1px solid #e4e8fa;border-radius:18px;padding:15px 14px;background:rgba(255,255,255,.9);min-height:156px;}
    .result-label {font-weight:750;font-size:.92rem;color:#253461;display:flex;justify-content:space-between;gap:6px;align-items:center;}
    .selected-pill {font-size:.68rem;background:#e2ddff;color:#5c43dc;padding:4px 8px;border-radius:20px;white-space:nowrap;}
    .result-digit {font-size:4.6rem;line-height:1.05;text-align:center;font-weight:850;color:#4c66ef;margin:8px 0 2px;}
    .result-digit-small {font-size:2.65rem;line-height:1.05;text-align:center;font-weight:850;color:#7350e8;margin:8px 0 2px;}
    .result-confidence {text-align:center;color:#4b55d8;font-weight:750;font-size:.94rem;margin-bottom:10px;}
    .result-foot {font-size:.77rem;color:#65749a;text-align:center;margin-top:8px;}
    .status-line {display:flex;align-items:center;gap:7px;font-size:.82rem;margin:0;color:#27365f;white-space:nowrap;}
    .dot-ok {height:8px;width:8px;flex:0 0 8px;border-radius:50%;background:#25b85a;display:inline-block;}
    .dot-bad {height:8px;width:8px;flex:0 0 8px;border-radius:50%;background:#ef4444;display:inline-block;}
    .model-status-title {font-size:.78rem;font-weight:750;color:#65749a;margin:12px 0 7px;text-transform:uppercase;letter-spacing:.05em;}
    .model-status-row {display:flex;align-items:center;gap:9px;padding:9px 11px;margin:0 0 6px;background:rgba(255,255,255,.78);border:1px solid #e3e9fb;border-radius:11px;color:#27365f;font-size:.88rem;}
    .model-status-value {margin-left:auto;font-weight:650;font-size:.82rem;}
    .small-muted {font-size:.83rem;color:#6b7a9f;}
    div.stButton > button[kind="primary"] {border:0;border-radius:13px;background:linear-gradient(100deg,#4c67f2,#7a4ce9);box-shadow:0 7px 16px #6b58e833;font-weight:700;min-height:45px;}
    div.stButton > button {border-radius:12px;border-color:#dbe2fb;}
    [data-testid="stSelectbox"] label,[data-testid="stSlider"] label {font-weight:650;color:#253461;}
    div[data-testid="stAlert"] {border-radius:14px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load every model so comparison cards can show results together.
classical_model, classical_error = safe_load(load_classical_model)
mlp_model, mlp_error = safe_load(load_mlp_model)
cnn_model, cnn_error = safe_load(load_cnn_model)
models = {"Classical ML": classical_model, "MLP": mlp_model, "CNN": cnn_model}
load_errors = {"Classical ML": classical_error, "MLP": mlp_error, "CNN": cnn_error}

# --- Header ---
header, stat1, stat2, stat3 = st.columns([2.8, 1, 1, 1])
with header:
    st.markdown(
        '<div class="hero"><div class="hero-icon">✏</div><div><div class="hero-title">MNIST <span>Digit Recognizer</span></div><div class="hero-sub">Draw a digit (0–9) and see what our trained AI models predict.</div></div></div>',
        unsafe_allow_html=True,
    )
for col, icon, main, sub in [
    (stat1, "▱", "3 Models", "CNN · MLP · Classical"),
    (stat2, "ϟ", "Real-time", "Instant prediction"),
    (stat3, "▥", "MNIST", "Handwritten digits"),
]:
    with col:
        st.markdown(f'<div class="stat"><div class="stat-icon">{icon}</div><div><div class="stat-main">{main}</div><div class="stat-sub">{sub}</div></div></div>', unsafe_allow_html=True)

st.write("")

# --- Main area ---
left, right = st.columns([1, 1.08], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="section-head"><div class="section-icon">✎</div><div><div class="section-title">1. Draw a Digit</div><div class="section-sub">Draw one digit in the canvas below.</div></div></div>', unsafe_allow_html=True)
        if "canvas_key" not in st.session_state:
            st.session_state.canvas_key = 0

        # Keep model selection and both actions together, above the drawing surface.
        key_for_label = {v: k for k, v in MODEL_NAMES.items()}
        control_model_col, predict_col, clear_col = st.columns([2.5, 1.15, 0.9], gap="small", vertical_alignment="bottom")
        with control_model_col:
            selected_model = st.selectbox(
                "Model to highlight",
                SELECT_OPTIONS,
                index=0,
                key="selected_model_control",
            )
        selected_key = key_for_label[selected_model]
        with predict_col:
            predict_clicked = st.button("✦ Predict", type="primary", use_container_width=True)
        with clear_col:
            clear_clicked = st.button("↻ Clear", use_container_width=True)
        if clear_clicked:
            st.session_state.canvas_key += 1
            st.session_state.last_results = None
            st.rerun()

        stroke_width = st.slider("Brush size", min_value=10, max_value=30, value=20, step=2)
        canvas_kwargs = {
            "fill_color": "rgba(0,0,0,0)",
            "stroke_width": stroke_width,
            # Keep white strokes on black: this matches image_utils.py's current preprocessing.
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
            '<div class="small-muted" style="margin-top:8px">Draw one digit (0–9). The selected model will be highlighted, and the other available models will be compared.</div>',
            unsafe_allow_html=True,
        )

        # Stack model status items vertically beneath the drawing for a compact, readable list.
        st.markdown('<div class="model-status-title">Model status</div>', unsafe_allow_html=True)
        status_info = [("CNN", CNN_PATH, cnn_model), ("MLP", MLP_PATH, mlp_model), ("Classical ML", CLASSICAL_PATH, classical_model)]
        for key, path, model in status_info:
            if model is not None:
                label, color, dot_class = "Loaded", "#16964a", "dot-ok"
            else:
                label, color, dot_class = "Unavailable", "#dc3838", "dot-bad"
            st.markdown(
                f'<div class="model-status-row"><span class="{dot_class}"></span><strong>{key}</strong><span class="model-status-value" style="color:{color}">{label}</span></div>',
                unsafe_allow_html=True,
            )

with right:
    with st.container(border=True):
        st.markdown('<div class="section-head"><div class="section-icon">▥</div><div><div class="section-title">2. Prediction Results</div><div class="section-sub">The selected model is highlighted; compare the others below.</div></div></div>', unsafe_allow_html=True)
        if predict_clicked:
            raw_image = getattr(canvas_result, "image_data", None) if canvas_result is not None else None
            if raw_image is None:
                st.session_state.last_results = None
                st.warning("The canvas is empty. Draw a digit first, then select Predict digit.")
            else:
                flat_tensor, cnn_tensor, img2d, meta = preprocess_canvas_image(raw_image)
                if meta.get("status") != "success":
                    st.session_state.last_results = None
                    if meta.get("status") == "too_small":
                        st.warning("That drawing is too small. Please draw a complete digit.")
                    else:
                        st.warning("The canvas appears blank. Draw a clear digit and try again.")
                else:
                    with st.spinner("Running the three models…"):
                        results = {key: run_prediction(key, models, flat_tensor, cnn_tensor) for key in models}
                    st.session_state.last_results = {"results": results, "img2d": img2d}

        saved = st.session_state.get("last_results")
        if saved:
            results = saved["results"]
            selected_result = results[selected_key]
            if selected_result.get("available"):
                st.markdown(
                    f'<div class="result-selected"><div class="result-label"><span>✦ {selected_key} prediction</span><span class="selected-pill">SELECTED MODEL</span></div><div class="result-digit">{selected_result["digit"]}</div><div class="result-confidence">{selected_result["confidence"] * 100:.1f}% confidence</div><div class="result-foot">{accuracy_text(selected_key)}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"Selected model unavailable: {selected_result.get('error', 'Unknown error')}")

            other_keys = [key for key in ("CNN", "MLP", "Classical ML") if key != selected_key]
            other_cols = st.columns(2)
            for col, key in zip(other_cols, other_keys):
                result = results[key]
                with col:
                    if result.get("available"):
                        st.markdown(
                            f'<div class="result-other"><div class="result-label">{key}</div><div class="result-digit-small">{result["digit"]}</div><div class="result-confidence">{result["confidence"] * 100:.1f}% confidence</div><div class="result-foot">{accuracy_text(key)}</div></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning(f"{key} unavailable")
                        if load_errors.get(key):
                            st.caption(load_errors[key])
                        else:
                            st.caption(result.get("error", "Model file missing."))

            st.markdown("#### Class probabilities — selected model")
            if selected_result.get("available"):
                prob_df = pd.DataFrame({"Digit": [str(i) for i in range(10)], "Probability (%)": selected_result["probabilities"] * 100})
                st.bar_chart(prob_df.set_index("Digit"), y="Probability (%)", color="#6b5be8")
                with st.expander("View the 28 × 28 image the model received"):
                    st.image(saved["img2d"], caption="Preprocessed grayscale input", width=180, clamp=True)
        else:
            st.markdown('<div style="padding:42px 12px;text-align:center;color:#7482a6"><div style="font-size:3rem;margin-bottom:8px">✧</div><div style="font-weight:750;color:#34436f;font-size:1.05rem">Your predictions will appear here</div><div style="font-size:.88rem;margin-top:7px">Draw a digit, choose the model to highlight, then click Predict digit.</div></div>', unsafe_allow_html=True)

with st.expander("About the models and preprocessing"):
    st.markdown(
        """
        - **Preprocessing:** the existing `src/image_utils.py` pipeline converts the canvas drawing to a centered, normalized 28 × 28 image.
        - **Classical ML:** a saved scikit-learn model receives 784 flattened pixel values.
        - **MLP:** a feed-forward neural network receives 784 flattened pixel values.
        - **CNN:** a convolutional neural network receives the 28 × 28 image with one grayscale channel.
        - **Comparison:** each available model predicts the same drawing. The selected model is highlighted, and the other results are shown alongside it.
        """
    )

st.caption("MNIST Digit Recognizer · UI refresh only · Saved models and preprocessing are not modified")
