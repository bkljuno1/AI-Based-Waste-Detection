import streamlit as st

import config
from model_utils import load_keras_model
from ui_components import inject_custom_css, render_sidebar, render_tabs

# --- Page Setup ---
st.set_page_config(**config.PAGE_CONFIG)
inject_custom_css()
render_sidebar()

# --- Model Loading ---
with st.spinner("🧠 Loading AI Model, please wait..."):
    model = load_keras_model(config.MODEL_PATH)

# --- Session State Initialization ---
if "frozen_frame" not in st.session_state:
    st.session_state.frozen_frame = None

# --- Main App Interface ---
st.title("♻️ Waste Detection")
st.markdown(
    "Ready to get started? **Upload a photo or launch the live camera** to classify your item."
)

# --- Render Tabs ---
render_tabs(model)
