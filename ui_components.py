import cv2
import streamlit as st
import tensorflow as tf
from optree import PyTree
from PIL import Image
from streamlit_webrtc import webrtc_streamer

from camera_handler import FrameProcessor, get_roi_coordinates
from config import CLASS_NAMES, CONFIDENCE_THRESHOLD, RECYCLING_TIPS, SIDEBAR_INFO
from model_utils import get_prediction


def inject_custom_css():
    """Injects custom CSS for styling the Streamlit app."""
    st.markdown(
        """
    <style>
        /* Target the video element for sizing */
        div[data-testid="stVideo"] video {
            border-radius: 10px;
            max-width: 100%;
        }
        .bordered-container {
            border: 2px solid #2E3A59; border-radius: 10px; padding: 20px;
            background-color: #1E1E1E; margin-bottom: 20px;
        }
        /* Style for prediction text */
        .st-emotion-cache-10trblm { color: #4A90E2; }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Renders the sidebar content using data from the config dictionary."""
    st.sidebar.title(SIDEBAR_INFO["title"])

    st.sidebar.header("Objective")
    st.sidebar.info(SIDEBAR_INFO["objective"])

    st.sidebar.header("Core Technology")
    st.sidebar.markdown(SIDEBAR_INFO["core_technology"])

    st.sidebar.header("Key Features")
    st.sidebar.markdown(SIDEBAR_INFO["features"])

    st.sidebar.write("---")
    st.sidebar.subheader("Privacy Commitment")
    st.sidebar.success(SIDEBAR_INFO["privacy_note"], icon="🔒")

    st.sidebar.write("---")
    st.sidebar.subheader("Project Repository")
    st.sidebar.markdown(SIDEBAR_INFO["repo_link"])


def render_results(
    predicted_category: str, probability: float, full_prediction: PyTree
):
    """
    Displays the prediction results in a standardized format.
    Checks if the prediction confidence is above a defined threshold.
    """
    st.markdown('<div class="bordered-container">', unsafe_allow_html=True)
    st.subheader("Prediction Result:")

    if probability < CONFIDENCE_THRESHOLD:
        st.warning(
            f"**Uncertain Prediction** (Confidence: {probability:.1%})\n\n"
            "The model is not confident enough. The object may not be a supported "
            "waste type, or image quality could be improved.",
            icon="⚠️",
        )
    else:
        st.info(f"Top Prediction: **{predicted_category.capitalize()}**", icon="🎯")
        st.text("Confidence:")
        st.progress(probability, text=f"{probability * 100:.1f}%")
        tip = RECYCLING_TIPS.get(predicted_category, "No tip available.")
        st.info(tip, icon="💡")

    # Always show the probability distribution for transparency
    with st.expander("See probability distribution"):
        prob_dict = {
            CLASS_NAMES[i]: full_prediction[i] for i in range(len(CLASS_NAMES))
        }
        st.bar_chart(prob_dict)

    st.markdown("</div>", unsafe_allow_html=True)


def render_tabs(model: tf.keras.Model):
    """
    Creates and renders the main content tabs for the application.
    Args:
        model: The loaded Keras model for predictions.
    """
    tab1, tab2 = st.tabs(["📷 Upload an Image", "📹 Use Live Camera"])

    # --- Tab 1: File Uploader ---
    with tab1:
        st.header("Classify from an Image File")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        col1, col2 = st.columns([0.7, 1])

        with col1:
            if uploaded_file is None:
                st.info("☝️ Upload an image to begin analysis.")
            else:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

                file_size_mb = uploaded_file.size / (1024 * 1024)
                img_dims = f"{image.width}px x {image.height}px"
                st.caption(f"Dimensions: {img_dims} | Size: {file_size_mb:.2f} MB")

        with col2:
            if uploaded_file is not None:
                with st.spinner("🧠 Analyzing..."):
                    predicted_category, probability, full_prediction = get_prediction(
                        model, image
                    )
                    render_results(predicted_category, probability, full_prediction)
            else:
                st.info("The analysis of your uploaded image will appear here.")

    # --- Tab 2: Live Camera ---
    with tab2:
        col_live, col_analysis = st.columns([0.6, 0.4])

        with col_live:
            st.subheader("Classify from Live Feed")
            st.caption(
                "Center the object within the green rectangle and click 'Freeze'."
            )
            ctx = webrtc_streamer(
                key="waste-classifier-live",
                video_processor_factory=FrameProcessor,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            if not ctx.state.playing:
                st.markdown(
                    '<div class="bordered-container" style="text-align: center; color: #888;">'
                    "Camera feed will appear here once you press START.</div>",
                    unsafe_allow_html=True,
                )

        with col_analysis:
            st.subheader("Analysis Panel")
            btn_freeze_col, btn_clear_col = st.columns(2)

            if btn_freeze_col.button(
                "Freeze & Classify", use_container_width=True, type="primary"
            ):
                if ctx.video_processor:
                    with ctx.video_processor.frame_lock:
                        st.session_state.frozen_frame = ctx.video_processor.latest_frame
                else:
                    st.warning("Please start the camera feed first (click START).")

            if btn_clear_col.button("Clear Analysis", use_container_width=True):
                st.session_state.frozen_frame = None

            if st.session_state.frozen_frame is not None:
                frozen_frame = st.session_state.frozen_frame
                roi_x1, roi_y1, roi_x2, roi_y2 = get_roi_coordinates(frozen_frame)
                roi_to_analyze = frozen_frame[roi_y1:roi_y2, roi_x1:roi_x2]

                image_to_predict = Image.fromarray(
                    cv2.cvtColor(roi_to_analyze, cv2.COLOR_BGR2RGB)
                )

                st.image(image_to_predict, caption="Cropped & Analyzed Frame")
                with st.spinner("🧠 Analyzing..."):
                    predicted_cat, prob, full_pred = get_prediction(
                        model, image_to_predict
                    )
                    render_results(predicted_cat, prob, full_pred)
            else:
                st.info(
                    "❄️ Click 'Freeze & Classify' to analyze an object from the live feed."
                )
