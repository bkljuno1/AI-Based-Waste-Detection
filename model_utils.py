import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

from config import CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH


@st.cache_resource
def load_keras_model(model_path: str):
    """Load the Keras model from disk, caching it for performance."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}", icon="🚨")
        # Stop the app if the model fails to load, as it's critical.
        st.stop()


def get_prediction(model: tf.keras.Model, image: Image.Image):
    """
    Takes a PIL Image, preprocesses it, and returns prediction results.

    Args:
        model: The trained Keras model.
        image: The PIL Image to classify.

    Returns:
        A tuple containing (predicted_category, probability, full_prediction_array).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    target_size = (IMG_WIDTH, IMG_HEIGHT)
    image_resized = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    predicted_category = CLASS_NAMES[predicted_index]
    probability = float(prediction[0][predicted_index])

    return predicted_category, probability, prediction[0]
