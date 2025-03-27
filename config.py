from pathlib import Path

# --- App Configuration ---
PAGE_CONFIG = {
    "page_title": "Waste Detection",
    "page_icon": "♻️",
    "layout": "centered",
    "initial_sidebar_state": "expanded",
}

# --- Model & Image Configuration ---
MODEL_PATH = Path(__file__).parent / "models" / "waste-detection.h5"
IMG_WIDTH = 384
IMG_HEIGHT = 512
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CONFIDENCE_THRESHOLD = 0.70  # 70%

# --- Camera & ROI Configuration ---
ROI_RATIO = {"x": 0.2, "y": 0.1}  # 20% from sides, 10% from top/bottom

# --- UI Content ---
SIDEBAR_INFO = {
    "title": "📋 Project Vitals",
    "objective": """
        To build a practical AI tool that accurately classifies common waste materials,
        making recycling simpler and more effective.
    """,
    "core_technology": """
        - **Model:** CNN based on InceptionV3
        - **Framework:** TensorFlow / Keras
        - **App:** Streamlit
    """,
    "features": """
        - **Image Upload:** Classify static images.
        - **Live Camera:** Real-time local classification.
        - **Performance Dashboard:** Analyze model training and architecture.
        - **Recycling Tips:** Provides actionable advice.
    """,
    "privacy_note": """
        Your privacy is respected. This application processes images in memory 
        for classification and does not store, save, or collect any
        uploaded or captured images.
    """,
    "repo_link": "[![GitHub](https://img.shields.io/badge/GitHub-View_Source-blue?style=for-the-badge&logo=github)](https://github.com/bkljuno1/AI-Based-Waste-Detection)",
}

RECYCLING_TIPS = {
    "plastic": "Tip: Remember to rinse plastic containers and check local guidelines.",
    "paper": "Tip: Keep paper clean and dry. Greasy or food-soiled paper often cannot be recycled.",
    "cardboard": "Tip: Flatten all cardboard boxes to save space and make processing easier.",
    "metal": "Tip: Empty and rinse metal cans. Most metals are infinitely recyclable!",
    "glass": "Tip: Rinse glass bottles and jars. Some programs require separating by color.",
    "trash": "Tip: Items like plastic bags, styrofoam, and broken ceramics usually belong in the trash.",
}
