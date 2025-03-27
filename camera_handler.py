import threading

import cv2
from streamlit_webrtc import VideoProcessorBase

from config import ROI_RATIO


def get_roi_coordinates(frame):
    """Calculates the bounding box for the Region of Interest."""
    h, w, _ = frame.shape
    roi_x1 = int(w * ROI_RATIO["x"])
    roi_y1 = int(h * ROI_RATIO["y"])
    roi_x2 = int(w * (1 - ROI_RATIO["x"]))
    roi_y2 = int(h * (1 - ROI_RATIO["y"]))
    return roi_x1, roi_y1, roi_x2, roi_y2


class FrameProcessor(VideoProcessorBase):
    """
    Processes video frames from the webcam.
    - Stores the latest frame for analysis.
    - Draws a Region of Interest (ROI) box on the displayed feed.
    """

    def __init__(self):
        self.frame_lock = threading.Lock()
        self.latest_frame = None

    def recv(self, frame):
        """Receives a frame, draws the ROI, and stores it."""
        img = frame.to_ndarray(format="bgr24")
        with self.frame_lock:
            self.latest_frame = img

        # Draw the ROI box for user guidance
        h, w, _ = img.shape
        roi_x1 = int(w * ROI_RATIO["x"])
        roi_y1 = int(h * ROI_RATIO["y"])
        roi_x2 = int(w * (1 - ROI_RATIO["x"]))
        roi_y2 = int(h * (1 - ROI_RATIO["y"]))
        cv2.rectangle(img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")
