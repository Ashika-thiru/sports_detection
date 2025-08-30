import os
import cv2
import tempfile
import numpy as np
import torch
import streamlit as st
from ultralytics import YOLO

# ========================
# Load YOLO model (cached)
# ========================
@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO("yolov8n.pt", task="detect")

# ========================
# Process video with YOLO
# ========================
def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ Ensure frame is valid: always 3 channels, contiguous, uint8
        if frame is None:
            continue
        if len(frame.shape) == 2:  # grayscale → convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # RGBA → convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        # ✅ YOLO inference
        results = model.predict(source=frame)

        annotated_frame = results[0].plot()

        # Show in Streamlit
        stframe.image(annotated_frame, channels="BGR", use_column_width=True)

    cap.release()

# ========================
# Streamlit App Layout
# ========================
def main():
    st.title("🎯 Sports Detection with YOLOv8")

    model = load_model()

    RUNNING_IN_STREAMLIT_CLOUD = os.environ.get("STREAMLIT_RUNTIME") is not None

    if RUNNING_IN_STREAMLIT_CLOUD:
        source_type = "Upload a video"
    else:
        source_type = st.radio("Choose input source:", ["Upload a video", "Use webcam"], index=0)

    # ---------- Upload Video ----------
    if source_type == "Upload a video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            process_video(tfile.name, model)

    # ---------- Webcam Input ----------
    elif source_type == "Use webcam":
        st.info("Webcam mode is only available when running locally.")
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None:
                continue
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            frame = np.ascontiguousarray(frame, dtype=np.uint8)

            results = model.predict(source=frame)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()

# ========================
# Run the app
# ========================
if __name__ == "__main__":
    main()
