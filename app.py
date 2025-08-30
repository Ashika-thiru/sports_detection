import os
import cv2
import tempfile
import torch
import streamlit as st
from ultralytics import YOLO


# ========================
# Load YOLO model (cached)
# ========================
@st.cache_resource(show_spinner=False)
def load_model():
    # You can replace "yolov8n.pt" with your custom trained weights, e.g. "best.pt"
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

        # ✅ Run inference directly on numpy frame (avoids torch.from_numpy error)
        results = model(frame)

        # Annotate frame
        annotated_frame = results[0].plot()

        # Convert BGR → RGB for Streamlit
        stframe.image(annotated_frame, channels="BGR", use_column_width=True)

    cap.release()


# ========================
# Streamlit App Layout
# ========================
def main():
    st.title("🎯 Sports Detection ")

    model = load_model()

    # Detect if running in Streamlit Cloud
    RUNNING_IN_STREAMLIT_CLOUD = os.environ.get("STREAMLIT_RUNTIME") is not None

    if RUNNING_IN_STREAMLIT_CLOUD:
        source_type = "Upload a video"
    else:
        source_type = st.radio("Choose input source:", ["Upload a video", "Use webcam"], index=0)

    # ---------- Upload Video ----------
    if source_type == "Upload a video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            # Save uploaded video to a temp file
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

            results = model(frame)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()


# ========================
# Run the app
# ========================
if __name__ == "__main__":
    main()
