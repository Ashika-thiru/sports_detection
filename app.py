import os
import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO
import matplotlib.pyplot as plt


# ========================
# Load YOLO model (cached)
# ========================
@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO("yolov8n.pt")  # replace with your trained model if needed


# ========================
# Process video & collect stats
# ========================
def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)

    # Get video details
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Temp output file
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    stframe = st.empty()

    # Stats tracking
    frame_count = 0
    player_counts = []
    ball_counts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection directly on frame (no tmp save)
        results = model.predict(frame, verbose=False)
        annotated_frame = results[0].plot()

        # Count detections
        classes = results[0].boxes.cls.tolist()
        num_players = classes.count(0)  # class 0 = person
        num_balls = classes.count(32)   # class 32 = sports ball (COCO dataset)

        player_counts.append(num_players)
        ball_counts.append(num_balls)
        frame_count += 1

        # Overlay text stats
        cv2.putText(
            annotated_frame,
            f"Players: {num_players} | Balls: {num_balls}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Write to output
        out.write(annotated_frame)

        # Show preview in Streamlit
        stframe.image(annotated_frame, channels="BGR", use_column_width=True)

    cap.release()
    out.release()

    # Return video path & stats
    return out_path, player_counts, ball_counts, frame_count


# ========================
# Plot summary stats
# ========================
def plot_stats(player_counts, ball_counts):
    fig, ax = plt.subplots()
    ax.plot(player_counts, label="Players per frame")
    ax.plot(ball_counts, label="Balls per frame")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)

    st.write("📊 **Summary Statistics:**")
    st.write(f"➡️ Average players per frame: {sum(player_counts) / len(player_counts):.2f}")
    st.write(f"➡️ Total ball detections: {sum(ball_counts)}")


# ========================
# Streamlit App Layout
# ========================
def main():
    st.title("⚽ Sports Analytics System with YOLOv8")

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

            st.info("Processing video... please wait ⏳")
            output_path, player_counts, ball_counts, frame_count = process_video(tfile.name, model)

            # Show final video
            st.success("✅ Processing complete!")
            st.video(output_path)

            # Add download button
            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=f,
                    file_name="processed_output.mp4",
                    mime="video/mp4"
                )

            # Show analytics
            plot_stats(player_counts, ball_counts)

    # ---------- Webcam Input ----------
    elif source_type == "Use webcam":
        st.info("Webcam mode is only available locally.")
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, verbose=False)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()


# ========================
# Run the app
# ========================
if __name__ == "__main__":
    main()
