import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from pathlib import Path

# =========================
# App / Model Setup
# =========================
st.set_page_config(page_title="Sports Detection", layout="wide")

@st.cache_resource(show_spinner=False)
def load_model():
    # Use your custom trained model here if you have one
    return YOLO("yolov8n.pt")

model = load_model()

st.title("Sports Analytics System")

# =========================
# Sidebar sections
# =========================
st.sidebar.header("📊 Live Match Stats")
players_metric = st.sidebar.empty()
balls_metric   = st.sidebar.empty()
frames_metric  = st.sidebar.empty()

st.sidebar.subheader("🤝 Team Insights")
possession_metric = st.sidebar.empty()
passes_metric     = st.sidebar.empty()

st.sidebar.subheader("⚡ Movement Tracking")
ball_speed_metric    = st.sidebar.empty()
ball_distance_metric = st.sidebar.empty()

st.sidebar.subheader("🎯 Action Recognition")
last_action_metric = st.sidebar.empty()

st.sidebar.subheader("🏷️ Highlights")
highlights_saved_metric = st.sidebar.empty()

st.sidebar.subheader("📈 Cumulative Totals")
total_players_metric = st.sidebar.empty()
total_balls_metric   = st.sidebar.empty()

st.sidebar.subheader("📉 Detection Trends")
trends_placeholder = st.sidebar.empty()

st.sidebar.subheader("🔥 Ball Heatmap")
heatmap_placeholder = st.sidebar.empty()

# =========================
# Session State (all in one place)
# =========================
ss = st.session_state
def _init_state():
    ss.total_players = 0
    ss.total_balls = 0
    ss.history = []                     # [{frame, players, balls}]
    ss.ball_positions = []              # [(x, y)]
    ss.team_centroids = {"Team A": None, "Team B": None}  # HSV centroids
    ss.team_possession_frames = defaultdict(int)
    ss.last_ball_owner_team = None
    ss.last_ball_owner_pid = None
    ss.pass_count = 0
    ss.tracks = {}                      # pid -> (cx, cy)
    ss.track_id_counter = 1
    ss.ball_prev_pos = None
    ss.ball_total_distance = 0.0
    ss.ball_speeds = []
    ss.last_action = "—"
    ss.action_window = deque(maxlen=15)  # recent speeds/owners
    ss.highlights = []                   # [(frame_idx, filepath)]
    ss._last_highlight_frame = -9999

if "initialized" not in ss:
    _init_state()
    ss.initialized = True

# =========================
# Input Source controls
# =========================
col_src1, col_src2 = st.columns([1, 2])
with col_src1:
    source_type = st.radio("Choose input source:", ["Upload a video", "Use webcam"], index=0)
with col_src2:
    if source_type == "Upload a video":
        uploaded_file = st.file_uploader("Upload a sports video", type=["mp4", "avi", "mov", "mkv"])
    else:
        uploaded_file = None

start_btn = st.button("▶️ Start Analysis")

# Main video canvas
stframe = st.empty()

# =========================
# Helpers
# =========================
def extract_dominant_hsv(player_crop_bgr):
    """Return a dominant HSV color for a player's jersey using k-means."""
    if player_crop_bgr is None or player_crop_bgr.size == 0:
        return None
    crop = cv2.resize(player_crop_bgr, (32, 48), interpolation=cv2.INTER_AREA)
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    Z = hsv.reshape((-1, 3)).astype(np.float32)
    if len(Z) < 2:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _ret, _labels, centers = cv2.kmeans(Z, 2, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.float32)
    idx = int(np.argmax(centers[:, 1]))  # choose cluster with higher saturation
    return centers[idx]  # HSV

def hsv_distance(a, b):
    if a is None or b is None:
        return 1e9
    dh = min(abs(a[0] - b[0]), 180 - abs(a[0] - b[0])) / 90.0
    ds = abs(a[1] - b[1]) / 255.0
    dv = abs(a[2] - b[2]) / 255.0
    return 2.0 * dh + 1.0 * ds + 0.3 * dv

def assign_team_for_player(hsv_color):
    """EMA-cluster players to Team A/B by jersey color."""
    if hsv_color is None:
        return "Team A"
    A = ss.team_centroids["Team A"]
    B = ss.team_centroids["Team B"]
    if A is None:
        ss.team_centroids["Team A"] = hsv_color.copy()
        return "Team A"
    if B is None:
        if hsv_distance(hsv_color, A) > 0.5:
            ss.team_centroids["Team B"] = hsv_color.copy()
            return "Team B"
        ss.team_centroids["Team A"] = 0.9 * A + 0.1 * hsv_color
        return "Team A"
    dA, dB = hsv_distance(hsv_color, A), hsv_distance(hsv_color, B)
    if dA <= dB:
        ss.team_centroids["Team A"] = 0.9 * A + 0.1 * hsv_color
        return "Team A"
    ss.team_centroids["Team B"] = 0.9 * B + 0.1 * hsv_color
    return "Team B"

def match_tracks(prev_tracks, centers, max_dist=50.0):
    """Nearest-neighbor tracking to keep stable IDs."""
    assigned = set()
    new_tracks, result = {}, []
    # match old -> new
    for pid, (px, py) in prev_tracks.items():
        best, best_d, best_j = None, 1e9, -1
        for j, (cx, cy) in enumerate(centers):
            if j in assigned:
                continue
            d = np.hypot(cx - px, cy - py)
            if d < best_d:
                best, best_d, best_j = (cx, cy), d, j
        if best is not None and best_d <= max_dist:
            assigned.add(best_j)
            new_tracks[pid] = best
            result.append((pid, best))
    # new IDs
    for j, (cx, cy) in enumerate(centers):
        if j in assigned:
            continue
        pid = ss.track_id_counter
        ss.track_id_counter += 1
        new_tracks[pid] = (cx, cy)
        result.append((pid, (cx, cy)))
    ss.tracks = new_tracks
    return result

def recognize_action(speed_pxps, owner_changed, near_player):
    """
    Very simple rule-based action recognition:
    - shot: speed spike & not near player
    - pass: owner_changed
    - dribble: low speed but near player for several frames
    """
    # record to window
    ss.action_window.append((speed_pxps, owner_changed, near_player))
    # heuristics thresholds
    if speed_pxps is None:
        return None
    SHOT_SPEED = 1200  # px/s (adjust to your video scale)
    DRIBBLE_MAX = 350  # px/s
    if speed_pxps > SHOT_SPEED and not near_player:
        return "Shot"
    if owner_changed:
        return "Pass"
    # dribble when slower & near player for several frames
    if sum(1 for (_s, _ch, npn) in ss.action_window if npn and (_s or 0) < DRIBBLE_MAX) >= 8:
        return "Dribble"
    return None

def save_highlight(frame_bgr, frame_idx):
    """Save a JPEG keyframe when a highlight happens (spaced by 20 frames)."""
    if frame_idx - ss._last_highlight_frame < 20:
        return
    out_dir = Path("highlights")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"highlight_{frame_idx:06d}.jpg"
    cv2.imwrite(str(path), frame_bgr)
    ss.highlights.append((frame_idx, str(path)))
    ss._last_highlight_frame = frame_idx

# =========================
# Main analysis
# =========================
def analyze_video(video_source=None, use_webcam=False):
    # Reset run stats
    _init_state()

    # Open capture
    if use_webcam:
        # Windows webcams are happiest with CAP_DSHOW
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # Try to set a sane resolution to avoid noisy bands
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        st.error("❌ Could not open video source.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0  # fallback
    frame_idx = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        # skip obviously broken frames (webcam noise)
        if frame is None or frame.size == 0 or frame.std() < 1.0:
            continue

        frame_idx += 1

        # -------- YOLO inference --------
        yres = model(frame, conf=0.25, verbose=False)[0]
        boxes = yres.boxes
        names = model.names

        player_boxes, player_centers, player_teams = [], [], []
        ball_center = None
        player_count, ball_count = 0, 0

        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_id = int(b.cls[0])
                label = names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if label in ("person", "player"):
                    player_count += 1
                    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    hsv_dom = extract_dominant_hsv(crop)
                    team = assign_team_for_player(hsv_dom)
                    player_boxes.append((x1, y1, x2, y2))
                    player_centers.append((cx, cy))
                    player_teams.append(team)

                elif label in ("sports ball", "sports_ball", "ball"):
                    ball_count += 1
                    ball_center = (cx, cy)
                    ss.ball_positions.append(ball_center)

        # -------- tracking / possession --------
        tracked = match_tracks(ss.tracks, player_centers)
        owner_changed = False
        near_player = False
        if ball_center and tracked:
            bx, by = ball_center
            # nearest tracked player to ball
            best_i, best_d = -1, 1e9
            for i, (pid, (cx, cy)) in enumerate(tracked):
                d = np.hypot(bx - cx, by - cy)
                if d < best_d:
                    best_i, best_d = i, d
            owner_pid, (ocx, ocy) = tracked[best_i]
            # map to team by nearest detection
            if player_centers:
                det_idx = int(np.argmin([np.hypot(ocx - px, ocy - py) for (px, py) in player_centers]))
                owner_team = player_teams[det_idx]
                ss.team_possession_frames[owner_team] += 1
                near_player = best_d < 60  # heuristic proximity
                if ss.last_ball_owner_pid is not None and owner_pid != ss.last_ball_owner_pid:
                    ss.pass_count += 1
                    owner_changed = True
                ss.last_ball_owner_pid = owner_pid
                ss.last_ball_owner_team = owner_team

        # -------- speed & distance (ball) --------
        speed = None
        if ball_center is not None:
            if ss.ball_prev_pos is not None:
                dx = ball_center[0] - ss.ball_prev_pos[0]
                dy = ball_center[1] - ss.ball_prev_pos[1]
                dist = float(np.hypot(dx, dy))
                ss.ball_total_distance += dist
                speed = dist * fps  # px/s
                ss.ball_speeds.append(speed)
                ball_speed_metric.metric("Ball Speed", f"{speed:.1f} px/s")
                ball_distance_metric.metric("Ball Distance", f"{ss.ball_total_distance:.0f} px")
            ss.ball_prev_pos = ball_center

        # -------- action recognition + highlights --------
        action = recognize_action(speed, owner_changed, near_player)
        if action:
            ss.last_action = action
            if action in ("Shot",) or (speed and speed > 1500):
                save_highlight(frame, frame_idx)
        last_action_metric.metric("Last Action", ss.last_action)
        highlights_saved_metric.metric("Highlights Saved", len(ss.highlights))

        # -------- draw overlays --------
        # players (Team A red, Team B blue)
        for (x1, y1, x2, y2), team in zip(player_boxes, player_teams):
            color = (0, 0, 255) if team == "Team A" else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, team, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if ball_center:
            cv2.circle(frame, ball_center, 8, (0, 255, 0), -1)
            cv2.putText(frame, "Ball", (ball_center[0] + 10, ball_center[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # status ribbon
        poss_total = sum(ss.team_possession_frames.values()) + 1e-6
        teamA_pct = 100.0 * ss.team_possession_frames["Team A"] / poss_total
        teamB_pct = 100.0 * ss.team_possession_frames["Team B"] / poss_total
        ribbon = f"Possession  A: {teamA_pct:.1f}%  |  B: {teamB_pct:.1f}%   |  Passes: {ss.pass_count}   |  Action: {ss.last_action}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (30, 30, 30), -1)
        cv2.putText(frame, ribbon, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

        # -------- metrics + charts --------
        players_metric.metric("Players (this frame)", player_count)
        balls_metric.metric("Balls (this frame)", ball_count)
        frames_metric.metric("Frames Processed", frame_idx)

        ss.total_players += player_count
        ss.total_balls   += ball_count
        total_players_metric.metric("Total Players Detected", ss.total_players)
        total_balls_metric.metric("Total Balls Detected", ss.total_balls)

        ss.history.append({"frame": frame_idx, "players": player_count, "balls": ball_count})

        # show frame
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # trends
        if ss.history:
            df_hist = pd.DataFrame(ss.history).set_index("frame")
            # keep chart small to avoid lag
            if len(df_hist) > 500:
                df_hist = df_hist.iloc[-500:]
            trends_placeholder.line_chart(df_hist[["players", "balls"]])

        # heatmap
        if ss.ball_positions:
            xs, ys = zip(*ss.ball_positions)
            heatmap, _, _ = np.histogram2d(xs, ys, bins=50)
            fig, ax = plt.subplots()
            ax.imshow(heatmap.T, origin="lower", cmap="hot", alpha=0.9)
            ax.set_title("Ball Position Heatmap")
            ax.axis("off")
            heatmap_placeholder.pyplot(fig)
            plt.close(fig)

        # modest pacing for Streamlit
        time.sleep(0.01)

    cap.release()

# =========================
# Run
# =========================
if start_btn:
    if source_type == "Upload a video" and uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
        tfile.write(uploaded_file.read())
        tfile.flush()
        analyze_video(video_source=tfile.name, use_webcam=False)
    elif source_type == "Use webcam":
        analyze_video(use_webcam=True)
    else:
        st.warning("⚠️ Please upload a video or select webcam before starting.")
