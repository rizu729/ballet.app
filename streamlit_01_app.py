%%writefile app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

# --- MediaPipe Pose ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- 骨格描画 ---
def draw_skeleton_on_frame(frame, results_pose_landmarks, line_color=(255,0,0)):
    if results_pose_landmarks:
        drawing_spec = mp_drawing.DrawingSpec(color=line_color, thickness=2, circle_radius=2)
        mp_drawing.draw_landmarks(
            frame,
            results_pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )
    return frame

# --- フレームと骨格抽出 ---
@st.cache_data(show_spinner=False)
def extract_frames_and_skeletons(uploaded_file, model_complexity=1, max_frame_height=640):
    if uploaded_file is None:
        return [], [], 0, 0, 0

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error(f"動画を開けませんでした: {uploaded_file.name}")
        return [], [], 0, 0, 0

    frames = []
    landmarks_results = []
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    new_w, new_h = orig_w, orig_h
    if orig_h > max_frame_height:
        new_h = max_frame_height
        new_w = int(orig_w * (max_frame_height / orig_h))
        if new_w == 0: new_w = 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_text = f"処理中: {uploaded_file.name}"
    my_bar = st.progress(0, text=progress_text)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if orig_h > max_frame_height:
                frame = cv2.resize(frame, (new_w, new_h))
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            frames.append(frame)
            landmarks_results.append(results.pose_landmarks)
            my_bar.progress(min(100, int((frame_idx+1)/total_frames*100)))

    cap.release()
    my_bar.empty()
    os.remove(tmp_path)
    return frames, landmarks_results, new_w, new_h, fps

# --- Streamlit Layout ---
st.set_page_config(layout="wide", page_title="バレエフォーム比較AI")
st.title("バレエフォーム比較AI")

st.markdown("""
### 📖 使い方
1. 下のボタンから2つの動画をアップロードしてください。
2. 動画が処理されると、フレームごとの骨格比較ができます。

⚠️ **注意**: 長時間や高解像度の動画はアプリが重くなったりクラッシュする原因になります。  
推奨: 30秒以内・高さ最大640px以内の動画
""")

# --- アップロード ---
uploaded_file1 = st.file_uploader("動画1 (青い骨格)", type=['mp4','mov','avi'], key="upload1")
uploaded_file2 = st.file_uploader("動画2 (赤い骨格)", type=['mp4','mov','avi'], key="upload2")

model_complexity_option = st.selectbox(
    "ポーズ推定モデルの精度/速度",
    options=[(0,"低(高速)"), (1,"中(バランス)"), (2,"高(低速)")],
    format_func=lambda x:x[1],
    index=1
)[0]

MAX_FRAME_HEIGHT = 640

# --- セッション初期化 ---
for i in [1,2]:
    if f'frames{i}' not in st.session_state:
        st.session_state[f'frames{i}'] = []
        st.session_state[f'landmarks{i}'] = []
        st.session_state[f'w{i}'] = st.session_state[f'h{i}'] = st.session_state[f'fps{i}'] = 0
        st.session_state[f'frame_index{i}'] = 0

# --- 動画処理 ---
if uploaded_file1:
    st.session_state.frames1, st.session_state.landmarks1, st.session_state.w1, st.session_state.h1, st.session_state.fps1 = extract_frames_and_skeletons(uploaded_file1, model_complexity_option, MAX_FRAME_HEIGHT)
    st.session_state.frame_index1 = 0

if uploaded_file2:
    st.session_state.frames2, st.session_state.landmarks2, st.session_state.w2, st.session_state.h2, st.session_state.fps2 = extract_frames_and_skeletons(uploaded_file2, model_complexity_option, MAX_FRAME_HEIGHT)
    st.session_state.frame_index2 = 0

# --- 比較表示 ---
if st.session_state.frames1 and st.session_state.frames2:
    st.subheader("フレーム選択で骨格比較")

    display_width = st.slider("表示幅(px)", 100, 800, 350, 10)

    col1, col2 = st.columns(2)
    for i, col in enumerate([col1,col2], start=1):
        with col:
            st.subheader(f"動画{i}")
            max_frame = len(st.session_state[f'frames{i}']) - 1
            idx = st.number_input(f"フレーム{i}番号", 0, max_frame, st.session_state[f'frame_index{i}'], step=1, key=f'num_input{i}')
            st.session_state[f'frame_index{i}'] = idx
            frame = st.session_state[f'frames{i}'][idx].copy()
            color = (255,0,0) if i==1 else (0,0,255)
            draw_skeleton_on_frame(frame, st.session_state[f'landmarks{i}'][idx], line_color=color)
            st.image(frame, channels="BGR", width=display_width)
            st.info(f"フレーム {idx+1} / {len(st.session_state[f'frames{i}'])}")

elif uploaded_file1 or uploaded_file2:
    st.info("両方の動画をアップロードすると比較できます。")

