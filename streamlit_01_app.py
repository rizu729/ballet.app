import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import tempfile

# --- MediaPipe Pose 初期化 ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# --- 骨格描画関数 ---
def draw_skeleton_on_frame(frame, results_pose_landmarks, line_color=(255, 0, 0)):
    if results_pose_landmarks:
        custom_drawing_spec = mp_drawing.DrawingSpec(color=line_color, thickness=2, circle_radius=2)
        custom_connection_spec = mp_drawing.DrawingSpec(color=line_color, thickness=2, circle_radius=2)
        mp_drawing.draw_landmarks(
            frame,
            results_pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=custom_drawing_spec,
            connection_drawing_spec=custom_connection_spec
        )
    return frame


# --- 動画からフレームと骨格を抽出 ---
@st.cache_data(show_spinner=False)
def extract_frames_and_skeletons(uploaded_file, model_complexity=1, max_frame_height=640):
    if uploaded_file is None:
        return [], [], 0, 0, 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_file_path = tmp.name

    try:
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            st.error(f"Error: {uploaded_file.name} を開けませんでした。")
            return [], [], 0, 0, 0

        frames, landmarks_results = [], []
        original_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        # リサイズ設定
        new_height = min(original_frame_height, max_frame_height)
        new_width = int(original_frame_width * (new_height / original_frame_height))

        progress_text = f"処理中: {uploaded_file.name} (高さ {new_height}px)"
        my_bar = st.progress(0, text=progress_text)
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                frames.append(frame)
                landmarks_results.append(results.pose_landmarks)

                frame_count += 1
                if total_frames_in_video > 0:
                    progress_percentage = min(100, int(frame_count / total_frames_in_video * 100))
                    my_bar.progress(progress_percentage, text=f"{progress_text} {progress_percentage}%")

        my_bar.empty()
        cap.release()
        st.success(f"{uploaded_file.name} の処理が完了しました（{len(frames)}フレーム）")
        return frames, landmarks_results, new_width, new_height, fps

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- Streamlit アプリ設定 ---
st.set_page_config(layout="wide", page_title="バレエフォーム比較AI")
st.title("💃 バレエフォーム比較AI")

# --- 使い方説明 ---
st.markdown("""
### 📖 使い方
1. 下のアップロードボタンから比較したい動画を2つ選択してください。  
   - 1つ目の動画：青い骨格  
   - 2つ目の動画：赤い骨格  
2. 動画をアップロードすると自動で骨格推定が始まります。処理には少し時間がかかります。  
3. フレーム番号をスライダーまたは手入力で選び、2つのフォームを横並びで比較できます。  
4. 違いを見ながらフォーム改善や練習の分析に役立ててください。

⚠️ **注意**: 長時間または高解像度の動画は処理が重くなる可能性があります。  
推奨：30秒以内・高さ640px以内の動画。
""")

# --- 動画アップロード ---
uploaded_file1 = st.file_uploader("🎥 1つ目の動画をアップロード（青骨格）", type=['mp4', 'mov', 'avi'])
uploaded_file2 = st.file_uploader("🎥 2つ目の動画をアップロード（赤骨格）", type=['mp4', 'mov', 'avi'])

# --- モデルの複雑さ ---
model_complexity_option = st.selectbox(
    "ポーズ推定モデルの精度/速度",
    options=[(0, "低（高速）"), (1, "中（バランス）"), (2, "高（精密）")],
    format_func=lambda x: x[1],
    index=1
)[0]

MAX_FRAME_HEIGHT = 640

# --- セッション初期化 ---
for i in [1, 2]:
    if f'frames{i}' not in st.session_state:
        st.session_state[f'frames{i}'] = []
        st.session_state[f'landmarks{i}'] = []
        st.session_state[f'w{i}'], st.session_state[f'h{i}'], st.session_state[f'fps{i}'] = 0, 0, 0
        st.session_state[f'frame_index{i}'] = 0

# --- 動画処理 ---
if uploaded_file1 and (not st.session_state.frames1 or uploaded_file1.name != st.session_state.get('uploaded_file1_name')):
    with st.spinner("1つ目の動画を処理中..."):
        st.session_state.frames1, st.session_state.landmarks1, st.session_state.w1, st.session_state.h1, st.session_state.fps1 = extract_frames_and_skeletons(uploaded_file1, model_complexity_option, MAX_FRAME_HEIGHT)
    st.session_state.uploaded_file1_name = uploaded_file1.name

if uploaded_file2 and (not st.session_state.frames2 or uploaded_file2.name != st.session_state.get('uploaded_file2_name')):
    with st.spinner("2つ目の動画を処理中..."):
        st.session_state.frames2, st.session_state.landmarks2, st.session_state.w2, st.session_state.h2, st.session_state.fps2 = extract_frames_and_skeletons(uploaded_file2, model_complexity_option, MAX_FRAME_HEIGHT)
    st.session_state.uploaded_file2_name = uploaded_file2.name


# --- 比較表示 ---
if st.session_state.frames1 and st.session_state.frames2:
    st.subheader("🎬 フレーム選択で骨格比較")

    display_image_width = st.slider("表示サイズ（px）", 200, 800, 350, 10)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("青骨格動画")
        st.session_state.frame_index1 = st.number_input("フレーム番号入力", min_value=0, max_value=len(st.session_state.frames1)-1, value=st.session_state.frame_index1, step=1, key="input1")
        st.session_state.frame_index1 = st.slider("フレーム選択", 0, len(st.session_state.frames1)-1, st.session_state.frame_index1, 1, key="slider1")
        current_frame1 = st.session_state.frames1[st.session_state.frame_index1].copy()
        draw_skeleton_on_frame(current_frame1, st.session_state.landmarks1[st.session_state.frame_index1], (255, 0, 0))
        st.image(current_frame1, channels="BGR", caption=f"フレーム {st.session_state.frame_index1}", width=display_image_width)

    with col2:
        st.subheader("赤骨格動画")
        st.session_state.frame_index2 = st.number_input("フレーム番号入力", min_value=0, max_value=len(st.session_state.frames2)-1, value=st.session_state.frame_index2, step=1, key="input2")
        st.session_state.frame_index2 = st.slider("フレーム選択", 0, len(st.session_state.frames2)-1, st.session_state.frame_index2, 1, key="slider2")
        current_frame2 = st.session_state.frames2[st.session_state.frame_index2].copy()
        draw_skeleton_on_frame(current_frame2, st.session_state.landmarks2[st.session_state.frame_index2], (0, 0, 255))
        st.image(current_frame2, channels="BGR", caption=f"フレーム {st.session_state.frame_index2}", width=display_image_width)

elif uploaded_file1 or uploaded_file2:
    st.info("両方の動画をアップロードすると比較が開始されます。")
else:
    st.info("動画をアップロードして比較を開始してください。")
