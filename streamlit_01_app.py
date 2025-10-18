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
        spec = mp_drawing.DrawingSpec(color=line_color, thickness=2, circle_radius=2)
        mp_drawing.draw_landmarks(
            frame,
            results_pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=spec,
            connection_drawing_spec=spec
        )
    return frame

# --- 動画→フレーム/骨格抽出 ---
@st.cache_data(show_spinner=False)
def extract_frames_and_skeletons(uploaded_file, model_complexity=1, max_frame_height=640):
    if uploaded_file is None:
        return [], [], 0, 0, 0

    # 一時ファイル（環境依存を避ける）
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_file_path = tmp.name

    try:
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            return [], [], 0, 0, 0

        frames, landmarks_results = [], []
        ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # リサイズ設定
        nw, nh = ow, oh
        if oh > max_frame_height and oh > 0:
            nh = max_frame_height
            nw = max(1, int(ow * (max_frame_height / oh)))

        # 進捗バー（フレーム数が取れない動画もある）
        if total > 0:
            bar = st.progress(0, text=f"処理中: {uploaded_file.name}")
        else:
            bar = None

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if oh > max_frame_height and oh > 0:
                    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                frames.append(frame)
                landmarks_results.append(results.pose_landmarks)

                idx += 1
                if bar and total > 0:
                    bar.progress(min(100, int(idx / total * 100)), text=f"処理中: {uploaded_file.name} {min(100, int(idx / total * 100))}%")

        if bar:
            bar.empty()
        cap.release()
        return frames, landmarks_results, (nw or ow), (nh or oh), fps
    finally:
        # 一時ファイル掃除
        try:
            os.remove(temp_file_path)
        except Exception:
            pass

# --- UI ---
st.set_page_config(layout="wide", page_title="バレエフォーム比較AI")
st.title("バレエフォーム比較AI")

st.markdown("""
### 📖 使い方
1. 下のボタンから **2つの動画** をアップロードしてください（1つ目=青骨格、2つ目=赤骨格）。
2. 読み込み後、フレーム番号を **手入力** または **スライダー** で選ぶと、同フレームの骨格が並んで表示されます。

⚠️ **注意**: 長時間/高解像度の動画は処理落ちの原因になります。  
推奨: **30秒以内**・**高さ最大640px** 相当の解像度
""")

uploaded_file1 = st.file_uploader("1つ目の動画（青骨格）をアップロード", type=['mp4','mov','avi'], key="upload1")
uploaded_file2 = st.file_uploader("2つ目の動画（赤骨格）をアップロード", type=['mp4','mov','avi'], key="upload2")

model_complexity = st.selectbox(
    "ポーズ推定モデルの精度/速度",
    options=[(0,"低(高速)"),(1,"中(バランス)"),(2,"高(低速)")],
    index=1,
    format_func=lambda x: x[1]
)[0]

MAX_H = 640

# セッション初期化
for i in [1,2]:
    st.session_state.setdefault(f'frames{i}', [])
    st.session_state.setdefault(f'landmarks{i}', [])
    st.session_state.setdefault(f'w{i}', 0)
    st.session_state.setdefault(f'h{i}', 0)
    st.session_state.setdefault(f'fps{i}', 0)
    st.session_state.setdefault(f'frame_index{i}', 0)

# 動画処理（同名再アップロード時の再処理を避けるには名前チェック等を追加可能）
if uploaded_file1:
    st.session_state.frames1, st.session_state.landmarks1, st.session_state.w1, st.session_state.h1, st.session_state.fps1 = extract_frames_and_skeletons(uploaded_file1, model_complexity, MAX_H)
    st.session_state.frame_index1 = 0
if uploaded_file2:
    st.session_state.frames2, st.session_state.landmarks2, st.session_state.w2, st.session_state.h2, st.session_state.fps2 = extract_frames_and_skeletons(uploaded_file2, model_complexity, MAX_H)
    st.session_state.frame_index2 = 0

# フレーム有無チェック
if (uploaded_file1 and not st.session_state.frames1) or (uploaded_file2 and not st.session_state.frames2):
    st.error("動画のフレームを取得できませんでした。コーデック/解像度を下げて再度お試しください。")

# 比較UI
if st.session_state.frames1 and st.session_state.frames2:
    st.subheader("フレーム選択で骨格比較")
    # 画像幅（スマホ幅も考慮）
    default_width = 350 if st.session_state.w1 == 0 else min(350, st.session_state.w1)
    display_w = st.slider("表示画像幅(px)", 120, 800, default_width, 10)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1つ目の動画（青）")
        max1 = len(st.session_state.frames1) - 1
        idx1 = st.number_input("フレーム番号（手入力）", 0, max1, st.session_state.frame_index1, step=1, key="num1")
        idx1 = st.slider("フレーム（スライダー）", 0, max1, idx1, step=1, key="sld1")
        st.session_state.frame_index1 = idx1

        frame1 = st.session_state.frames1[idx1].copy()
        draw_skeleton_on_frame(frame1, st.session_state.landmarks1[idx1], (255,0,0))
        st.image(frame1, channels="BGR", width=display_w)
        st.caption(f"フレーム {idx1+1} / {max1+1}")

    with col2:
        st.subheader("2つ目の動画（赤）")
        max2 = len(st.session_state.frames2) - 1
        idx2 = st.number_input("フレーム番号（手入力） ", 0, max2, st.session_state.frame_index2, step=1, key="num2")
        idx2 = st.slider("フレーム（スライダー） ", 0, max2, idx2, step=1, key="sld2")
        st.session_state.frame_index2 = idx2

        frame2 = st.session_state.frames2[idx2].copy()
        draw_skeleton_on_frame(frame2, st.session_state.landmarks2[idx2], (0,0,255))
        st.image(frame2, channels="BGR", width=display_w)
        st.caption(f"フレーム {idx2+1} / {max2+1}")

elif uploaded_file1 or uploaded_file2:
    st.info("両方の動画をアップロードすると比較できます。")

