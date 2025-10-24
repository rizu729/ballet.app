import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os # 一時ファイル保存のため

# --- MediaPipe Poseの初期化 ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- 骨格描画用のヘルパー関数 ---
def draw_skeleton_on_frame(frame, results_pose_landmarks, line_color=(255, 0, 0)):
    """
    OpenCVフレーム上にMediaPipeの骨格を描画します。
    """
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

# --- 動画からフレームと骨格データを抽出する関数 ---
@st.cache_data(show_spinner=False) # Streamlitのキャッシュ機能で、動画が同じなら再計算を避ける
def extract_frames_and_skeletons(uploaded_file, model_complexity=1, max_frame_height=640):
    """
    アップロードされた動画からフレームとMediaPipeのポーズランドマーク（検出結果オブジェクト）を抽出します。
    この処理はリソースを大量に消費する可能性があります。
    max_frame_heightでフレームの最大高さを指定し、アスペクト比を維持して自動的にリサイズします。
    """
    if uploaded_file is None:
        return [], [], 0, 0, 0 # frames, landmarks_results, width, height, fps

    temp_dir = "/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            st.error(f"Error: Could not open video file: {uploaded_file.name}")
            return [], [], 0, 0, 0

        frames = []
        landmarks_results = []
        
        original_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # リサイズが必要な場合の新しいサイズを計算 (高さ基準)
        new_width = original_frame_width
        new_height = original_frame_height
        if original_frame_height > max_frame_height:
            new_height = max_frame_height
            new_width = int(original_frame_width * (max_frame_height / original_frame_height))
            if new_width == 0: new_width = 1 
        
        progress_text = f"処理中: {uploaded_file.name} (リサイズ高さ: {new_height}px)..."
        my_bar = st.progress(0, text=progress_text)
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # フレームをリサイズ
                if original_frame_height > max_frame_height:
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

                # MediaPipe処理用にRGBに変換
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
        st.success(f"Successfully processed {uploaded_file.name}. Total frames: {len(frames)}")
        return frames, landmarks_results, new_width, new_height, fps
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- Streamlit アプリケーションのレイアウト ---
st.set_page_config(layout="wide", page_title="バレエフォーム比較AI")
st.title("バレエフォーム比較AI")

st.write("2つの動画をアップロードして、それぞれ好きなフレームで骨格を比較できます。")
st.warning("⚠️ **重要**: 長い動画や高解像度の動画は、アプリがクラッシュする原因となります。アップロードされた動画は自動的に**最大高さ640ピクセル**にリサイズされますが、**短時間（推奨: 30秒以内）の動画**をご利用ください。")
st.info("処理中にアプリがフリーズしたように見えても、バックグラウンドで処理が進んでいる場合があります。しばらくお待ちください。")

# ここにメッセージを移動し、テキストを更新
initial_upload_message = "比較を開始するには、画面左上のサイドバーから動画ファイルをアップロードしてください。"


# --- ファイルアップロード ---
st.sidebar.header("動画をアップロード")
uploaded_file1 = st.sidebar.file_uploader("1つ目の動画をアップロード (青い骨格)", type=['mp4', 'mov', 'avi'])
uploaded_file2 = st.sidebar.file_uploader("2つ目の動画をアップロード (赤い骨格)", type=['mp4', 'mov', 'avi'])

# モデルの複雑さ選択 (サイドバーに移動)
model_complexity_option = st.sidebar.selectbox(
    "ポーズ推定モデルの精度/速度",
    options=[(0, "低 (高速)"), (1, "中 (バランス)"), (2, "高 (低速)")],
    format_func=lambda x: x[1],
    index=1,
    help="モデルの複雑さを下げると、処理速度が向上し、メモリ消費が抑えられますが、ポーズ推定精度が低下する可能性があります。"
)[0]

# 自動リサイズするフレームの最大高さ
MAX_FRAME_HEIGHT = 640

# 状態管理のためのキー
if 'frames1' not in st.session_state:
    st.session_state.frames1 = []
    st.session_state.landmarks1 = []
    st.session_state.w1, st.session_state.h1, st.session_state.fps1 = 0, 0, 0
    st.session_state.frame_index1 = 0

if 'frames2' not in st.session_state:
    st.session_state.frames2 = []
    st.session_state.landmarks2 = []
    st.session_state.w2, st.session_state.h2, st.session_state.fps2 = 0, 0, 0
    st.session_state.frame_index2 = 0

# ファイルがアップロードされたら処理を実行し、セッションステートに保存
if uploaded_file1 and (not st.session_state.frames1 or uploaded_file1.name != st.session_state.get('uploaded_file1_name') or st.session_state.get('model_complexity_applied_1') != model_complexity_option):
    with st.spinner(f"1つ目の動画を処理中..."):
        st.session_state.frames1, st.session_state.landmarks1, st.session_state.w1, st.session_state.h1, st.session_state.fps1 = extract_frames_and_skeletons(uploaded_file1, model_complexity=model_complexity_option, max_frame_height=MAX_FRAME_HEIGHT)
    st.session_state.uploaded_file1_name = uploaded_file1.name
    st.session_state.model_complexity_applied_1 = model_complexity_option
    st.session_state.frame_index1 = 0

if uploaded_file2 and (not st.session_state.frames2 or uploaded_file2.name != st.session_state.get('uploaded_file2_name') or st.session_state.get('model_complexity_applied_2') != model_complexity_option):
    with st.spinner(f"2つ目の動画を処理中..."):
        st.session_state.frames2, st.session_state.landmarks2, st.session_state.w2, st.session_state.h2, st.session_state.fps2 = extract_frames_and_skeletons(uploaded_file2, model_complexity=model_complexity_option, max_frame_height=MAX_FRAME_HEIGHT)
    st.session_state.uploaded_file2_name = uploaded_file2.name
    st.session_state.model_complexity_applied_2 = model_complexity_option
    st.session_state.frame_index2 = 0

# 両方の動画が処理されたら比較UIを表示
if st.session_state.frames1 and st.session_state.frames2:
    
    st.subheader("フレーム選択で骨格比較")
    
    # 表示画像幅のスライダー
    display_image_width = st.slider(
        "表示画像幅を調整 (ピクセル)",
        min_value=100,
        max_value=800,
        value=min(350, st.session_state.w1 if st.session_state.w1 > 0 else 350),
        step=10,
        help="比較画像の表示幅を調整します。値を小さくすると、スマートフォンでスクロールせずに表示できる可能性が高まりますが、画像が小さくなります。動画は自動的に最大高さ640pxにリサイズされます。"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1つ目の動画 (青)")
        st.session_state.frame_index1 = st.slider(
            "1つ目の動画のフレームを選択",
            min_value=0,
            max_value=len(st.session_state.frames1) - 1,
            value=st.session_state.frame_index1,
            step=1,
            key='slider1',
            help="1つ目の動画のフレームを選択します。"
        )
        current_frame1 = st.session_state.frames1[st.session_state.frame_index1].copy()
        draw_skeleton_on_frame(current_frame1, st.session_state.landmarks1[st.session_state.frame_index1], line_color=(255, 0, 0))
        st.image(current_frame1, channels="BGR", caption=f"1つ目の動画 - フレーム {st.session_state.frame_index1}", width=display_image_width)
        st.info(f"1つ目の動画のフレーム: {st.session_state.frame_index1+1} / {len(st.session_state.frames1)}")


    with col2:
        st.subheader("2つ目の動画 (赤)")
        st.session_state.frame_index2 = st.slider(
            "2つ目の動画のフレームを選択",
            min_value=0,
            max_value=len(st.session_state.frames2) - 1,
            value=st.session_state.frame_index2,
            step=1,
            key='slider2',
            help="2つ目の動画のフレームを選択します。"
        )
        current_frame2 = st.session_state.frames2[st.session_state.frame_index2].copy()
        draw_skeleton_on_frame(current_frame2, st.session_state.landmarks2[st.session_state.frame_index2], line_color=(0, 0, 255))
        st.image(current_frame2, channels="BGR", caption=f"2つ目の動画 - フレーム {st.session_state.frame_index2}", width=display_image_width)
        st.info(f"2つ目の動画のフレーム: {st.session_state.frame_index2+1} / {len(st.session_state.frames2)}")

elif st.session_state.frames1 or st.session_state.frames2:
    st.info("両方の動画をアップロードすると比較が開始されます。")
else:
    st.info(initial_upload_message)
