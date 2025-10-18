import os
import io
import tempfile
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

# ----（任意）うるさい警告を抑制 ----
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

# ---- MediaPipe 初期化 ----
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ---- メモリ退避用ラッパ ----
class MemFile:
    """StreamlitのUploadedFileから取り出したバイト列を、getbuffer()互換で扱うための簡易クラス"""
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data
    def getbuffer(self):
        return self._data

# ---- 骨格描画 ----
def draw_skeleton_on_frame(frame, results_pose_landmarks, line_color=(255, 0, 0)):
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

# ---- 動画→フレーム/骨格抽出 ----
@st.cache_data(show_spinner=False)
def extract_frames_and_skeletons(uploaded_file, model_complexity=1, max_frame_height=640):
    """
    uploaded_file: getbuffer() を持つオブジェクト（Streamlit UploadedFile でも MemFile でもOK）
    """
    if uploaded_file is None:
        return [], [], 0, 0, 0

    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name

    try:
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return [], [], 0, 0, 0

        frames, landmarks = [], []
        ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # リサイズ設定
        nw, nh = ow, oh
        if oh > 0 and oh > max_frame_height:
            nh = max_frame_height
            nw = max(1, int(ow * (max_frame_height / oh)))

        bar = st.progress(0, text=f"処理中: {uploaded_file.name}") if total > 0 else None

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
                if oh > 0 and oh > max_frame_height:
                    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                frames.append(frame)
                landmarks.append(results.pose_landmarks)

                idx += 1
                if bar and total > 0:
                    pct = min(100, int(idx / total * 100))
                    bar.progress(pct, text=f"処理中: {uploaded_file.name} {pct}%")

        if bar:
            bar.empty()
        cap.release()
        return frames, landmarks, (nw or ow), (nh or oh), fps
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

# ---- UI 設定 ----
st.set_page_config(layout="wide", page_title="バレエフォーム比較AI")
st.title("💃 バレエフォーム比較AI")

st.markdown("""
### 📖 使い方
1. 下のアップロード欄で **2本まとめて** 動画を選択してください（順に **青 → 赤** として扱います）。
2. **解析を開始** を押すと骨格推定がはじまり、フレームごとに横並びで比較できます。
3. フレーム番号は **手入力** と **スライダー** の両方で細かく調整できます。

⚠️ **推奨**：10〜20秒・720p以下・**H.264(MP4)**。  
iPhone標準の高圧縮 **.mov(HEVC)** は失敗しやすいです。**Wi-Fi** 推奨、アップロード中は画面を閉じないでください。
""")

# ---- モデル複雑さ ----
model_complexity_option = st.selectbox(
    "ポーズ推定モデルの精度/速度",
    options=[(0, "低（高速）"), (1, "中（バランス）"), (2, "高（精密）")],
    format_func=lambda x: x[1],
    index=1
)[0]

MAX_FRAME_HEIGHT = 640

# ---- セッション初期化 ----
for i in [1, 2]:
    st.session_state.setdefault(f'frames{i}', [])
    st.session_state.setdefault(f'landmarks{i}', [])
    st.session_state.setdefault(f'w{i}', 0)
    st.session_state.setdefault(f'h{i}', 0)
    st.session_state.setdefault(f'fps{i}', 0)
    st.session_state.setdefault(f'frame_index{i}', 0)
st.session_state.setdefault("last_files_sig", None)
st.session_state.setdefault("filebufs", [])

# ---- 同時アップロード + 送信ボタン + メモリ退避（ここが最重要）----
with st.form(key="upload_form", clear_on_submit=False):
    files = st.file_uploader(
        "動画を **2本まとめて** アップロード（順に 青 → 赤 として扱います）",
        type=['mp4', 'mov', 'avi'],
        accept_multiple_files=True
    )
    st.info("⚠️ 推奨: 10〜20秒以内・720p以下・H.264(MP4)。iPhoneの高圧縮MOV(HEVC)は失敗しやすいです。Wi-Fiで、アップロード中は画面を閉じないでください。")
    submitted = st.form_submit_button("解析を開始")

# 新しい選択が来たら生バイトをセッションに退避（URL失効対策）
if files:
    sig = tuple((f.name, f.size) for f in files[:2])
    if st.session_state["last_files_sig"] != sig:
        st.session_state["last_files_sig"] = sig
        st.session_state["filebufs"] = []
        for f in files[:2]:
            st.session_state["filebufs"].append({"name": f.name, "bytes": f.getvalue()})
        # 以前の結果をクリア
        for i in [1, 2]:
            st.session_state[f'frames{i}'] = []
            st.session_state[f'landmarks{i}'] = []
            st.session_state[f'w{i}'] = st.session_state[f'h{i}'] = st.session_state[f'fps{i}'] = 0
            st.session_state[f'frame_index{i}'] = 0

# フォーム送信で確定
uploaded_file1 = uploaded_file2 = None
bufs = st.session_state.get("filebufs", [])
if submitted and len(bufs) >= 2:
    uploaded_file1 = MemFile(bufs[0]["name"], bufs[0]["bytes"])
    uploaded_file2 = MemFile(bufs[1]["name"], bufs[1]["bytes"])
elif submitted and len(bufs) < 2:
    st.warning("2本の動画を選んでから『解析を開始』を押してください。")

# ---- 動画処理（送信時にだけ実行）----
if uploaded_file1 is not None and not st.session_state.frames1:
    with st.spinner("1つ目の動画を処理中..."):
        st.session_state.frames1, st.session_state.landmarks1, st.session_state.w1, st.session_state.h1, st.session_state.fps1 = extract_frames_and_skeletons(
            uploaded_file1, model_complexity=model_complexity_option, max_frame_height=MAX_FRAME_HEIGHT
        )
    st.session_state.frame_index1 = 0

if uploaded_file2 is not None and not st.session_state.frames2:
    with st.spinner("2つ目の動画を処理中..."):
        st.session_state.frames2, st.session_state.landmarks2, st.session_state.w2, st.session_state.h2, st.session_state.fps2 = extract_frames_and_skeletons(
            uploaded_file2, model_complexity=model_complexity_option, max_frame_height=MAX_FRAME_HEIGHT
        )
    st.session_state.frame_index2 = 0

# ---- 比較UI ----
if st.session_state.frames1 and st.session_state.frames2:
    st.subheader("🎬 フレーム選択で骨格比較")
    display_w = st.slider("表示画像幅（px）", 200, 800, 350, 10)

    col1, col2 = st.columns(2)
    # 左（青）
    with col1:
        st.subheader("青骨格")
        max1 = len(st.session_state.frames1) - 1
        idx1 = st.number_input("フレーム番号（手入力）", 0, max1, st.session_state.frame_index1, step=1, key="num1")
        idx1 = st.slider("フレーム（スライダー）", 0, max1, idx1, step=1, key="sld1")
        st.session_state.frame_index1 = idx1

        f1 = st.session_state.frames1[idx1].copy()
        draw_skeleton_on_frame(f1, st.session_state.landmarks1[idx1], (255, 0, 0))
        st.image(f1, channels="BGR", width=display_w)
        st.caption(f"フレーム {idx1+1} / {max1+1}")

    # 右（赤）
    with col2:
        st.subheader("赤骨格")
        max2 = len(st.session_state.frames2) - 1
        idx2 = st.number_input("フレーム番号（手入力） ", 0, max2, st.session_state.frame_index2, step=1, key="num2")
        idx2 = st.slider("フレーム（スライダー） ", 0, max2, idx2, step=1, key="sld2")
        st.session_state.frame_index2 = idx2

        f2 = st.session_state.frames2[idx2].copy()
        draw_skeleton_on_frame(f2, st.session_state.landmarks2[idx2], (0, 0, 255))
        st.image(f2, channels="BGR", width=display_w)
        st.caption(f"フレーム {idx2+1} / {max2+1}")

elif submitted and (not st.session_state.frames1 or not st.session_state.frames2):
    st.error("動画の読み込み・解析に失敗しました。コーデックをH.264/MP4にする、長さを短くする（10〜20秒）、解像度を下げる（≤720p）などをお試しください。")

else:
    st.info("2本の動画を選んで『解析を開始』を押すと比較できます。")
