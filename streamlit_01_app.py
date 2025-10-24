import os
import cv2
import numpy as np
import tempfile
import subprocess
import streamlit as st
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# ---- ログ抑制（任意）----
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ------------------------
# ffmpeg で常時正規化（H.264/MP4・720p・30fps・音声なし）
# ------------------------
def ffmpeg_normalize(input_path: str) -> str:
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "scale='min(1280,iw)':-2", "-r", "30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "23",
        "-an",
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out_path

# ------------------------
# 動画メタ情報
# ------------------------
def get_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 0, 0, 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    cap.release()
    return total, fps, w, h

# ------------------------
# フレームを1枚だけ読み込む（オンデマンド）
# ------------------------
def read_frame(path: str, idx: int, max_h=640):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    if h > max_h:
        nh = max_h
        nw = int(w * (max_h / h))
        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return frame

# ------------------------
# Pose 推定（キャッシュ）
# ------------------------
@st.cache_resource
def get_pose(model_complexity: int):
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

@st.cache_data
def detect_landmarks(path, idx, model_complexity):
    pose = get_pose(model_complexity)
    frame = read_frame(path, idx)
    if frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    lm = getattr(results, "pose_landmarks", None)
    if lm is None:
        return None
    arr = np.zeros((33, 4), np.float32)
    for i, p in enumerate(lm.landmark):
        arr[i] = [p.x, p.y, p.z, p.visibility]
    return arr

def draw_skeleton(frame, arr, color=(255,0,0)):
    if arr is None:
        return frame
    lm_list = landmark_pb2.NormalizedLandmarkList(
        landmark=[
            landmark_pb2.NormalizedLandmark(x=float(x), y=float(y), z=float(z), visibility=float(v))
            for (x, y, z, v) in arr
        ]
    )
    spec = mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
    mp_drawing.draw_landmarks(frame, lm_list, mp_pose.POSE_CONNECTIONS, spec, spec)
    return frame

# ------------------------
# UI
# ------------------------
st.set_page_config(layout="wide", page_title="バレエフォーム比較AI")
st.title("💃 バレエフォーム比較AI")

st.markdown("""
### ご利用方法
1. 比較したい動画を **2本アップロード**（順に 青 → 赤）
2. **解析を開始** を押すと、AIが自動で骨格を検出して比較します。

> ℹ️ 変換や設定は不要です（アプリ側で **H.264/MP4・720p・30fps** に自動調整）。  
> 🎯 最も正確に解析できるのは **1人の全身が映っている動画** です。  
> ⏱️ **長さの目安**：**10〜60秒/本** をおすすめ、**上限の目安は〜2分/本**（それ以上は体感が重くなります）。
""")

model_complexity = st.selectbox(
    "解析精度の設定",
    options=[(0, "低（高速）"), (1, "中（バランス）"), (2, "高（精密）")],
    format_func=lambda x: x[1], index=1
)[0]

# 手入力・スライダーの状態をセッションに保持
st.session_state.setdefault("idx1", 0)
st.session_state.setdefault("idx2", 0)

with st.form("upload"):
    files = st.file_uploader("動画を2本選択（青→赤）", type=["mp4", "mov", "avi"], accept_multiple_files=True)
    submitted = st.form_submit_button("解析を開始")

if submitted:
    if not files or len(files) < 2:
        st.warning("2本の動画を選んでください。")
    else:
        paths, metas = [], []
        for f in files[:2]:
            # 一時保存 → 正規化
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1])
            temp.write(f.getvalue())
            temp.flush()
            norm = ffmpeg_normalize(temp.name)
            try:
                os.remove(temp.name)
            except Exception:
                pass
            paths.append(norm)
            metas.append(get_video_info(norm))
        st.session_state["paths"] = paths
        st.session_state["metas"] = metas
        # 初期位置を0に戻す
        st.session_state.idx1 = 0
        st.session_state.idx2 = 0

if "paths" in st.session_state:
    p1, p2 = st.session_state["paths"]
    (t1, fps1, w1, h1), (t2, fps2, w2, h2) = st.session_state["metas"]

    st.subheader("🎬 骨格比較（オンデマンド）")
    col1, col2 = st.columns(2)
    disp_w = st.slider("表示サイズ(px)", 200, 900, 360, 10)

    # 体感を軽くするため、スライダーは15fps相当で刻む
    step1 = max(1, int((fps1 or 30) // 15))
    step2 = max(1, int((fps2 or 30) // 15))
    max1 = max(0, t1 - 1)
    max2 = max(0, t2 - 1)

    with col1:
        st.markdown("**青動画**")
        # 手入力（1刻み）
        num1 = st.number_input("フレーム番号（手入力）", min_value=0, max_value=max1, value=int(st.session_state.idx1), step=1, key="num1_input")
        # スライダー（軽い刻み）
        sld1 = st.slider("フレーム位置（スライダー）", 0, max1, int(num1), step=step1, key="sld1")
        # 最終値を統一
        st.session_state.idx1 = int(sld1)

        f1 = read_frame(p1, st.session_state.idx1)
        lm1 = detect_landmarks(p1, st.session_state.idx1, model_complexity)
        if f1 is not None:
            draw_skeleton(f1, lm1, (255,0,0))
            st.image(f1, channels="BGR", width=disp_w)
            cap = f"フレーム {st.session_state.idx1+1}/{max(1,t1)}"
            if lm1 is None:
                cap += "（このフレームは未検出）"
            st.caption(cap)
        else:
            st.error("フレームの読み込みに失敗しました。")

    with col2:
        st.markdown("**赤動画**")
        num2 = st.number_input("フレーム番号（手入力） ", min_value=0, max_value=max2, value=int(st.session_state.idx2), step=1, key="num2_input")
        sld2 = st.slider("フレーム位置（スライダー） ", 0, max2, int(num2), step=step2, key="sld2")
        st.session_state.idx2 = int(sld2)

        f2 = read_frame(p2, st.session_state.idx2)
        lm2 = detect_landmarks(p2, st.session_state.idx2, model_complexity)
        if f2 is not None:
            draw_skeleton(f2, lm2, (0,0,255))
            st.image(f2, channels="BGR", width=disp_w)
            cap = f"フレーム {st.session_state.idx2+1}/{max(1,t2)}"
            if lm2 is None:
                cap += "（このフレームは未検出）"
            st.caption(cap)
        else:
            st.error("フレームの読み込みに失敗しました。")

else:
    st.info("2本の動画を選んで『解析を開始』を押してください。")
