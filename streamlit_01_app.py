# streamlit_iphone_optimized_app.py
# ------------------------------------------------------------
# iPhone最適化UI / 2本比較 / フレーム操作（±ボタン＋スライダー＋手入力）
# 骨格のみ重ね比較 / 骨格描画入り動画を書き出して再生 / ffmpegでH.264正規化
# ------------------------------------------------------------

import os
import io
import cv2
import sys
import math
import time
import json
import shutil
import hashlib
import tempfile
import subprocess
import numpy as np
import streamlit as st
import mediapipe as mp

# ---- ログ/警告の抑制（必要なエラーのみ） ----
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

# ============================================================
#                 ユーティリティ（描画/変換/動画化）
# ============================================================

def draw_skeleton_on_frame(frame_bgr, pose_landmarks, color=(255, 0, 0)):
    """元フレームに骨格を描画（BGRフレームを破壊的に上書き）"""
    if pose_landmarks:
        spec = mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
        mp_drawing.draw_landmarks(
            frame_bgr,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=spec,
            connection_drawing_spec=spec
        )
    return frame_bgr

def render_skeleton_only(h: int, w: int, pose_landmarks, color=(255, 0, 0)):
    """真っ黒キャンバス(h,w,3 BGR)に骨格のみ描画"""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if pose_landmarks:
        spec = mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
        mp_drawing.draw_landmarks(
            canvas, pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=spec, connection_drawing_spec=spec
        )
    return canvas

def overlay_two_skeletons(h, w, lm1, lm2, color1=(255,0,0), color2=(0,0,255), alpha=0.8):
    """骨格のみを青/赤で同キャンバスに重ねる"""
    a = render_skeleton_only(h, w, lm1, color1)
    b = render_skeleton_only(h, w, lm2, color2)
    return cv2.addWeighted(a, alpha, b, alpha, 0.0)

def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def normalize_with_ffmpeg(src_path: str, target_h=720) -> str:
    """
    入力動画を H.264/AAC に正規化（最大高 target_h）。失敗したら元を返す。
    Safari/iPhone互換性を最大化。
    """
    if not have_ffmpeg():
        return src_path
    dst_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    # -pix_fmt yuv420p はiOS互換で重要
    cmd = [
        "ffmpeg","-y","-i", src_path,
        "-vf", f"scale='min({max(1,target_h*2)},iw)':-2,setsar=1",
        "-c:v","libx264","-preset","veryfast","-crf","23","-pix_fmt","yuv420p",
        "-c:a","aac","-b:a","128k",
        dst_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst_path
    except Exception:
        try: os.remove(dst_path)
        except: pass
        return src_path

def frames_to_mp4(frames_bgr, out_fps: int, prefer_h264: bool = True) -> bytes:
    """
    フレーム配列をmp4化。ffmpegがあればH.264/AACへ再エンコードして互換性UP。
    """
    if not frames_bgr:
        return b""
    h, w = frames_bgr[0].shape[:2]
    tmp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # まずはmp4vで書く
    vw = cv2.VideoWriter(tmp_raw, fourcc, max(1, int(out_fps or 30)), (w, h))
    for f in frames_bgr:
        vw.write(f)
    vw.release()

    if prefer_h264 and have_ffmpeg():
        tmp_h264 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        cmd = [
            "ffmpeg","-y","-i", tmp_raw,
            "-c:v","libx264","-preset","veryfast","-crf","23","-pix_fmt","yuv420p",
            "-c:a","aac","-b:a","128k", tmp_h264
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(tmp_h264, "rb") as rf:
                data = rf.read()
        except Exception:
            with open(tmp_raw, "rb") as rf:
                data = rf.read()
        finally:
            for p in (tmp_raw, tmp_h264):
                try: os.remove(p)
                except: pass
        return data
    else:
        with open(tmp_raw, "rb") as rf:
            data = rf.read()
        try: os.remove(tmp_raw)
        except: pass
        return data

# ============================================================
#                フレーム抽出（キャッシュ＆正規化）
# ============================================================

@st.cache_data(show_spinner=False, hash_funcs={bytes: lambda b: hashlib.md5(b).hexdigest()})
def extract_frames_and_skeletons(file_bytes: bytes, filename: str,
                                 model_complexity=1, max_frame_height=640):
    """
    bytes入力を一時ファイル化 → ffmpegでH.264/AAC/≤720pへ正規化 → OpenCV読込 → MediaPipe Pose。
    返り値: frames(BGR list), landmarks(list), width, height, fps
    """
    if not file_bytes:
        return [], [], 0, 0, 0

    suffix = os.path.splitext(filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        src_path = tmp.name

    norm_path = normalize_with_ffmpeg(src_path, target_h=720)

    try:
        cap = cv2.VideoCapture(norm_path)
        if not cap.isOpened():
            return [], [], 0, 0, 0

        frames, landmarks = [], []
        ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # 表示・計算の安定用に高さ制限
        nh, nw = oh, ow
        if oh > 0 and oh > max_frame_height:
            nh = max_frame_height
            nw = max(1, int(ow * (max_frame_height / oh)))

        bar = st.progress(0, text=f"処理中: {os.path.basename(filename)}") if total > 0 else None

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=int(model_complexity),
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if oh > max_frame_height > 0:
                    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                frames.append(frame)
                landmarks.append(results.pose_landmarks)
                idx += 1
                if bar and total > 0:
                    pct = min(100, int(idx/total*100))
                    bar.progress(pct, text=f"処理中: {os.path.basename(filename)} {pct}%")

        if bar:
            bar.empty()
        cap.release()
        return frames, landmarks, (nw or ow), (nh or oh), fps

    finally:
        # 一時ファイル掃除（元と正規化後）
        for p in (src_path, norm_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

# ============================================================
#                       UI（iPhone最適化）
# ============================================================

st.set_page_config(page_title="バレエフォーム比較AI", layout="centered", initial_sidebar_state="collapsed")

# モバイル向けCSS（タッチ領域/スライダー/余白）
st.markdown("""
<style>
html, body, [class*="css"] { font-size: 16px; line-height: 1.45; }
button[kind="primary"], button, .stDownloadButton { padding: 14px 16px; font-size: 16px; }
div[data-testid="stFileUploader"] section { padding: 12px; }
div[data-testid="stFileUploader"] label { font-size: 16px; }
[data-baseweb="slider"] div[role="slider"] { width: 28px; height: 28px; }
[data-baseweb="slider"] div[role="slider"]::before { width: 28px; height: 28px; }
[data-baseweb="slider"] > div { padding-top: 12px; padding-bottom: 20px; }
.block-container { padding-top: 0.8rem; padding-bottom: 5rem; }
</style>
""", unsafe_allow_html=True)

st.title("💃 バレエフォーム比較AI")
st.caption("2本の動画を“横で比較”。iPhoneでもサクッと使えるよう最適化。")

# 主要操作（本文上部に集約）
co1, co2 = st.columns(2)
with co1:
    model_complexity = st.selectbox(
        "精度/速度", options=[(0,"低（軽い）"), (1,"標準"), (2,"高（重い）")],
        index=1, format_func=lambda x: x[1], help="標準でOK。重い場合は低へ。"
    )[0]
with co2:
    display_w = st.slider("表示幅(px)", 240, 720, 360, 10, help="見た目サイズのみ（計算負荷は変わりません）")

# 必要最小限の注意（折りたたみ）
with st.expander("推奨・注意（タップで開く）", expanded=False):
    st.markdown(
        "- **長さ**: 推奨 10〜60秒（上限〜2分）\n"
        "- **人数**: **1人**（複数人だと主対象が揺れてブレやすい）\n"
        "- **形式**: そのままでOK（内部でH.264/AACに自動変換）\n"
        "- **通信**: 大きい動画はWi-Fi推奨\n"
    )

# アップロード（大ボタン / 1カラム）
st.subheader("動画を2本選ぶ（青 → 赤）")
files = st.file_uploader("2本まとめて選択", type=["mp4","mov","avi"], accept_multiple_files=True)

# セッション初期化
for i in [1, 2]:
    st.session_state.setdefault(f'frames{i}', [])
    st.session_state.setdefault(f'landmarks{i}', [])
    st.session_state.setdefault(f'w{i}', 0)
    st.session_state.setdefault(f'h{i}', 0)
    st.session_state.setdefault(f'fps{i}', 0)
    st.session_state.setdefault(f'frame_index{i}', 0)
st.session_state.setdefault("filebufs", [])

# 新規選択 → 生バイト確保（iOSの一時URL切れ対策）＆結果リセット
if files:
    st.session_state["filebufs"] = [{"name": f.name, "bytes": f.getvalue()} for f in files[:2]]
    for i in [1, 2]:
        st.session_state[f'frames{i}'] = []
        st.session_state[f'landmarks{i}'] = []
        st.session_state[f'w{i}'] = st.session_state[f'h{i}'] = st.session_state[f'fps{i}'] = 0
        st.session_state[f'frame_index{i}'] = 0

# 解析ボタン（大）
start = st.button("🔍 解析を開始", type="primary", use_container_width=True)

# 実行
if start:
    bufs = st.session_state.get("filebufs", [])
    if len(bufs) < 2:
        st.warning("2本選んでから『解析を開始』を押してください。")
    else:
        with st.status("1つ目の動画を処理中…", expanded=True) as s:
            (st.session_state.frames1,
             st.session_state.landmarks1,
             st.session_state.w1,
             st.session_state.h1,
             st.session_state.fps1) = extract_frames_and_skeletons(
                file_bytes=bufs[0]["bytes"],
                filename=bufs[0]["name"],
                model_complexity=model_complexity,
                max_frame_height=640
            )
            s.update(label="2つ目の動画を処理中…")
            (st.session_state.frames2,
             st.session_state.landmarks2,
             st.session_state.w2,
             st.session_state.h2,
             st.session_state.fps2) = extract_frames_and_skeletons(
                file_bytes=bufs[1]["bytes"],
                filename=bufs[1]["name"],
                model_complexity=model_complexity,
                max_frame_height=640
            )
            s.update(label="完了！")

# ============================================================
#                        比較UI ＋ 再生
# ============================================================

def frame_controls(key_prefix: str, max_idx: int):
    """親指で押しやすい前後ボタン＋スライダー＋手入力（すべて連動）"""
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        if st.button("⟵ 5", use_container_width=True, key=f"{key_prefix}_b-5"):
            st.session_state[key_prefix] = max(0, st.session_state[key_prefix] - 5)
    with c2:
        if st.button("⟵ 1", use_container_width=True, key=f"{key_prefix}_b-1"):
            st.session_state[key_prefix] = max(0, st.session_state[key_prefix] - 1)
    with c3:
        if st.button("1 ⟶", use_container_width=True, key=f"{key_prefix}_b+1"):
            st.session_state[key_prefix] = min(max_idx, st.session_state[key_prefix] + 1)
    with c4:
        if st.button("5 ⟶", use_container_width=True, key=f"{key_prefix}_b+5"):
            st.session_state[key_prefix] = min(max_idx, st.session_state[key_prefix] + 5)

    idx = st.slider("スライダーで選択", 0, max_idx, st.session_state[key_prefix], key=f"{key_prefix}_sld")
    idx = st.number_input("フレーム番号（手入力）", 0, max_idx, int(idx), step=1, key=f"{key_prefix}_num")
    st.session_state[key_prefix] = int(idx)

def show_playbacks():
    """動画再生モード（骨格入り／骨格のみ／骨格重ね）"""
    if not (st.session_state.frames1 and st.session_state.frames2):
        return

    st.markdown("### ▶ 再生/エクスポート")
    mode = st.radio(
        "再生モードを選択",
        ["A. 骨格を描いた元動画（個別）",
         "B. 骨格だけの動画（個別）",
         "C. 骨格だけを重ねて1本で再生"],
        index=0
    )

    fps1 = int(st.session_state.fps1 or 30)
    fps2 = int(st.session_state.fps2 or 30)
    h1, w1 = st.session_state.h1, st.session_state.w1
    h2, w2 = st.session_state.h2, st.session_state.w2

    if mode == "A. 骨格を描いた元動画（個別）":
        frames_drawn1 = []
        for f, lm in zip(st.session_state.frames1, st.session_state.landmarks1):
            fr = f.copy()
            draw_skeleton_on_frame(fr, lm, (255,0,0))
            frames_drawn1.append(fr)
        st.write("**1本目（青）**")
        st.video(frames_to_mp4(frames_drawn1, fps1))

        frames_drawn2 = []
        for f, lm in zip(st.session_state.frames2, st.session_state.landmarks2):
            fr = f.copy()
            draw_skeleton_on_frame(fr, lm, (0,0,255))
            frames_drawn2.append(fr)
        st.write("**2本目（赤）**")
        st.video(frames_to_mp4(frames_drawn2, fps2))

    elif mode == "B. 骨格だけの動画（個別）":
        sk_only1 = [render_skeleton_only(h1, w1, lm, (255,0,0)) for lm in st.session_state.landmarks1]
        st.write("**1本目（青：骨格のみ）**")
        st.video(frames_to_mp4(sk_only1, fps1))

        sk_only2 = [render_skeleton_only(h2, w2, lm, (0,0,255)) for lm in st.session_state.landmarks2]
        st.write("**2本目（赤：骨格のみ）**")
        st.video(frames_to_mp4(sk_only2, fps2))

    else:  # "C. 骨格だけを重ねて1本で再生"
        n = min(len(st.session_state.landmarks1), len(st.session_state.landmarks2))
        h, w = h1, w1  # サイズは1本目に合わせる（必要ならここで共通解像度に）
        over_frames = []
        for i in range(n):
            over_frames.append(
                overlay_two_skeletons(h, w,
                                      st.session_state.landmarks1[i],
                                      st.session_state.landmarks2[i],
                                      color1=(255,0,0), color2=(0,0,255), alpha=0.85)
            )
        st.write("**骨格重ね（青＋赤）**")
        st.video(frames_to_mp4(over_frames, fps1))

# 表示
if st.session_state.frames1 and st.session_state.frames2:
    st.subheader("フレームを選んで比較")

    # 青
    st.markdown("### 青（1本目）")
    max1 = len(st.session_state.frames1) - 1
    frame_controls("frame_index1", max1)
    f1 = st.session_state.frames1[st.session_state.frame_index1].copy()
    draw_skeleton_on_frame(f1, st.session_state.landmarks1[st.session_state.frame_index1], (255,0,0))
    st.image(f1, channels="BGR", width=display_w, caption=f"{st.session_state.frame_index1+1}/{max1+1}")

    # 赤
    st.markdown("### 赤（2本目）")
    max2 = len(st.session_state.frames2) - 1
    frame_controls("frame_index2", max2)
    f2 = st.session_state.frames2[st.session_state.frame_index2].copy()
    draw_skeleton_on_frame(f2, st.session_state.landmarks2[st.session_state.frame_index2], (0,0,255))
    st.image(f2, channels="BGR", width=display_w, caption=f"{st.session_state.frame_index2+1}/{max2+1}")

    # 再生UI
    show_playbacks()
else:
    st.info("動画を2本選んで『解析を開始』を押してください。")
