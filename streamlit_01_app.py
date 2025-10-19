import os
import cv2
import numpy as np
import tempfile
import hashlib
import streamlit as st
import mediapipe as mp
import subprocess  # ← 追加：ffmpeg呼び出しに使用

# ----（任意）警告を抑制（必要なエラーのみ表示）----
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


# ---- ffmpeg で H.264/MP4/≤720p/30fps に正規化（必要時のみ呼ぶ）----
def _ffmpeg_normalize_to_h264_720p(in_path: str) -> str:
    """
    失敗しがちな HEVC/MOV や 1080p/4K/VFR を、H.264/MP4/≤720p/30fps に正規化。
    縦横比は維持。-2 で偶数に揃える。
    """
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-vf", "scale='min(1280,iw)':-2",     # 横最大1280（=720p相当）、縦は自動・偶数
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "23",
        "-r", "30",                           # 可変フレームレートの揺れ対策
        "-c:a", "aac", "-b:a", "128k",
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out_path


# ---- 動画→フレーム/骨格抽出（キャッシュはしない！）----
def extract_frames_and_skeletons(file_bytes: bytes, filename: str,
                                 model_complexity=1, max_frame_height=640):
    """
    file_bytes: 動画の生バイト
    filename  : 元のファイル名（拡張子取得や表示に使用）
    戻り値    : (frames(list[np.ndarray BGR]),
                 landmarks(list[NormalizedLandmarkList|None]),
                 width, height, fps)
    """
    if not file_bytes:
        return [], [], 0, 0, 0

    # 一時ファイルに書き出し
    suffix = os.path.splitext(filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    # 条件に応じて正規化するか判定
    norm_path = None
    cap = None
    try:
        # まずはそのまま開いてみる
        cap = cv2.VideoCapture(temp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 0
        oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 0

        need_normalize = False
        # 1) そもそも開けないorフレーム0
        if (not cap.isOpened()) or total == 0:
            need_normalize = True
        # 2) 高解像度（>720p）
        elif oh and oh > max_frame_height:
            need_normalize = True
        # 3) .mov は HEVC 率が高く、環境依存で失敗しやすいので救済
        elif suffix.lower() == ".mov":
            need_normalize = True

        if need_normalize:
            if cap:
                cap.release()
            norm_path = _ffmpeg_normalize_to_h264_720p(temp_path)
            cap = cv2.VideoCapture(norm_path)
            if not cap.isOpened():
                return [], [], 0, 0, 0

        # 正式にメタデータを取得し直す
        ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # 表示/処理サイズ（720p以下に）
        nw, nh = ow, oh
        if oh > 0 and oh > max_frame_height:
            nh = max_frame_height
            nw = max(1, int(ow * (max_frame_height / oh)))

        bar = st.progress(0, text=f"処理中: {os.path.basename(filename)}") if total > 0 else None

        frames, landmarks = [], []
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

                # リサイズ（必要時）
                if oh > 0 and oh > max_frame_height:
                    try:
                        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                    except Exception:
                        # 異常フレームはスキップ
                        idx += 1
                        continue

                # 色空間変換の保険
                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception:
                    idx += 1
                    continue

                # Mediapipe 推定
                results = pose.process(image_rgb)
                frames.append(frame)
                landmarks.append(getattr(results, "pose_landmarks", None))

                idx += 1
                if bar and total > 0:
                    pct = min(100, int(idx / total * 100))
                    bar.progress(pct, text=f"処理中: {os.path.basename(filename)} {pct}%")

        if bar:
            bar.empty()
        cap.release()
        return frames, landmarks, (nw or ow), (nh or oh), fps

    finally:
        # 一時ファイル掃除（存在チェックしてから）
        for p in [temp_path, norm_path]:
            try:
                if p and isinstance(p, str) and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# ---- UI 基本設定 ----
st.set_page_config(layout="wide", page_title="バレエフォーム比較AI")
st.title("💃 バレエフォーム比較AI")

st.markdown("""
### 📖 使い方
1. 下のアップロード欄で **2本まとめて** 動画を選択してください（順に **青 → 赤** として扱います）。
2. **解析を開始** を押すと骨格推定が始まり、フレームごとに横並びで比較できます。
3. フレーム番号は **手入力** と **スライダー** の両方で細かく調整できます。

⚠️ **推奨**：10〜20秒・720p以下・**H.264(MP4)**。  
iPhone標準の高圧縮 **.mov(HEVC)** は失敗しやすいです。**Wi-Fi** 推奨、アップロード中は画面を閉じないでください。
""")

# ---- モデル精度/速度 ----
model_complexity_option = st.selectbox(
    "ポーズ推定モデルの精度/速度",
    options=[(0, "低（高速）"), (1, "中（バランス）"), (2, "高（精密）")],
    format_func=lambda x: x[1],
    index=1
)[0]

MAX_FRAME_HEIGHT = 640  # 重ければ 480 に下げるとさらに安定

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

# ---- 同時アップロード + 送信ボタン（フォームでセッション安定化）----
with st.form(key="upload_form", clear_on_submit=False):
    files = st.file_uploader(
        "動画を **2本まとめて** アップロード（順に 青 → 赤 として扱います）",
        type=['mp4', 'mov', 'avi'],
        accept_multiple_files=True
    )
    st.info("⚠️ 推奨: 10〜20秒以内・720p以下・H.264(MP4)。iPhoneの高圧縮MOV(HEVC)は失敗しやすいです。Wi-Fiで、アップロード中は画面を閉じないでください。")
    submitted = st.form_submit_button("解析を開始")

# 新しい選択が来たら生バイト退避（URL失効対策）＆前回結果クリア
if files:
    sig = tuple((f.name, f.size) for f in files[:2])
    if st.session_state["last_files_sig"] != sig:
        st.session_state["last_files_sig"] = sig
        st.session_state["filebufs"] = []
        for f in files[:2]:
            # iPhone/Safariでも安定するよう getvalue() で生バイト確保
            st.session_state["filebufs"].append({"name": f.name, "bytes": f.getvalue()})
        for i in [1, 2]:
            st.session_state[f'frames{i}'] = []
            st.session_state[f'landmarks{i}'] = []
            st.session_state[f'w{i}'] = st.session_state[f'h{i}'] = st.session_state[f'fps{i}'] = 0
            st.session_state[f'frame_index{i}'] = 0

bufs = st.session_state.get("filebufs", [])

# ---- 送信時のみ解析実行（再実行に強い）----
if submitted and len(bufs) >= 2:
    if not st.session_state.frames1:
        with st.spinner("1つ目の動画を処理中..."):
            (st.session_state.frames1,
             st.session_state.landmarks1,
             st.session_state.w1,
             st.session_state.h1,
             st.session_state.fps1) = extract_frames_and_skeletons(
                file_bytes=bufs[0]["bytes"],
                filename=bufs[0]["name"],
                model_complexity=model_complexity_option,
                max_frame_height=MAX_FRAME_HEIGHT
            )
        st.session_state.frame_index1 = 0

    if not st.session_state.frames2:
        with st.spinner("2つ目の動画を処理中..."):
            (st.session_state.frames2,
             st.session_state.landmarks2,
             st.session_state.w2,
             st.session_state.h2,
             st.session_state.fps2) = extract_frames_and_skeletons(
                file_bytes=bufs[1]["bytes"],
                filename=bufs[1]["name"],
                model_complexity=model_complexity_option,
                max_frame_height=MAX_FRAME_HEIGHT
            )
        st.session_state.frame_index2 = 0

elif submitted and len(bufs) < 2:
    st.warning("2本の動画を選んでから『解析を開始』を押してください。")

# ---- 比較UI ----
if st.session_state.frames1 and st.session_state.frames2:
    st.subheader("🎬 フレーム選択で骨格比較")
    display_w = st.slider("表示画像幅（px）", 200, 900, 360, 10)

    col1, col2 = st.columns(2)

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
    st.error("動画の読み込み・解析に失敗しました。コーデックを H.264/MP4 にする、長さを短くする（10〜20秒）、解像度を下げる（≤720p）などをお試しください。")

else:
    st.info("2本の動画を選んで『解析を開始』を押すと比較できます。")

