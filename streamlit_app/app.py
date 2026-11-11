"""Streamlit demo: people tracking in one video, and the full multi-camera pipeline on a WILDTRACK sample.

    uv sync --group app && uv run streamlit run streamlit_app/app.py

Streamlit Community Cloud: main file `streamlit_app/app.py` (dependencies come from streamlit_app/requirements.txt).
"""
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # use the package without installing it (keeps the cloud build CPU-only)

from mcmot.detection import PersonDetector  # noqa: E402
from mcmot.evaluate import evaluate  # noqa: E402
from mcmot.run import load_config, render_frames, track_all  # noqa: E402
from mcmot.single_video import H264Writer, track_video, video_info  # noqa: E402
from mcmot.tracking import AppearanceEmbedder  # noqa: E402
from mcmot.wildtrack import FPS, Wildtrack  # noqa: E402

DEFAULT_VIDEO = ROOT / "assets" / "sample" / "default_video.mp4"
SAMPLE_DATA = ROOT / "assets" / "sample" / "wildtrack"
MODELS = {"YOLO11n (fastest)": "yolo11n.pt", "YOLO11s": "yolo11s.pt", "YOLO11m (best, slow on CPU)": "yolo11m.pt"}

st.set_page_config(page_title="Multi-Camera Multi-Object Tracking", page_icon="🎥", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run downloads the YOLO weights)...")
def load_models(weights: str):
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    det = PersonDetector(str(ROOT / weights) if (ROOT / weights).exists() else weights, device=device)
    return det, AppearanceEmbedder(gpu=device != "cpu"), device


def progress_bar(label: str):
    bar = st.progress(0.0, text=label)
    t0 = time.time()

    def update(done: int, total: int) -> None:
        bar.progress(min(done / max(total, 1), 1.0), text=f"{label} {done}/{total} frames ({time.time() - t0:.0f} s)")

    return bar, update


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Person detector", list(MODELS), index=1)
conf = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.3, 0.05)
det, emb, device = load_models(MODELS[model_name])
det.conf = conf
st.sidebar.caption(f"Running on: **{'GPU' if device != 'cpu' else 'CPU'}**")

st.title("Multi-Camera Multi-Object Tracking")
st.markdown("Pretrained YOLO11 person detection + DeepSORT tracking. With calibrated, overlapping cameras, people "
            "are fused on the ground plane so each person gets **one identity in every camera**.")
tab_video, tab_multi = st.tabs(["🎬 Track a video", "🗺️ Multi-camera (WILDTRACK sample)"])

# ---------------------------------------------------------------- single video
with tab_video:
    st.markdown("Upload a video, or leave it empty to use the **default video** (WILDTRACK camera C1, 2 fps). "
                "A single uncalibrated video has no ground plane, so ids are per-video DeepSORT track ids.")
    upload = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv", "webm", "mpeg4"])
    if upload is not None:
        suffix = Path(upload.name).suffix or ".mp4"
        src = Path(tempfile.gettempdir()) / f"mcmot_upload_{upload.file_id}{suffix}"
        if not src.exists():
            src.write_bytes(upload.getbuffer())
        source_name = upload.name
    else:
        src, source_name = DEFAULT_VIDEO, "default video (WILDTRACK C1)"

    try:
        fps, total, w, h = video_info(src)
    except ValueError:
        st.error("This file could not be read as a video.")
        st.stop()
    st.caption(f"Input: **{source_name}**: {w}x{h}, {fps:.1f} fps, {total} frames ({total / fps:.1f} s)")

    c1, c2, c3 = st.columns(3)
    default_stride = max(1, round(fps / 5)) if upload is not None else 1  # ~5 fps keeps CPU time reasonable
    stride = c1.number_input("Process every n-th frame", 1, 30, default_stride,
                             help="Higher = faster; the output plays at fps / n.")
    max_frames = c2.number_input("Max processed frames", 10, 2000, min(150, max(10, -(-total // stride))),
                                 help="Caps run time on the CPU-only cloud.")
    width = c3.select_slider("Processing width (px)", [480, 640, 960, 1280], 960)
    imgsz = 1280 if width > 960 else 960

    key = ("video", str(src), stride, max_frames, width, model_name, conf)
    if st.button("▶ Run tracking", type="primary", key="run_video"):
        det.imgsz = imgsz
        out = Path(tempfile.mkdtemp()) / "tracked.mp4"
        bar, update = progress_bar("Detecting + tracking")
        rows = track_video(src, out, det, emb, stride=int(stride), max_frames=int(max_frames), width=width,
                           progress=update)
        bar.empty()
        st.session_state["video_result"] = (key, out.read_bytes(), rows)

    result = st.session_state.get("video_result")
    if result and result[0] == key:
        _, video_bytes, rows = result
        left, right = st.columns([3, 1])
        left.video(video_bytes)
        df = pd.DataFrame([{**{k: v for k, v in r.items() if k != "box"},
                            **dict(zip(["x1", "y1", "x2", "y2"], r["box"]))} for r in rows])
        right.metric("Tracked people (unique ids)", df["track_id"].nunique() if len(df) else 0)
        right.metric("Boxes", len(df))
        right.download_button("Download video", video_bytes, "tracked.mp4", "video/mp4")
        right.download_button("Download tracks (CSV)", df.to_csv(index=False), "tracks.csv", "text/csv")
        if len(df):
            st.caption("People tracked per processed frame")
            st.line_chart(df.groupby("frame").size().rename("people"))
    elif result:
        st.info("Settings or input changed: press **Run tracking** again.")

# ---------------------------------------------------------------- multi-camera
with tab_multi:
    ds = Wildtrack(SAMPLE_DATA)
    st.markdown(f"The full pipeline on a bundled WILDTRACK sample: **{ds.num_cameras} calibrated cameras**, "
                f"{len(ds.frames)} frames ({len(ds.frames) / FPS:.0f} s at {FPS} fps), half resolution. "
                "Detect + DeepSORT in each camera → feet projected to the ground plane → cameras fused per frame "
                "→ ground-plane tracking → global ids, scored against the ground truth.")
    st.caption("Box colour and number = global identity (the same in every camera). Bird's-eye view: coloured dots "
               "= tracker, grey rings = ground truth.")

    key = ("multi", model_name, conf)
    if st.button("▶ Run multi-camera tracking", type="primary", key="run_multi"):
        cfg = load_config(str(ROOT / "configs" / "wildtrack.yaml"), [])
        det.imgsz = 960  # sample images are 960x540
        bar, update = progress_bar("Detecting + tracking in 7 cameras")
        rows, ground = track_all(ds, ds.frames, cfg, det=det, emb=emb, progress=update)
        bar.progress(1.0, text="Rendering video...")
        gt = {f: ds.annotations(f) for f in ds.frames}
        out = Path(tempfile.mkdtemp()) / "multi_camera.mp4"
        writer = H264Writer(out, FPS)
        for canvas in render_frames(ds, ds.frames, rows, ground, gt):
            writer.write(canvas)
        writer.close()
        bar.empty()
        e = cfg["evaluation"]
        metrics = evaluate(gt, rows, ground, ds.num_cameras, e["iou_threshold"], e["ground_dist_cm"])
        st.session_state["multi_result"] = (key, out.read_bytes(), metrics, rows, ground)

    result = st.session_state.get("multi_result")
    if result and result[0] == key:
        _, video_bytes, metrics, rows, ground = result
        st.video(video_bytes)
        m = metrics["ground"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Global identities", len({g for d in ground.values() for g in d}))
        c2.metric("Ground MOTA", f"{m['mota']:.3f}")
        c3.metric("Ground IDF1", f"{m['idf1']:.3f}")
        c4.metric("Cross-camera IDF1", f"{metrics['cross_camera']['idf1']:.3f}")
        table = pd.DataFrame(metrics).T[["mota", "idf1", "precision", "recall", "num_switches", "num_unique_objects"]]
        st.dataframe(table.style.format("{:.3f}", subset=["mota", "idf1", "precision", "recall"]))
        st.caption(f"Only {len(ds.frames)} frames: the numbers are noisier than the full-dataset results in the "
                   "README (ground MOTA 0.49 / IDF1 0.58 on all 400 frames with YOLO11m at full resolution).")
        st.download_button("Download video", video_bytes, "multi_camera.mp4", "video/mp4")
    elif result:
        st.info("Settings changed: press **Run multi-camera tracking** again.")
