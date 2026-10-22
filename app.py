"""
Autonomous Driving Perception Stack — Streamlit Demonstration Application.
Interactive demo for road object detection, ByteTrack tracking,
OpenCV lane detection, and MiDaS monocular relative depth estimation.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# Setup project path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.depth.depth_estimator import MonocularDepthEstimator
from src.detection.detector import RoadObjectDetector
from src.lanes.lane_detector import OpenCVLaneDetector
from src.pipeline.perception_pipeline import PerceptionPipeline, PipelineConfig
from src.tracking.tracker import RoadObjectTracker

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Autonomous Driving Perception Stack",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E88E5;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #78909C;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-active { background-color: #2E7D32; color: white; }
    .badge-inactive { background-color: #546E7A; color: white; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector(model_path: str, conf: float, imgsz: int):
    return RoadObjectDetector(model_path=model_path, conf_thresh=conf, imgsz=imgsz, device="cpu")


@st.cache_resource
def load_tracker(model_path: str, conf: float, imgsz: int):
    return RoadObjectTracker(model_path=model_path, conf_thresh=conf, imgsz=imgsz, device="cpu")


@st.cache_resource
def load_lane_detector():
    return OpenCVLaneDetector()


@st.cache_resource
def load_depth_estimator():
    return MonocularDepthEstimator(model_type="MiDaS_small", device="cpu")


def main():
    st.markdown('<div class="main-title">🚗 Autonomous Driving Perception Stack</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Real-time road obstacle detection, ByteTrack tracking, '
        'classical lane corridor estimation, and relative depth assessment on CPU.</div>',
        unsafe_allow_html=True,
    )

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("⚙️ Perception Modules")

    enable_tracking = st.sidebar.checkbox("Multi-Object Tracking (ByteTrack)", value=True)
    enable_lanes = st.sidebar.checkbox("Lane Detection (OpenCV)", value=True)
    enable_depth = st.sidebar.checkbox("Monocular Depth (MiDaS_small)", value=False)

    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Parameters")
    conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
    img_size = st.sidebar.selectbox("Inference Resolution", options=[480, 640], index=1)
    max_frames = st.sidebar.slider("Max Video Frames (CPU preview)", min_value=15, max_value=200, value=60, step=15)

    if enable_depth:
        depth_freq = st.sidebar.slider("Depth Inference Frequency", min_value=1, max_value=10, value=5, help="Compute depth every N frames to conserve CPU")
    else:
        depth_freq = 5

    # Model Checkpoint Selector (Preserves Baseline as default)
    available_models = {"Baseline (best.pt)": project_root / "models" / "best.pt"}
    ft_path = project_root / "models" / "finetuned_best.pt"
    if ft_path.exists():
        available_models["Fine-Tuned Subset (finetuned_best.pt)"] = ft_path

    selected_model_name = st.sidebar.selectbox(
        "Detector Checkpoint",
        options=list(available_models.keys()),
        index=0,
        help="Select between the baseline model and the fine-tuned mini-subset checkpoint."
    )
    model_path = available_models[selected_model_name]

    # System Status Indicator
    status_html = "<div>"
    status_html += f'<span class="status-badge badge-active">Detection: {selected_model_name}</span>'
    status_html += f'<span class="status-badge {"badge-active" if enable_tracking else "badge-inactive"}">Tracking: {"ByteTrack" if enable_tracking else "Off"}</span>'
    status_html += f'<span class="status-badge {"badge-active" if enable_lanes else "badge-inactive"}">Lanes: {"OpenCV" if enable_lanes else "Off"}</span>'
    status_html += f'<span class="status-badge {"badge-active" if enable_depth else "badge-inactive"}">Depth: {"MiDaS (Rel)" if enable_depth else "Off"}</span>'
    status_html += '<span class="status-badge badge-active">Hardware: CPU (Intel Iris Xe)</span>'
    status_html += "</div>"
    st.markdown(status_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not model_path.exists():
        st.error(f"Model checkpoint not found at `{model_path}`. Run `python scripts/download_model.py` first.")
        return

    # --- MAIN TABS ---
    tab_sample, tab_image, tab_video = st.tabs(["🚀 Sample Driving Media", "🖼️ Image Upload", "🎥 Video Upload"])

    # --- TAB 1: SAMPLE DRIVING MEDIA ---
    with tab_sample:
        st.write("Test the perception stack immediately on included BDD100K test assets:")
        sample_img_path = project_root / "data" / "sample_road.jpg"
        sample_vid_path = project_root / "data" / "demo_banner.mp4"

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Sample Frame")
            if sample_img_path.exists():
                st.image(str(sample_img_path), caption="BDD100K Test Frame", use_container_width=True)
                if st.button("Run Perception on Sample Image", key="btn_sample_img"):
                    with st.spinner("Processing image perception..."):
                        img = cv2.imread(str(sample_img_path))
                        process_single_image(img, model_path, conf_thresh, img_size, enable_lanes, enable_depth)
            else:
                st.warning("Sample road image not found in `data/`.")

        with col_b:
            st.markdown("#### Sample Driving Video")
            if sample_vid_path.exists():
                st.video(str(sample_vid_path))
                if st.button("Process Sample Video Clip", key="btn_sample_vid"):
                    run_video_pipeline(
                        str(sample_vid_path),
                        model_path,
                        conf_thresh,
                        img_size,
                        enable_tracking,
                        enable_lanes,
                        enable_depth,
                        depth_freq,
                        max_frames,
                    )
            else:
                st.warning("Sample driving video not found in `data/`.")

    # --- TAB 2: IMAGE UPLOAD ---
    with tab_image:
        uploaded_file = st.file_uploader("Upload Driving Scene Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
            if st.button("Process Uploaded Image", key="btn_proc_img"):
                with st.spinner("Executing perception pipeline on uploaded image..."):
                    process_single_image(img, model_path, conf_thresh, img_size, enable_lanes, enable_depth)

    # --- TAB 3: VIDEO UPLOAD ---
    with tab_video:
        uploaded_video = st.file_uploader("Upload Road Driving Video", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            # Save uploaded video to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            temp_vid_path = tfile.name

            st.video(temp_vid_path)
            if st.button("Start Video Perception", key="btn_proc_vid"):
                run_video_pipeline(
                    temp_vid_path,
                    model_path,
                    conf_thresh,
                    img_size,
                    enable_tracking,
                    enable_lanes,
                    enable_depth,
                    depth_freq,
                    max_frames,
                )


def process_single_image(img: np.ndarray, model_path: Path, conf: float, imgsz: int, enable_lanes: bool, enable_depth: bool):
    """Processes and renders a single frame with metrics."""
    detector = RoadObjectDetector(model_path=model_path, conf_thresh=conf, imgsz=imgsz, device="cpu")
    t0 = time.perf_counter()
    det_res = detector.detect(img)

    lane_res = None
    if enable_lanes:
        lane_detector = load_lane_detector()
        lane_res = lane_detector.detect(img)

    depth_res = None
    if enable_depth:
        depth_estimator = load_depth_estimator()
        depth_res = depth_estimator.estimate(img)

    total_lat = (time.perf_counter() - t0) * 1000.0

    # Render
    annotated = img.copy()
    if lane_res:
        annotated = lane_detector.draw_lanes(annotated, lane_res)
    annotated = detector.draw_detections(annotated, det_res)

    # Display Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Latency", f"{total_lat:.1f} ms", f"{1000.0/total_lat:.1f} FPS")
    with c2:
        st.metric("Detected Objects", f"{det_res.count}")
    with c3:
        lane_stat = f"{lane_res.lane_center_offset_px:.1f} px" if (lane_res and lane_res.lane_center_offset_px is not None) else "N/A"
        st.metric("Lane Offset", lane_stat)
    with c4:
        st.metric("Depth Status", "Active (Rel)" if depth_res else "Disabled")

    # Render Visuals
    if depth_res:
        col_out1, col_out2 = st.columns(2)
        with col_out1:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Perception Output (Detections + Lanes)", use_container_width=True)
        with col_out2:
            st.image(cv2.cvtColor(depth_res.colored_map, cv2.COLOR_BGR2RGB), caption="Relative Depth Map (MiDaS_small - Not Meters)", use_container_width=True)
    else:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Perception Output (Detections + Lanes)", use_container_width=True)

    # Class breakdown
    if det_res.count > 0:
        st.markdown("##### Detected Classes Breakdown")
        counts = {}
        for cname in det_res.class_names:
            counts[cname] = counts.get(cname, 0) + 1
        st.write(", ".join([f"**{c}**: {n}" for c, n in sorted(counts.items())]))


def run_video_pipeline(
    video_source: str,
    model_path: Path,
    conf: float,
    imgsz: int,
    enable_tracking: bool,
    enable_lanes: bool,
    enable_depth: bool,
    depth_freq: int,
    max_frames: int,
):
    """Processes video sequence, displays real-time progress, and previews output video."""
    config = PipelineConfig(
        model_path=str(model_path),
        enable_tracking=enable_tracking,
        enable_lanes=enable_lanes,
        enable_depth=enable_depth,
        depth_frequency=depth_freq,
        conf_thresh=conf,
        imgsz=imgsz,
        device="cpu",
    )
    pipeline = PerceptionPipeline(config=config)

    progress_bar = st.progress(0, text="Initializing video pipeline...")
    status_text = st.empty()
    frame_preview = st.empty()

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = results_dir / "streamlit_demo_output.mp4"

    cap = cv2.VideoCapture(video_source)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writer for output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps_in, (width, height))

    frame_idx = 0
    latencies = []
    t_start = time.perf_counter()

    try:
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            result = pipeline.process_frame(frame, frame_index=frame_idx)
            latencies.append(result.total_latency_ms)
            writer.write(result.annotated_frame)

            # Update progress
            pct = int(frame_idx / max_frames * 100)
            progress_bar.progress(pct, text=f"Processing frame {frame_idx}/{max_frames} ({result.effective_fps:.1f} FPS)...")

            # Update preview every 5 frames
            if frame_idx % 5 == 0 or frame_idx == max_frames:
                rgb_preview = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
                frame_preview.image(rgb_preview, caption=f"Live Preview (Frame {frame_idx})", use_container_width=True)

    finally:
        cap.release()
        writer.release()

    total_time = time.perf_counter() - t_start
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    overall_fps = frame_idx / total_time if total_time > 0 else 0.0

    progress_bar.progress(100, text="Processing completed!")
    status_text.success(
        f"✅ Finished {frame_idx} frames in {total_time:.2f}s "
        f"(Avg Latency: {avg_latency:.1f} ms, Overall: {overall_fps:.1f} FPS)"
    )

    # Provide Download and Playback
    st.markdown("#### Output Video Preview")
    if out_video_path.exists():
        st.video(str(out_video_path))
        with open(out_video_path, "rb") as f:
            st.download_button(
                label="📥 Download Annotated Video",
                data=f.read(),
                file_name="annotated_perception_clip.mp4",
                mime="video/mp4",
            )


if __name__ == "__main__":
    main()
