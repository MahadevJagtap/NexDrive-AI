"""
FastAPI Server for Autonomous Driving Perception Stack.
Provides REST API endpoints and serves the modern HTML5/CSS3/Vanilla JS frontend.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Prevent thread contention and CPU starvation on containerized cloud hosts (Render, Docker)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure DLL search path on Windows for OpenH264 (H.264 browser-compatible video encoding)
if sys.platform == "win32":
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(PROJECT_ROOT))
        except Exception:
            pass
        venv_scripts = PROJECT_ROOT / ".venv" / "Scripts"
        if venv_scripts.exists():
            try:
                os.add_dll_directory(str(venv_scripts))
            except Exception:
                pass
    os.environ["PATH"] = f"{PROJECT_ROOT};{PROJECT_ROOT / '.venv' / 'Scripts'};" + os.environ.get("PATH", "")

import cv2
cv2.setNumThreads(1)
import numpy as np
import torch
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.pipeline.perception_pipeline import PerceptionPipeline, PipelineConfig, PipelineFrameResult

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("auto_drive.server")

# Directories
FRONTEND_DIR = PROJECT_ROOT / "frontend"
RESULTS_DIR = PROJECT_ROOT / "results"
JOBS_DIR = RESULTS_DIR / "jobs"
UPLOADS_DIR = RESULTS_DIR / "uploads"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline instances cache
_PIPELINE_CACHE: Dict[str, PerceptionPipeline] = {}
_CACHE_LOCK = threading.Lock()

# Job management
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def get_cached_pipeline(config: PipelineConfig) -> PerceptionPipeline:
    """Retrieves or instantiates a cached PerceptionPipeline instance to avoid reloading models."""
    cache_key = (
        f"{config.model_path}:{config.enable_tracking}:{config.enable_lanes}:"
        f"{getattr(config, 'lane_method', 'classical')}:{config.enable_depth}:"
        f"{config.depth_frequency}:{config.conf_thresh}:{config.imgsz}:{config.device}"
    )
    with _CACHE_LOCK:
        if cache_key not in _PIPELINE_CACHE:
            logger.info(f"Instantiating new cached pipeline with key: {cache_key}")
            _PIPELINE_CACHE[cache_key] = PerceptionPipeline(config=config)
        return _PIPELINE_CACHE[cache_key]


def init_video_writer(out_path: Path, fps: float, width: int, height: int) -> Tuple[cv2.VideoWriter, str]:
    """
    Initializes a VideoWriter with browser-compatible codecs.
    Tries 'avc1' (H.264 standard for HTML5 <video> in Chrome/Edge/Firefox/Safari),
    then 'H264', and falls back to 'mp4v'.
    """
    codecs_to_try = ["avc1", "H264", "mp4v"]
    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if writer.isOpened():
                logger.info(f"Initialized VideoWriter with codec '{codec}' for {out_path.name}")
                return writer, codec
            writer.release()
        except Exception as e:
            logger.debug(f"Failed VideoWriter attempt with codec '{codec}': {e}")

    raise RuntimeError(f"Could not open VideoWriter for {out_path} with any supported codec.")


def stream_video_file(file_path: Path, request: Request, download: bool = False, filename: Optional[str] = None):
    """
    Streams a video file supporting HTTP 206 Partial Content (Byte Range requests)
    for seamless seeking/buffering in HTML5 <video>, or returns an attachment on download.
    """
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found.")

    file_size = file_path.stat().st_size
    if file_size == 0:
        raise HTTPException(status_code=500, detail="Video file is empty (0 bytes).")

    safe_filename = filename or file_path.name

    # If explicit download requested, serve as attachment
    if download:
        return FileResponse(
            path=file_path,
            media_type="video/mp4",
            filename=safe_filename,
            headers={"Content-Disposition": f'attachment; filename="{safe_filename}"'},
        )

    # Check for HTTP Range header for seeking and smooth HTML5 streaming
    range_header = request.headers.get("range")
    if range_header and range_header.startswith("bytes="):
        try:
            range_str = range_header.replace("bytes=", "").strip()
            if range_str.startswith("-"):
                start = max(0, file_size - int(range_str[1:]))
                end = file_size - 1
            elif "-" in range_str:
                parts = range_str.split("-")
                start = int(parts[0])
                end = int(parts[1]) if parts[1] else file_size - 1
            else:
                start = 0
                end = file_size - 1

            start = max(0, start)
            end = min(end, file_size - 1)
            content_length = end - start + 1

            def file_range_iterator(p: Path, offset: int, length: int, chunk_size: int = 128 * 1024):
                with open(p, "rb") as f:
                    f.seek(offset)
                    bytes_left = length
                    while bytes_left > 0:
                        to_read = min(chunk_size, bytes_left)
                        chunk = f.read(to_read)
                        if not chunk:
                            break
                        bytes_left -= len(chunk)
                        yield chunk

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Content-Type": "video/mp4",
                "Content-Disposition": f'inline; filename="{safe_filename}"',
            }
            return StreamingResponse(
                file_range_iterator(file_path, start, content_length),
                status_code=206,
                headers=headers,
            )
        except Exception as e:
            logger.warning(f"Error serving range request for {file_path.name}: {e}")

    # Full file response with Accept-Ranges
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=safe_filename,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{safe_filename}"',
        },
    )


def ensure_models_and_assets() -> None:
    """Ensures required baseline model checkpoint exists, auto-downloading if necessary."""
    baseline_pt = MODELS_DIR / "best.pt"
    if not baseline_pt.exists() or baseline_pt.stat().st_size == 0:
        logger.info("Baseline model checkpoint (best.pt) not found. Downloading from Hugging Face...")
        try:
            from huggingface_hub import hf_hub_download
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            dl_path = hf_hub_download(
                repo_id="dronefreak/bdd100k-yolov8n",
                filename="best.pt",
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False,
            )
            if Path(dl_path).exists() and not baseline_pt.exists():
                shutil.copy(dl_path, baseline_pt)
            logger.info(f"Successfully downloaded models/best.pt ({baseline_pt.stat().st_size} bytes)")
        except Exception as e:
            logger.error(f"Failed to auto-download models/best.pt: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure baseline YOLO model is downloaded and ready at startup
    ensure_models_and_assets()

    # Pre-warm baseline detector model in background to eliminate first-frame cold start lag
    def _warmup():
        try:
            baseline_pt = MODELS_DIR / "best.pt"
            if baseline_pt.exists():
                from ultralytics import YOLO
                warm_model = YOLO(str(baseline_pt))
                dummy = np.zeros((384, 640, 3), dtype=np.uint8)
                warm_model.predict(dummy, imgsz=640, device="cpu", verbose=False)
                logger.info("YOLO baseline detector pre-warmed successfully.")
        except Exception as e:
            logger.warning(f"Model warm-up note: {e}")

    threading.Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(
    title="Autonomous Driving Perception Stack API",
    description="REST backend for real-time road perception, obstacle tracking, lane analysis, and relative depth.",
    version="2.0.0",
    lifespan=lifespan,
)

# Optional CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz() -> str:
    """Instantaneous ping for cloud health probes."""
    return "OK"


@app.get("/api/health")
async def api_health() -> Dict[str, Any]:
    """Lightweight, non-blocking health check for Render and orchestration monitors."""
    baseline_pt = MODELS_DIR / "best.pt"
    finetuned_pt = MODELS_DIR / "finetuned_best.pt"
    sample_video = DATA_DIR / "demo_banner.mp4"

    return {
        "status": "online",
        "device": "cpu",
        "models": {
            "baseline_yolov8n": baseline_pt.exists(),
            "finetuned_yolov8n": finetuned_pt.exists(),
        },
        "sample_video_ready": sample_video.exists(),
        "active_jobs": len(_JOBS),
        "server_time": time.time(),
    }


@app.get("/api/samples")
def get_sample_videos() -> List[Dict[str, Any]]:
    """List available pre-packaged sample driving videos."""
    samples = []
    sample_video = DATA_DIR / "demo_banner.mp4"
    if sample_video.exists():
        cap = cv2.VideoCapture(str(sample_video))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 255
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 2560
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 720
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 30.0
        cap.release()

        samples.append({
            "id": "demo_banner",
            "name": "BDD100K Multi-Lane Highway Sequence",
            "filename": "demo_banner.mp4",
            "size_mb": round(sample_video.stat().st_size / (1024.0 * 1024.0), 2),
            "resolution": f"{w}x{h}",
            "frames": fc,
            "fps": fps,
            "path": str(sample_video),
        })

    return samples




def _run_job_worker(job_id: str) -> None:
    """Background worker thread for processing a perception job."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        job["status"] = "processing"
        job["start_time"] = time.perf_counter()

    ensure_models_and_assets()

    cfg_dict = job["config"]
    input_path = job["input_path"]
    job_dir = Path(job["job_dir"])
    out_video_path = job_dir / "annotated_perception.mp4"
    telemetry_path = job_dir / "telemetry.json"

    # Select model checkpoint
    model_variant = cfg_dict.get("model_variant", "baseline")
    if model_variant == "finetuned" and (MODELS_DIR / "finetuned_best.pt").exists():
        model_path = str(MODELS_DIR / "finetuned_best.pt")
    else:
        model_path = str(MODELS_DIR / "best.pt")

    # Safety: ensure depth frequency is not too aggressive on CPU (minimum 8 frames between depth passes)
    depth_freq = max(8, cfg_dict.get("depth_freq", 8))

    config = PipelineConfig(
        model_path=model_path,
        enable_tracking=cfg_dict.get("enable_tracking", True),
        enable_lanes=cfg_dict.get("enable_lanes", True),
        lane_method=cfg_dict.get("lane_method", "classical"),
        enable_depth=cfg_dict.get("enable_depth", False),
        depth_frequency=depth_freq,
        conf_thresh=cfg_dict.get("conf_thresh", 0.25),
        imgsz=cfg_dict.get("imgsz", 640),
        device=cfg_dict.get("device", "cpu"),
        draw_hud=True,
        enable_hazard_assessment=True,
    )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        with _JOBS_LOCK:
            job["status"] = "failed"
            job["error"] = f"Could not decode input video: {Path(input_path).name}"
        return

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    if raw_fps is None or np.isnan(raw_fps) or raw_fps <= 1.0 or raw_fps > 120.0:
        logger.warning(f"Unreliable or missing FPS ({raw_fps}) in video. Defaulting to 30.0 FPS.")
        fps_in = 30.0
    else:
        fps_in = float(raw_fps)

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_in_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_in_frames <= 0:
        total_in_frames = 0

    # High-throughput resolution capping: Cap processing resolution to 1280px max width for fast CPU execution
    MAX_PROC_WIDTH = 1280
    if orig_width > MAX_PROC_WIDTH:
        scale = MAX_PROC_WIDTH / float(orig_width)
        proc_w = int(orig_width * scale)
        proc_h = int(orig_height * scale)
        proc_w = proc_w - (proc_w % 2)
        proc_h = proc_h - (proc_h % 2)
    else:
        proc_w = orig_width - (orig_width % 2)
        proc_h = orig_height - (orig_height % 2)

    # Adaptive frame stride: For 24+ FPS videos, process at smooth 12-15 FPS.
    # This maintains exact real-world playback duration while cutting CPU processing time by 50%-75%.
    stride = 1
    if fps_in >= 48.0:
        stride = 4
    elif fps_in >= 24.0:
        stride = 2
    fps_out = max(1.0, round(fps_in / stride, 2))

    effective_total_frames = int(np.ceil(total_in_frames / stride)) if total_in_frames > 0 else 0
    max_frames = cfg_dict.get("max_frames", 0)
    if max_frames and max_frames > 0:
        target_frames = min(effective_total_frames, max_frames) if effective_total_frames > 0 else max_frames
    else:
        target_frames = effective_total_frames

    with _JOBS_LOCK:
        job["total_frames"] = target_frames if target_frames > 0 else None
        job["fps_in"] = fps_in
        job["fps_out"] = fps_out
        job["resolution"] = f"{proc_w}x{proc_h}"

    writer, used_codec = init_video_writer(out_video_path, fps_out, proc_w, proc_h)

    pipeline = get_cached_pipeline(config)
    pipeline.reset()

    raw_frame_idx = 0
    frame_idx = 0
    latencies: List[float] = []
    telemetry_records: List[Dict[str, Any]] = []
    all_tracked_objects: Dict[Union[int, str], Dict[str, Any]] = {}
    cancel_evt = job["cancel_event"]

    try:
        while True:
            if cancel_evt.is_set():
                logger.info(f"Job {job_id} was cancelled by client request.")
                with _JOBS_LOCK:
                    job["status"] = "cancelled"
                break

            # If user explicitly requested a positive max_frames limit, honor it
            if max_frames and max_frames > 0 and frame_idx >= max_frames:
                logger.info(f"Reached user-specified max_frames limit: {max_frames}")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info(f"Reached end of video stream at frame {frame_idx} (raw frame {raw_frame_idx}).")
                break

            raw_frame_idx += 1
            if stride > 1 and (raw_frame_idx - 1) % stride != 0:
                continue

            frame_idx += 1

            # Downscale frame if needed for fast CPU inference and encoding
            if (frame.shape[1], frame.shape[0]) != (proc_w, proc_h):
                frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

            result: PipelineFrameResult = pipeline.process_frame(frame, frame_index=frame_idx)
            latencies.append(result.total_latency_ms)
            telemetry_records.append(result.telemetry)

            if writer:
                writer.write(result.annotated_frame)

            # Class breakdown and frame obstacles
            class_counts: Dict[str, int] = {}
            active_obstacles: List[Dict[str, Any]] = []

            if result.tracks and len(result.tracks.track_ids) > 0:
                for tid, cname, conf in zip(result.tracks.track_ids, result.tracks.class_names, result.tracks.scores):
                    tid_int = int(tid) if tid >= 0 else None
                    conf_float = round(float(conf), 2)
                    cname_str = str(cname)
                    class_counts[cname_str] = class_counts.get(cname_str, 0) + 1
                    active_obstacles.append({
                        "track_id": tid_int,
                        "class_name": cname_str,
                        "confidence": conf_float,
                    })

                    # Accumulate persistent track observations across the entire video
                    key = tid_int if tid_int is not None else f"{cname_str}_{len(all_tracked_objects)}"
                    if key not in all_tracked_objects:
                        all_tracked_objects[key] = {
                            "track_id": tid_int,
                            "class_name": cname_str,
                            "confidence": conf_float,
                            "status": "Tracked" if tid_int is not None else "Detected",
                            "first_frame": frame_idx,
                            "last_frame": frame_idx,
                            "frames_visible": 1,
                        }
                    else:
                        obj = all_tracked_objects[key]
                        obj["last_frame"] = frame_idx
                        obj["frames_visible"] += 1
                        if conf_float > obj["confidence"]:
                            obj["confidence"] = conf_float
            elif result.detections and len(result.detections.class_names) > 0:
                for cname, conf in zip(result.detections.class_names, result.detections.scores):
                    conf_float = round(float(conf), 2)
                    cname_str = str(cname)
                    class_counts[cname_str] = class_counts.get(cname_str, 0) + 1
                    active_obstacles.append({
                        "track_id": None,
                        "class_name": cname_str,
                        "confidence": conf_float,
                    })

                    key = f"{cname_str}_{frame_idx}_{len(all_tracked_objects)}"
                    all_tracked_objects[key] = {
                        "track_id": None,
                        "class_name": cname_str,
                        "confidence": conf_float,
                        "status": "Detected",
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "frames_visible": 1,
                    }

            lane_status = "Bypassed"
            lane_offset = None
            if config.enable_lanes:
                if result.lanes and result.lanes.lane_center_offset_px is not None:
                    lane_status = f"{config.lane_method.capitalize()} Active"
                    lane_offset = result.lanes.lane_center_offset_px
                else:
                    lane_status = "Tracking Lanes"

            depth_status = "Bypassed"
            if config.enable_depth:
                depth_status = "Active (Rel. Proximity)" if (result.depth is not None) else "Cadence Standby"

            if target_frames > 0:
                pct = min(99.0, round((frame_idx / target_frames) * 100.0, 1))
            else:
                pct = min(95.0, round((frame_idx / (frame_idx + 30)) * 100.0, 1))

            with _JOBS_LOCK:
                job["current_frame"] = frame_idx
                job["progress"] = pct
                job["live_metrics"] = {
                    "frame": frame_idx,
                    "total_frames": target_frames if target_frames > 0 else None,
                    "fps": round(result.effective_fps, 1),
                    "latency_ms": round(result.total_latency_ms, 1),
                    "playback_fps": round(fps_out, 2),
                    "output_duration_sec": round(frame_idx / fps_out, 2) if fps_out > 0 else 0.0,
                    "object_count": result.detections.count if result.detections else (len(result.tracks.track_ids) if result.tracks else 0),
                    "active_tracks": len(result.tracks.track_ids) if result.tracks else 0,
                    "lane_status": lane_status,
                    "lane_offset_px": lane_offset,
                    "depth_status": depth_status,
                    "hazard_status": result.hazard_status,
                    "class_counts": class_counts,
                    "active_obstacles": active_obstacles[:8],
                }

            # Cooperative thread yield: Release GIL and give CPU time to Uvicorn event loop
            # so health checks (/api/health) and polling requests respond instantly in < 1ms
            time.sleep(0.005)

    except Exception as e:
        logger.exception(f"Error processing job {job_id}: {e}")
        with _JOBS_LOCK:
            job["status"] = "failed"
            job["error"] = str(e)
    finally:
        cap.release()
        if writer:
            writer.release()
            writer = None

    # Verify generated video integrity and decodability before marking completed
    with _JOBS_LOCK:
        if job["status"] not in ("cancelled", "failed"):
            try:
                if not out_video_path.exists() or out_video_path.stat().st_size == 0:
                    raise RuntimeError("Annotated video output was not created or is empty (0 bytes).")

                # Test decodability and check that frames are not blank black
                test_cap = cv2.VideoCapture(str(out_video_path))
                if not test_cap.isOpened():
                    raise RuntimeError("Annotated video header could not be decoded by OpenCV.")

                checked_frames = 0
                has_non_black = False
                while checked_frames < 10:
                    ret_check, frame_check = test_cap.read()
                    if not ret_check or frame_check is None:
                        break
                    checked_frames += 1
                    if int(np.max(frame_check)) > 10:
                        has_non_black = True

                test_cap.release()

                if checked_frames == 0:
                    raise RuntimeError("Annotated video contains 0 readable frames.")
                if not has_non_black:
                    raise RuntimeError("Annotated video contains only solid black frames.")

            except Exception as val_err:
                logger.error(f"Video validation failed for job {job_id}: {val_err}")
                job["status"] = "failed"
                job["error"] = f"Video validation failed: {val_err}"

    # Finalize job status if not cancelled or failed
    with _JOBS_LOCK:
        if job["status"] not in ("cancelled", "failed"):
            total_duration = max(0.001, time.perf_counter() - job["start_time"])
            avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
            avg_fps = (frame_idx / total_duration) if total_duration > 0 else 0.0
            output_dur = round(frame_idx / fps_out, 2) if fps_out > 0 else 0.0
            actual_in_frames = raw_frame_idx if raw_frame_idx > 0 else total_in_frames
            input_dur = round(actual_in_frames / fps_in, 2) if (actual_in_frames > 0 and fps_in > 0) else output_dur

            # Compile unique detected/tracked objects observed throughout the video
            unique_class_counts: Dict[str, int] = {}
            for obj in all_tracked_objects.values():
                c = obj["class_name"]
                unique_class_counts[c] = unique_class_counts.get(c, 0) + 1

            detected_objects_list = sorted(
                all_tracked_objects.values(),
                key=lambda x: (
                    0 if x.get("track_id") is not None else 1,
                    x.get("track_id") if x.get("track_id") is not None else 0,
                    -x.get("frames_visible", 0),
                ),
            )
            total_unique_tracks = sum(1 for o in all_tracked_objects.values() if o.get("track_id") is not None)

            # Save telemetry log to JSON file
            telemetry_summary = {
                "job_id": job_id,
                "input_source": str(input_path),
                "summary": {
                    "processed_frames": frame_idx,
                    "total_input_frames": actual_in_frames if actual_in_frames > 0 else frame_idx,
                    "playback_fps": round(fps_out, 2),
                    "source_fps": round(fps_in, 2),
                    "stride": stride,
                    "input_duration_sec": input_dur,
                    "output_duration_sec": output_dur,
                    "processing_duration_sec": round(total_duration, 2),
                    "avg_latency_ms": round(avg_lat, 2),
                    "avg_fps": round(avg_fps, 2),
                    "avg_throughput_fps": round(avg_fps, 2),
                    "total_detected": len(all_tracked_objects),
                    "total_unique_tracks": total_unique_tracks,
                    "unique_class_counts": unique_class_counts,
                },
                "processed_frames": frame_idx,
                "total_duration_sec": round(total_duration, 2),
                "avg_latency_ms": round(avg_lat, 2),
                "avg_fps": round(avg_fps, 2),
                "total_detected": len(all_tracked_objects),
                "total_unique_tracks": total_unique_tracks,
                "unique_class_counts": unique_class_counts,
                "detected_objects": detected_objects_list,
                "config": cfg_dict,
                "frames": telemetry_records,
            }

            try:
                with open(telemetry_path, "w", encoding="utf-8") as tf:
                    json.dump(telemetry_summary, tf, indent=2)
            except Exception as e:
                logger.warning(f"Could not write telemetry file: {e}")

            job["status"] = "completed"
            job["progress"] = 100.0
            job["output_video_path"] = str(out_video_path)
            job["telemetry_path"] = str(telemetry_path)
            job["summary"] = {
                "processed_frames": frame_idx,
                "total_input_frames": actual_in_frames if actual_in_frames > 0 else frame_idx,
                "playback_fps": round(fps_out, 2),
                "source_fps": round(fps_in, 2),
                "stride": stride,
                "input_duration_sec": input_dur,
                "output_duration_sec": output_dur,
                "processing_duration_sec": round(total_duration, 2),
                "duration_sec": round(total_duration, 2),
                "avg_latency_ms": round(avg_lat, 1),
                "avg_fps": round(avg_fps, 1),
                "avg_throughput_fps": round(avg_fps, 1),
                "output_video_size_bytes": out_video_path.stat().st_size if out_video_path.exists() else 0,
                "total_detected": len(all_tracked_objects),
                "total_unique_tracks": total_unique_tracks,
                "unique_class_counts": unique_class_counts,
                "detected_objects": detected_objects_list,
            }
            logger.info(
                f"Job {job_id} successfully completed in {total_duration:.2f}s | "
                f"Processed {frame_idx} frames (stride {stride}) | Output duration: {output_dur:.2f}s ({avg_fps:.1f} FPS throughput) | "
                f"Tracks observed: {total_unique_tracks} ({len(all_tracked_objects)} total objects)"
            )


@app.post("/api/process")
async def start_processing_job(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    enable_tracking: bool = Form(True),
    enable_lanes: bool = Form(True),
    lane_method: str = Form("classical"),
    enable_depth: bool = Form(False),
    depth_freq: int = Form(5),
    conf_thresh: float = Form(0.25),
    max_frames: int = Form(0),
    model_variant: str = Form("baseline"),
) -> Dict[str, Any]:
    """Uploads a video or selects a preset sample and queues a perception processing job."""
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_video_path: Optional[Path] = None
    input_display_name = ""

    if sample_id == "demo_banner":
        sample_path = DATA_DIR / "demo_banner.mp4"
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail="Preset sample video not found on server.")
        input_video_path = sample_path
        input_display_name = "demo_banner.mp4"
    elif file is not None:
        filename = file.filename or "upload.mp4"
        valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        ext = Path(filename).suffix.lower()
        if ext not in valid_exts:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported video format '{ext}'. Supported formats: {', '.join(sorted(valid_exts))}",
            )

        saved_upload = job_dir / f"original{ext}"
        try:
            with open(saved_upload, "wb") as buf:
                shutil.copyfileobj(file.file, buf)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed saving uploaded file: {e}")

        if saved_upload.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

        input_video_path = saved_upload
        input_display_name = filename
    else:
        raise HTTPException(
            status_code=400,
            detail="No video provided. Please either upload a road video file or choose a preset sample.",
        )

    # Store Job
    job_record: Dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "current_frame": 0,
        "total_frames": max_frames if max_frames > 0 else None,
        "input_filename": input_display_name,
        "input_path": str(input_video_path),
        "job_dir": str(job_dir),
        "output_video_path": None,
        "telemetry_path": None,
        "cancel_event": threading.Event(),
        "config": {
            "enable_tracking": enable_tracking,
            "enable_lanes": enable_lanes,
            "lane_method": lane_method if lane_method in ("classical", "segmentation") else "classical",
            "enable_depth": enable_depth,
            "depth_freq": max(1, depth_freq),
            "conf_thresh": max(0.05, min(0.95, conf_thresh)),
            "max_frames": max_frames,
            "model_variant": model_variant,
            "device": "cpu",
        },
        "live_metrics": {
            "frame": 0,
            "total_frames": max_frames,
            "fps": 0.0,
            "latency_ms": 0.0,
            "object_count": 0,
            "active_tracks": 0,
            "lane_status": "Queued",
            "depth_status": "Queued",
            "hazard_status": "STATUS: QUEUED",
            "class_counts": {},
            "active_obstacles": [],
        },
        "summary": None,
        "error": None,
    }

    with _JOBS_LOCK:
        _JOBS[job_id] = job_record

    # Launch background processing worker
    worker_thread = threading.Thread(target=_run_job_worker, args=(job_id,), daemon=True)
    worker_thread.start()

    logger.info(f"Queued perception job {job_id} for file '{input_display_name}'")
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Job queued successfully",
        "input_filename": input_display_name,
    }


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str) -> Dict[str, Any]:
    """Poll job processing progress, metrics, and completion state."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

        # Build clean JSON response (omit un-serializable objects like threading.Event)
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "current_frame": job["current_frame"],
            "total_frames": job["total_frames"],
            "input_filename": job["input_filename"],
            "live_metrics": job["live_metrics"],
            "summary": job["summary"],
            "error": job["error"],
            "has_output_video": bool(job["output_video_path"] and Path(job["output_video_path"]).exists()),
            "has_telemetry": bool(job["telemetry_path"] and Path(job["telemetry_path"]).exists()),
        }


@app.get("/api/jobs/{job_id}/results")
def get_job_results(job_id: str) -> Dict[str, Any]:
    """Retrieves full analysis results, summary metrics, and detected/tracked objects list for a job."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

        summary = job.get("summary") or {}
        detected_objects = summary.get("detected_objects", [])
        unique_class_counts = summary.get("unique_class_counts", {})
        total_unique_tracks = summary.get("total_unique_tracks", sum(1 for o in detected_objects if o.get("track_id") is not None))
        total_detected = summary.get("total_detected", len(detected_objects))

        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "summary": summary,
            "detected_objects": detected_objects,
            "class_counts": unique_class_counts,
            "total_detected": total_detected,
            "total_tracked": total_unique_tracks,
            "live_metrics": job.get("live_metrics", {}),
            "has_output_video": bool(job.get("output_video_path") and Path(job["output_video_path"]).exists()),
            "has_telemetry": bool(job.get("telemetry_path") and Path(job["telemetry_path"]).exists()),
        }


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, Any]:
    """Request graceful cancellation of a running job."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        if job["status"] in ("completed", "failed", "cancelled"):
            return {"job_id": job_id, "status": job["status"], "message": f"Job already {job['status']}."}

        job["cancel_event"].set()
        job["status"] = "cancelled"

    logger.info(f"Cancellation requested for job {job_id}")
    return {"job_id": job_id, "status": "cancelled", "message": "Cancellation request acknowledged."}


@app.get("/api/jobs/{job_id}/video")
def get_job_video(job_id: str, request: Request, download: bool = Query(False)):
    """Streams or downloads the generated annotated MP4 video with HTTP 206 Partial Content support."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        vpath = job.get("output_video_path")

    if not vpath or not Path(vpath).exists():
        raise HTTPException(status_code=404, detail="Annotated video not found or not yet generated.")

    return stream_video_file(
        Path(vpath),
        request=request,
        download=download,
        filename=f"annotated_perception_{job_id}.mp4",
    )


@app.get("/api/jobs/{job_id}/original-video")
def get_job_original_video(job_id: str, request: Request, download: bool = Query(False)):
    """Streams the original uploaded/preset video for side-by-side or tab comparison with HTTP 206 support."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        ipath = job.get("input_path")

    if not ipath or not Path(ipath).exists():
        raise HTTPException(status_code=404, detail="Original video not found.")

    return stream_video_file(
        Path(ipath),
        request=request,
        download=download,
        filename=f"original_{Path(ipath).name}",
    )


@app.get("/api/jobs/{job_id}/telemetry")
def get_job_telemetry(job_id: str):
    """Downloads the full structured JSON telemetry log for a completed job."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        tpath = job.get("telemetry_path")

    if not tpath or not Path(tpath).exists():
        raise HTTPException(status_code=404, detail="Telemetry log not found or not yet generated.")

    return FileResponse(
        path=tpath,
        media_type="application/json",
        filename=f"telemetry_{job_id}.json",
    )


# Mount Frontend static directory and root index
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend_index():
        """Serves the primary HTML5 dashboard interface."""
        index_file = FRONTEND_DIR / "index.html"
        if not index_file.exists():
            return JSONResponse(
                status_code=404,
                content={"error": "frontend/index.html not found"},
            )
        return FileResponse(index_file)

    @app.get("/favicon.svg")
    @app.get("/favicon.ico")
    def serve_favicon():
        """Serves the NexDrive AI brand favicon."""
        favicon_file = FRONTEND_DIR / "favicon.svg"
        if favicon_file.exists():
            return FileResponse(favicon_file, media_type="image/svg+xml")
        raise HTTPException(status_code=404, detail="Favicon not found")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("=" * 70)
    print("AUTONOMOUS DRIVING PERCEPTION STACK -- WEB INTERFACE")
    print("=" * 70)
    print(f"  Local Dashboard URL: http://localhost:{port}")
    print(f"  REST API Base      : http://localhost:{port}/api")
    print(f"  Interactive Docs   : http://localhost:{port}/docs")
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("server:app", host=host, port=port, reload=False, log_level="info")
