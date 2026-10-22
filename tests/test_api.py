"""
Automated Test Suite for FastAPI Backend and Perception Pipeline API.
Validates health, sample listings, validation error guards, job lifecycle, and downloads.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server import app, _JOBS

client = TestClient(app)


def test_api_health_endpoint():
    """Verify backend health endpoint returns online status and model availability."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["healthy"] is True
    assert data["ready"] is True
    assert data["device"] == "cpu"
    assert "models" in data
    assert data["models"]["baseline_yolov8n"] is True


def test_render_health_endpoint():
    """Verify /health endpoint probed by Render returns HTTP 200 and healthy flag."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["healthy"] is True
    assert data["ready"] is True


def test_healthz_endpoint():
    """Verify standard Kubernetes / container /healthz endpoint returns HTTP 200."""
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json()["ready"] is True


def test_health_check_degraded_when_model_missing(monkeypatch, tmp_path):
    """Verify /health returns HTTP 503 if baseline model checkpoint is missing."""
    import server
    monkeypatch.setattr(server, "MODELS_DIR", tmp_path / "empty_models")
    res = client.get("/health")
    assert res.status_code == 503
    data = res.json()
    assert data["ready"] is False
    assert data["status"] == "degraded"
    assert data["models"]["baseline_yolov8n"] is False


def test_api_samples_endpoint():
    """Verify available sample videos listing."""
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    sample = data[0]
    assert sample["id"] == "demo_banner"
    assert "frames" in sample
    assert "resolution" in sample
    assert Path(sample["path"]).exists()


def test_api_process_validation_no_input():
    """Verify 400 error when neither file nor sample is provided."""
    res = client.post("/api/process", data={})
    assert res.status_code == 400
    assert "No video provided" in res.json()["detail"]


def test_api_process_validation_invalid_file_extension():
    """Verify 400 error when uploading non-video file."""
    fake_file = io.BytesIO(b"fake text content")
    res = client.post(
        "/api/process",
        files={"file": ("malicious.exe", fake_file, "application/octet-stream")},
        data={"max_frames": 5},
    )
    assert res.status_code == 400
    assert "Unsupported video format" in res.json()["detail"]


def test_api_job_not_found():
    """Verify 404 response when querying nonexistent job."""
    res = client.get("/api/jobs/nonexistent_id_999")
    assert res.status_code == 404


def test_api_process_sample_submission_and_lifecycle():
    """
    Integration test: Submits a job with sample video on 3 frames,
    polls status until completion, and verifies video and telemetry download endpoints.
    """
    res = client.post(
        "/api/process",
        data={
            "sample_id": "demo_banner",
            "enable_tracking": "true",
            "enable_lanes": "true",
            "lane_method": "classical",
            "enable_depth": "false",
            "conf_thresh": 0.25,
            "max_frames": 3,
            "model_variant": "baseline",
        },
    )
    assert res.status_code == 200
    job_info = res.json()
    assert "job_id" in job_info
    job_id = job_info["job_id"]
    assert job_info["status"] == "queued"

    # Poll for completion with timeout
    completed = False
    data = None
    for _ in range(60):  # Up to 30 seconds
        time.sleep(0.5)
        poll_res = client.get(f"/api/jobs/{job_id}")
        assert poll_res.status_code == 200
        data = poll_res.json()
        if data["status"] in ("completed", "failed", "cancelled"):
            completed = True
            break

    assert completed is True, f"Job did not complete in time, state: {data}"
    assert data["status"] == "completed"
    assert data["progress"] == 100.0
    assert data["current_frame"] == 3
    assert data["has_output_video"] is True
    assert data["has_telemetry"] is True

    # Test streaming output video with full response
    vid_res = client.get(f"/api/jobs/{job_id}/video")
    assert vid_res.status_code == 200
    assert "video/mp4" in vid_res.headers.get("content-type", "")
    assert len(vid_res.content) > 0

    # Test HTTP 206 Partial Content Range streaming
    range_res = client.get(f"/api/jobs/{job_id}/video", headers={"Range": "bytes=0-1024"})
    assert range_res.status_code == 206
    assert "Content-Range" in range_res.headers
    assert range_res.headers["Content-Range"].startswith("bytes 0-1024/")
    assert len(range_res.content) == 1025

    # Test downloading video as attachment (?download=1)
    dl_res = client.get(f"/api/jobs/{job_id}/video?download=1")
    assert dl_res.status_code == 200
    assert "attachment" in dl_res.headers.get("Content-Disposition", "")

    # Verify generated video frames with OpenCV (non-black, valid dimensions)
    import cv2, numpy as np, tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tf.write(vid_res.content)
        temp_path = tf.name

    test_cap = cv2.VideoCapture(temp_path)
    assert test_cap.isOpened()
    ret_read, frame_read = test_cap.read()
    test_cap.release()
    Path(temp_path).unlink(missing_ok=True)

    assert ret_read is True
    assert frame_read is not None
    assert int(np.max(frame_read)) > 20, "Video frame must not be completely black"

    # Test downloading telemetry log
    tel_res = client.get(f"/api/jobs/{job_id}/telemetry")
    assert tel_res.status_code == 200
    assert tel_res.headers.get("content-type") == "application/json"
    tel_data = tel_res.json()
    assert "job_id" in tel_data
    assert len(tel_data["frames"]) == 3

    # Test dedicated /api/jobs/{job_id}/results endpoint
    results_res = client.get(f"/api/jobs/{job_id}/results")
    assert results_res.status_code == 200
    results_data = results_res.json()
    assert results_data["job_id"] == job_id
    assert results_data["status"] == "completed"
    assert "detected_objects" in results_data
    assert isinstance(results_data["detected_objects"], list)
    assert "total_detected" in results_data
    assert "total_tracked" in results_data
    assert "class_counts" in results_data

    # If objects were detected, verify fields
    for obj in results_data["detected_objects"]:
        assert "class_name" in obj
        assert "confidence" in obj
        assert "track_id" in obj
        assert "status" in obj


def test_api_process_with_segmentation_and_depth():
    """Verify execution with deep segmentation lanes and depth enabled."""
    res = client.post(
        "/api/process",
        data={
            "sample_id": "demo_banner",
            "enable_tracking": "false",
            "enable_lanes": "true",
            "lane_method": "segmentation",
            "enable_depth": "true",
            "depth_freq": 1,
            "conf_thresh": 0.3,
            "max_frames": 2,
            "model_variant": "baseline",
        },
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    # Poll until completed
    completed = False
    data = None
    for _ in range(60):
        time.sleep(0.5)
        poll_res = client.get(f"/api/jobs/{job_id}")
        data = poll_res.json()
        if data["status"] in ("completed", "failed", "cancelled"):
            completed = True
            break

    assert completed is True
    assert data["status"] == "completed"
    assert data["current_frame"] == 2


def test_frontend_index_serving():
    """Verify GET / serves HTML dashboard."""
    res = client.get("/")
    assert res.status_code == 200
    assert "Autonomous Driving Perception Stack" in res.text
    assert "video-viewport" in res.text
