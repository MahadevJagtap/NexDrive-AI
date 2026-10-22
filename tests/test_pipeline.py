"""
Tests for PerceptionPipeline integration.
Validates orchestration, telemetry output, and video processing.
"""
import pytest
import cv2
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.perception_pipeline import PerceptionPipeline, PipelineConfig, PipelineFrameResult
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
SAMPLE_IMG = PROJECT_ROOT / "data" / "sample_road.jpg"

def test_pipeline_config_defaults():
    cfg = PipelineConfig(model_path=str(MODEL_PATH))
    assert cfg.enable_tracking is True
    assert cfg.enable_lanes is True
    assert cfg.enable_depth is False
    assert cfg.device == "cpu"

def test_pipeline_process_single_frame():
    cfg = PipelineConfig(
        model_path=str(MODEL_PATH),
        enable_tracking=True,
        enable_lanes=True,
        enable_depth=False,
        device="cpu",
    )
    pipeline = PerceptionPipeline(config=cfg)
    img = cv2.imread(str(SAMPLE_IMG))
    res = pipeline.process_frame(img, frame_index=1)
    
    assert isinstance(res, PipelineFrameResult)
    assert res.annotated_frame.shape == img.shape
    assert res.total_latency_ms > 0
    assert res.effective_fps > 0
    assert "total_latency_ms" in res.telemetry

def test_pipeline_with_depth_enabled():
    cfg = PipelineConfig(
        model_path=str(MODEL_PATH),
        enable_tracking=False,  # Use detection mode
        enable_lanes=True,
        enable_depth=True,
        depth_frequency=1,
        device="cpu",
    )
    pipeline = PerceptionPipeline(config=cfg)
    img = cv2.imread(str(SAMPLE_IMG))
    res = pipeline.process_frame(img, frame_index=1)
    
    assert isinstance(res, PipelineFrameResult)
    assert res.depth is not None
    assert res.detections is not None
    assert "depth_latency_ms" in res.telemetry

def test_pipeline_hazard_assessment():
    cfg = PipelineConfig(
        model_path=str(MODEL_PATH),
        enable_tracking=True,
        enable_lanes=True,
        enable_depth=False,
        enable_hazard_assessment=True,
        device="cpu",
    )
    pipeline = PerceptionPipeline(config=cfg)
    img = cv2.imread(str(SAMPLE_IMG))
    res = pipeline.process_frame(img, frame_index=1)
    
    assert res.hazard_status in ["STATUS: CLEAR", "SLOW DOWN: OBSTACLE AHEAD", "HAZARD: OBSTACLE IN PATH"]
    assert "hazard_status" in res.telemetry
    assert res.telemetry["hazard_status"] == res.hazard_status

