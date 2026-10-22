"""
Tests for RoadObjectDetector module.
Covers missing checkpoint error handling, real inference, and boundary conditions.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import RoadObjectDetector, DetectionResult
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
SAMPLE_IMG = PROJECT_ROOT / "data" / "sample_road.jpg"

def test_missing_checkpoint_raises_error():
    with pytest.raises(FileNotFoundError):
        _ = RoadObjectDetector(model_path="non_existent_weights.pt")

def test_detector_initialization():
    assert MODEL_PATH.exists(), "Model checkpoint must exist"
    detector = RoadObjectDetector(model_path=MODEL_PATH, device="cpu")
    assert detector is not None
    assert len(detector.names) == 10
    assert 2 in detector.names and detector.names[2] == "car"

def test_detector_invalid_input():
    detector = RoadObjectDetector(model_path=MODEL_PATH, device="cpu")
    with pytest.raises(ValueError):
        detector.detect(None)
    with pytest.raises(ValueError):
        detector.detect(np.array([]))

def test_detector_blank_image():
    detector = RoadObjectDetector(model_path=MODEL_PATH, device="cpu")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(blank)
    assert isinstance(res, DetectionResult)
    assert res.is_empty()
    assert res.count == 0

def test_detector_real_inference():
    assert SAMPLE_IMG.exists(), "Sample road image must exist"
    detector = RoadObjectDetector(model_path=MODEL_PATH, conf_thresh=0.25, device="cpu")
    res = detector.detect(SAMPLE_IMG)
    assert isinstance(res, DetectionResult)
    assert res.count > 0, "Should detect road objects in sample image"
    assert "car" in res.class_names
    assert res.inference_time_ms > 0
    assert res.boxes.shape == (res.count, 4)
    assert res.scores.shape == (res.count,)

def test_draw_detections():
    detector = RoadObjectDetector(model_path=MODEL_PATH, device="cpu")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_res = DetectionResult(
        boxes=np.array([[50, 50, 150, 150]], dtype=np.float32),
        scores=np.array([0.85], dtype=np.float32),
        class_ids=np.array([2], dtype=np.int32),
        class_names=["car"],
        inference_time_ms=25.0,
    )
    annotated = detector.draw_detections(blank, dummy_res)
    assert annotated.shape == blank.shape
    assert not np.array_equal(annotated, blank), "Drawing should modify canvas"

def test_finetuned_detector_inference():
    ft_checkpoint = PROJECT_ROOT / "models" / "finetuned_best.pt"
    if ft_checkpoint.exists():
        detector = RoadObjectDetector(model_path=ft_checkpoint, conf_thresh=0.25, device="cpu")
        res = detector.detect(SAMPLE_IMG)
        assert isinstance(res, DetectionResult)
        assert res.count > 0, "Fine-tuned model should detect objects in sample road image"
        assert len(detector.names) == 10

