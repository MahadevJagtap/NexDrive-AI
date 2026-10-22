"""
Tests for MonocularDepthEstimator module.
Verifies relative uncalibrated depth map output and object proximity association.
"""
import pytest
import cv2
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.depth.depth_estimator import MonocularDepthEstimator, DepthResult
SAMPLE_IMG = PROJECT_ROOT / "data" / "sample_road.jpg"

def test_depth_estimator_initialization():
    estimator = MonocularDepthEstimator(model_type="MiDaS_small", device="cpu")
    assert estimator is not None

def test_depth_estimator_invalid_input():
    estimator = MonocularDepthEstimator(model_type="MiDaS_small", device="cpu")
    with pytest.raises(ValueError):
        estimator.estimate(None)
    with pytest.raises(ValueError):
        estimator.estimate(np.array([]))

def test_depth_estimator_real_frame():
    estimator = MonocularDepthEstimator(model_type="MiDaS_small", device="cpu")
    img = cv2.imread(str(SAMPLE_IMG))
    res = estimator.estimate(img)
    assert isinstance(res, DepthResult)
    assert res.normalized_depth.shape == img.shape[:2]
    assert res.colored_map.shape == img.shape
    assert res.latency_ms > 0
    assert not res.is_calibrated_metric, "Depth must be explicitly uncalibrated"
    assert "Relative" in res.label

def test_object_proximity():
    estimator = MonocularDepthEstimator(model_type="MiDaS_small", device="cpu")
    img = cv2.imread(str(SAMPLE_IMG))
    res = estimator.estimate(img)
    # Test bounding box at center of frame
    h, w = img.shape[:2]
    box = (w // 4, h // 2, 3 * w // 4, h - 20)
    prox = estimator.get_object_proximity(res, box)
    assert 0.0 <= prox <= 1.0, "Proximity score must be in [0, 1]"
