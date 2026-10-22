"""
Tests for OpenCVLaneDetector module.
Covers classical computer vision lane detection, polygon fitting, and edge cases.
"""
import pytest
import cv2
import numpy as np
from pathlib import Path
from src.lanes.lane_detector import OpenCVLaneDetector, LaneDetectionResult
from src.lanes.segmentation_lane_detector import SegmentationLaneDetector

PROJECT_ROOT = Path("D:/auto_drive")
SAMPLE_IMG = PROJECT_ROOT / "data" / "sample_road.jpg"

def test_lane_detector_initialization():
    detector = OpenCVLaneDetector()
    assert detector is not None
    detector.reset()
    assert detector.prev_left is None
    assert detector.prev_right is None

def test_lane_detector_invalid_input():
    detector = OpenCVLaneDetector()
    with pytest.raises(ValueError):
        detector.detect(None)
    with pytest.raises(ValueError):
        detector.detect(np.array([]))

def test_lane_detector_blank_frame():
    detector = OpenCVLaneDetector()
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    res = detector.detect(blank)
    assert isinstance(res, LaneDetectionResult)
    # A fresh detector on blank frame should have no lines
    assert res.left_line is None or res.right_line is None or not res.has_lanes

def test_lane_detector_sample_frame():
    detector = OpenCVLaneDetector()
    img = cv2.imread(str(SAMPLE_IMG))
    res = detector.detect(img)
    assert isinstance(res, LaneDetectionResult)
    assert res.latency_ms > 0
    # Visual drawing test
    annotated = detector.draw_lanes(img, res)
    assert annotated.shape == img.shape

def test_segmentation_lane_detector_initialization():
    seg_detector = SegmentationLaneDetector(device="cpu")
    assert seg_detector is not None

def test_segmentation_lane_detector_invalid_input():
    seg_detector = SegmentationLaneDetector(device="cpu")
    with pytest.raises(ValueError):
        seg_detector.detect(None)
    with pytest.raises(ValueError):
        seg_detector.detect(np.array([]))

def test_segmentation_lane_detector_sample_frame():
    seg_detector = SegmentationLaneDetector(device="cpu")
    img = cv2.imread(str(SAMPLE_IMG))
    res = seg_detector.detect(img)
    assert isinstance(res, LaneDetectionResult)
    assert res.latency_ms > 0
    assert res.lane_polygon is not None
    annotated = seg_detector.draw_lanes(img, res)
    assert annotated.shape == img.shape

