"""
Tests for RoadObjectTracker module.
Validates ByteTrack multi-object tracking and track ID persistence.
"""
import pytest
import cv2
import numpy as np
from pathlib import Path
from src.tracking.tracker import RoadObjectTracker, TrackingResult

PROJECT_ROOT = Path("D:/auto_drive")
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
SAMPLE_IMG = PROJECT_ROOT / "data" / "sample_road.jpg"

def test_tracker_initialization_and_reset():
    tracker = RoadObjectTracker(model_path=MODEL_PATH, device="cpu")
    assert tracker is not None
    tracker.reset()
    assert len(tracker.trail_history) == 0
    assert len(tracker.active_ids) == 0

def test_tracker_single_frame():
    tracker = RoadObjectTracker(model_path=MODEL_PATH, device="cpu")
    img = cv2.imread(str(SAMPLE_IMG))
    res = tracker.update(img)
    assert isinstance(res, TrackingResult)
    assert res.count > 0
    assert res.track_ids.shape == (res.count,)
    assert len(res.class_names) == res.count

def test_tracker_multi_frame_persistence():
    tracker = RoadObjectTracker(model_path=MODEL_PATH, device="cpu")
    img = cv2.imread(str(SAMPLE_IMG))
    
    # Process the same frame twice to simulate consecutive video frames
    res1 = tracker.update(img)
    res2 = tracker.update(img)
    
    # Active tracks should persist
    assert len(tracker.active_ids) > 0
    # Common persistent track IDs between frame 1 and frame 2
    common_ids = set(res1.track_ids).intersection(set(res2.track_ids))
    # At least some objects should maintain persistent IDs across consecutive frames
    valid_common = [tid for tid in common_ids if tid >= 0]
    assert len(valid_common) > 0, "Tracking should preserve IDs across consecutive frames"
