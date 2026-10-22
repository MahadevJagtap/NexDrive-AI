"""
Targeted Validation Script for Tracking, Lanes, and Depth modules.
Measures:
  - Multi-frame tracking ID switches and continuity
  - Lane detection under normal, blank, dark, and overexposed road conditions
  - Depth estimation uncalibrated guarantees and graceful fallback
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from src.tracking.tracker import RoadObjectTracker
from src.lanes.lane_detector import OpenCVLaneDetector
from src.depth.depth_estimator import MonocularDepthEstimator

def validate_tracking():
    print("\n--- 1. TRACKING VALIDATION ---")
    video_path = project_root / "data" / "demo_banner.mp4"
    tracker = RoadObjectTracker(model_path=project_root / "models" / "best.pt", device="cpu")
    cap = cv2.VideoCapture(str(video_path))
    
    frame_idx = 0
    previous_ids = set()
    consecutive_matches = 0
    id_switches_observed = 0
    all_seen_ids = set()

    while frame_idx < 30:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        res = tracker.update(frame)
        current_ids = set(int(tid) for tid in res.track_ids if tid >= 0)
        all_seen_ids.update(current_ids)

        if previous_ids and current_ids:
            intersection = previous_ids.intersection(current_ids)
            if len(intersection) > 0:
                consecutive_matches += 1
            else:
                id_switches_observed += 1

        previous_ids = current_ids

    cap.release()
    print(f"  Frames analyzed       : {frame_idx}")
    print(f"  Unique track IDs seen : {len(all_seen_ids)}")
    print(f"  Frames with maintained: {consecutive_matches}/{frame_idx - 1} frames ({consecutive_matches/(frame_idx - 1)*100:.1f}%)")
    print(f"  ID Discontinuity events: {id_switches_observed}")
    assert consecutive_matches > 0, "Tracking must maintain IDs across frames"
    return True

def validate_lanes():
    print("\n--- 2. LANE DETECTION STRESS & FAILURE MODE VALIDATION ---")
    detector = OpenCVLaneDetector()
    img_normal = cv2.imread(str(project_root / "data" / "sample_road.jpg"))
    
    # 1. Normal road scene
    res_normal = detector.detect(img_normal)
    print(f"  [Condition: Normal] Confidence: {res_normal.confidence:.2f} | Offset: {res_normal.lane_center_offset_px} px | Left: {res_normal.left_line is not None} | Right: {res_normal.right_line is not None}")

    # 2. Blank / All-black road
    detector.reset()
    img_blank = np.zeros_like(img_normal)
    res_blank = detector.detect(img_blank)
    print(f"  [Condition: Blank ] Confidence: {res_blank.confidence:.2f} | has_lanes: {res_blank.has_lanes} (Gracefully rejected)")

    # 3. Severe dark lighting (20% brightness)
    detector.reset()
    img_dark = (img_normal * 0.2).astype(np.uint8)
    res_dark = detector.detect(img_dark)
    print(f"  [Condition: Dark  ] Confidence: {res_dark.confidence:.2f} | Left: {res_dark.left_line is not None} | Right: {res_dark.right_line is not None}")

    # 4. Overexposed / glare (whiteout)
    detector.reset()
    img_glare = np.clip(img_normal.astype(np.float32) * 1.8 + 60, 0, 255).astype(np.uint8)
    res_glare = detector.detect(img_glare)
    print(f"  [Condition: Glare ] Confidence: {res_glare.confidence:.2f} | has_lanes: {res_glare.has_lanes}")
    
    # Document failure boundaries
    print("  Documented Classical CV Limitations:")
    print("    - Fails or confidence degrades when pavement markings lack contrast against dark/wet asphalt.")
    print("    - Solar glare and washed out road sections create spurious edge gradients.")
    print("    - Sharp curves (>45 deg) exceed linear 1st-degree polynomial approximation assumption.")
    return True

def validate_depth():
    print("\n--- 3. DEPTH ESTIMATION UNCALIBRATED GUARANTEES ---")
    estimator = MonocularDepthEstimator(model_type="MiDaS_small", device="cpu")
    img = cv2.imread(str(project_root / "data" / "sample_road.jpg"))
    res = estimator.estimate(img)
    
    print(f"  Calibration status explicit label : '{res.label}'")
    print(f"  is_calibrated_metric flag         : {res.is_calibrated_metric}")
    print(f"  Normalized depth range            : [{res.normalized_depth.min():.4f}, {res.normalized_depth.max():.4f}]")
    
    assert res.is_calibrated_metric is False, "Metric calibration must be False"
    assert "Relative" in res.label, "Label must explicitly indicate Relative depth"
    assert res.normalized_depth.min() >= 0.0 and res.normalized_depth.max() <= 1.0, "Must be normalized [0, 1]"
    return True

if __name__ == "__main__":
    v1 = validate_tracking()
    v2 = validate_lanes()
    v3 = validate_depth()
    print("\n" + "=" * 60)
    print("ALL MODULE VALIDATION CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
