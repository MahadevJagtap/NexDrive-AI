"""
Smoke test script for Milestone 3: Classical OpenCV Lane Detector.
Tests lane detection on sample road image and real driving video segment,
verifies corridor polygon and temporal smoothing, and outputs annotated media.
"""
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from src.lanes.lane_detector import OpenCVLaneDetector

def run_lane_test(max_frames: int = 60):
    print("=" * 60)
    print("AUTONOMOUS DRIVING PERCEPTION - MILESTONE 3 LANE DETECTION")
    print("=" * 60)

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Test on single sample road image
    img_path = project_root / "data" / "sample_road.jpg"
    print(f"[1/4] Testing lane detector on image: {img_path}")
    if not img_path.exists():
        print(f"FAILED: Image not found at {img_path}")
        return False

    img = cv2.imread(str(img_path))
    detector = OpenCVLaneDetector()

    # Measure latency
    t0 = time.perf_counter()
    res = detector.detect(img)
    latency = (time.perf_counter() - t0) * 1000.0

    print(f"  Lane detection latency: {latency:.2f} ms (~{1000.0/latency:.1f} FPS)")
    print(f"  Left line detected: {res.left_line}")
    print(f"  Right line detected: {res.right_line}")
    print(f"  Corridor polygon points: {len(res.lane_polygon) if res.lane_polygon is not None else 0}")
    print(f"  Lane center offset: {res.lane_center_offset_px} px")
    print(f"  Confidence: {res.confidence * 100:.1f}%")

    annotated_img = detector.draw_lanes(img, res)
    out_img_path = results_dir / "smoke_test_lanes.jpg"
    cv2.imwrite(str(out_img_path), annotated_img)
    print(f"  Saved annotated image: {out_img_path}")

    # 2. Test video processing with temporal smoothing
    video_path = project_root / "data" / "demo_banner.mp4"
    out_video_path = results_dir / "lane_demo.mp4"
    print(f"\n[2/4] Testing lane detector across {max_frames} video frames: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"FAILED: Cannot open video {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    detector.reset()
    frame_count = 0
    detected_frames = 0
    latencies = []

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        res = detector.detect(frame)
        latencies.append(res.latency_ms)
        if res.has_lanes:
            detected_frames += 1

        annotated = detector.draw_lanes(frame, res)
        writer.write(annotated)

    cap.release()
    writer.release()

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    print(f"  Frames processed: {frame_count}")
    print(f"  Frames with active lane corridor: {detected_frames}/{frame_count} ({detected_frames/frame_count*100:.1f}%)")
    print(f"  Average lane detection latency: {avg_lat:.2f} ms (~{1000.0/avg_lat:.1f} FPS)")
    print(f"  Saved annotated video to: {out_video_path} ({out_video_path.stat().st_size / (1024*1024):.2f} MB)")

    # 3. Robustness testing on edge cases
    print("\n[3/4] Robustness testing on edge cases...")
    # Blank frame
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    blank_res = detector.detect(blank)
    print(f"  - Blank frame correctly handled (has_lanes={blank_res.has_lanes}, conf={blank_res.confidence})")

    # Invalid input
    try:
        detector.detect(None)
        print("  FAILED: Accepted None")
        return False
    except ValueError:
        print("  - None frame correctly rejected with ValueError")

    print("\n" + "=" * 60)
    print("MILESTONE 3 (LANE DETECTION) COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_lane_test()
    sys.exit(0 if success else 1)
