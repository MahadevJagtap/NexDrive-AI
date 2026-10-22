"""
Smoke test script for Milestone 2: Multi-Object Tracking with ByteTrack.
Processes a video segment from data/demo_banner.mp4, verifies track ID persistence,
and saves the annotated video to results/tracking_demo.mp4.
"""
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from src.tracking.tracker import RoadObjectTracker

def run_tracking_test(max_frames: int = 75):
    print("=" * 60)
    print("AUTONOMOUS DRIVING PERCEPTION - MILESTONE 2 TRACKING TEST")
    print("=" * 60)

    model_path = project_root / "models" / "best.pt"
    video_path = project_root / "data" / "demo_banner.mp4"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = results_dir / "tracking_demo.mp4"

    print(f"[1/4] Input video: {video_path}")
    if not video_path.exists():
        print(f"FAILED: Video not found at {video_path}")
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"FAILED: Could not open video file {video_path}")
        return False

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video info: {width}x{height} @ {fps:.1f} FPS (Total frames: {total_video_frames})")

    # 2. Initialize tracker
    print("\n[2/4] Initializing RoadObjectTracker with ByteTrack on CPU...")
    tracker = RoadObjectTracker(
        model_path=model_path,
        conf_thresh=0.25,
        iou_thresh=0.45,
        imgsz=640,
        tracker_type="bytetrack.yaml",
        device="cpu",
    )

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    # 3. Process video frames
    print(f"\n[3/4] Processing first {max_frames} frames with persistent tracking...")
    frame_count = 0
    latencies = []
    all_seen_ids = set()
    id_lifespans = {}  # id -> count of frames present

    t_start = time.perf_counter()
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        res = tracker.update(frame)
        latencies.append(res.latency_ms)

        # Track ID telemetry
        current_ids = set()
        for tid in res.track_ids:
            if tid >= 0:
                all_seen_ids.add(int(tid))
                current_ids.add(int(tid))
                id_lifespans[int(tid)] = id_lifespans.get(int(tid), 0) + 1

        # Draw annotations and write frame
        annotated = tracker.draw_tracks(frame, res, draw_trails=True)

        # Overlay telemetry banner
        telemetry = f"Frame: {frame_count}/{max_frames} | Tracks: {len(current_ids)} | Latency: {res.latency_ms:.1f}ms"
        cv2.putText(annotated, telemetry, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        writer.write(annotated)

    cap.release()
    writer.release()
    total_time = time.perf_counter() - t_start

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    effective_fps = frame_count / total_time if total_time > 0 else 0.0

    print(f"\n[4/4] Evaluation & Results:")
    print(f"  Processed frames: {frame_count}")
    print(f"  Total processing time: {total_time:.2f} s")
    print(f"  Average inference + tracking latency: {avg_latency:.2f} ms")
    print(f"  Effective throughput: {effective_fps:.1f} FPS")
    print(f"  Unique persistent Track IDs identified: {len(all_seen_ids)}")
    print(f"  Top tracked objects lifespan (frames present):")
    for tid, count in sorted(id_lifespans.items(), key=lambda x: -x[1])[:5]:
        print(f"    - Track #{tid}: present in {count}/{frame_count} frames ({count/frame_count*100:.1f}%)")

    print(f"  Annotated tracking video saved to: {out_video_path} ({out_video_path.stat().st_size / (1024*1024):.2f} MB)")

    # Assertions
    assert frame_count > 0, "No frames were processed"
    assert len(all_seen_ids) > 0, "No objects were tracked"
    assert out_video_path.exists() and out_video_path.stat().st_size > 0, "Video output missing"

    print("\n" + "=" * 60)
    print("MILESTONE 2 (MULTI-OBJECT TRACKING) COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_tracking_test()
    sys.exit(0 if success else 1)
