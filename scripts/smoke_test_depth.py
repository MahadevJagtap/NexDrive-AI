"""
Smoke test script for Milestone 4: Monocular Depth Estimation (MiDaS_small).
Validates relative depth map generation, object depth association, and side-by-side output.
"""
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from src.depth.depth_estimator import MonocularDepthEstimator
from src.detection.detector import RoadObjectDetector

def run_depth_test():
    print("=" * 60)
    print("AUTONOMOUS DRIVING PERCEPTION - MILESTONE 4 DEPTH TEST")
    print("=" * 60)

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Depth Estimator
    print("[1/4] Initializing MonocularDepthEstimator (MiDaS_small on CPU)...")
    t0 = time.perf_counter()
    depth_estimator = MonocularDepthEstimator(
        model_type="MiDaS_small",
        device="cpu",
    )
    init_time = (time.perf_counter() - t0) * 1000.0
    print(f"  Initialized in {init_time:.2f} ms")

    # 2. Test on sample image
    img_path = project_root / "data" / "sample_road.jpg"
    print(f"\n[2/4] Estimating depth on sample image: {img_path}")
    if not img_path.exists():
        print(f"FAILED: Image not found at {img_path}")
        return False

    img = cv2.imread(str(img_path))

    # Warmup
    _ = depth_estimator.estimate(img)

    # Measured run
    res = depth_estimator.estimate(img)
    print(f"  Depth estimation completed successfully!")
    print(f"  Inference latency: {res.latency_ms:.2f} ms (~{1000.0/res.latency_ms:.1f} FPS)")
    print(f"  Depth map dimensions: {res.normalized_depth.shape}")
    print(f"  Min normalized depth: {res.normalized_depth.min():.4f}, Max: {res.normalized_depth.max():.4f}")
    print(f"  Calibration status: {res.label} (Metric: {res.is_calibrated_metric})")

    # 3. Associate depth with detected objects
    print("\n[3/4] Associating relative proximity with detected road objects...")
    detector = RoadObjectDetector(model_path=project_root / "models" / "best.pt", device="cpu")
    det_res = detector.detect(img)

    annotated = img.copy()
    print(f"  Detections found: {det_res.count}")
    for i, (box, cname, score) in enumerate(zip(det_res.boxes, det_res.class_names, det_res.scores)):
        prox = depth_estimator.get_object_proximity(res, tuple(box))
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cname} | Rel. Prox: {prox:.2f}"
        cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        if i < 5:
            print(f"    - Object {i+1} [{cname}]: Relative Proximity = {prox:.2f} (0=far, 1=close)")

    # 4. Save side-by-side composite
    print("\n[4/4] Creating side-by-side visual output...")
    composite = depth_estimator.create_side_by_side(annotated, res)
    out_path = results_dir / "smoke_test_depth.jpg"
    cv2.imwrite(str(out_path), composite)
    print(f"  Saved composite visual: {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")

    # Assertions
    assert res.normalized_depth.shape == img.shape[:2], "Depth shape mismatch"
    assert not res.is_calibrated_metric, "Depth must be explicitly marked as relative"
    assert out_path.exists() and out_path.stat().st_size > 0, "Visual output missing"

    print("\n" + "=" * 60)
    print("MILESTONE 4 (DEPTH ESTIMATION) COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_depth_test()
    sys.exit(0 if success else 1)
