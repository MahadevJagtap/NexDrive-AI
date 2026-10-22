"""
Smoke test script for Milestone 1: Pretrained YOLOv8 Detector.
Validates loading, inference on real driving image, error handling, and visual output.
"""
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from src.detection.detector import RoadObjectDetector

def run_smoke_test():
    print("=" * 60)
    print("AUTONOMOUS DRIVING PERCEPTION - MILESTONE 1 SMOKE TEST")
    print("=" * 60)

    # 1. Verify model checkpoint exists
    model_path = project_root / "models" / "best.pt"
    print(f"[1/5] Checking model checkpoint: {model_path}")
    if not model_path.exists():
        print(f"FAILED: Checkpoint not found at {model_path}")
        return False
    print(f"  Checkpoint size: {model_path.stat().st_size / (1024*1024):.2f} MB")

    # 2. Initialize detector
    print("\n[2/5] Initializing RoadObjectDetector on CPU...")
    t0 = time.perf_counter()
    detector = RoadObjectDetector(
        model_path=model_path,
        conf_thresh=0.25,
        iou_thresh=0.45,
        imgsz=640,
        device="cpu",
    )
    init_time = (time.perf_counter() - t0) * 1000.0
    print(f"  Detector initialized in {init_time:.2f} ms")
    print(f"  Supported classes count: {len(detector.names)}")

    # 3. Test on real road driving sample image
    sample_image_path = project_root / "data" / "sample_road.jpg"
    print(f"\n[3/5] Testing inference on sample image: {sample_image_path}")
    if not sample_image_path.exists():
        print(f"FAILED: Sample image not found at {sample_image_path}")
        return False

    img = cv2.imread(str(sample_image_path))
    h, w, c = img.shape
    print(f"  Input image dimensions: {w}x{h} (channels: {c})")

    # Warmup
    _ = detector.detect(img)

    # Measured inference
    latencies = []
    for _ in range(3):
        res = detector.detect(img)
        latencies.append(res.inference_time_ms)

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    print(f"  Inference completed successfully!")
    print(f"  Detections found: {res.count}")
    print(f"  Average inference latency: {avg_latency:.2f} ms (~{fps:.1f} FPS on CPU)")

    # Print summary of detected classes
    class_counts = {}
    for cname in res.class_names:
        class_counts[cname] = class_counts.get(cname, 0) + 1
    for cname, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"    - {cname}: {count}")

    # 4. Save annotated visual output
    print("\n[4/5] Rendering and saving annotated visual output...")
    annotated = detector.draw_detections(img, res)
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "smoke_test_detection.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"  Saved annotated output: {out_path} ({out_path.stat().st_size} bytes)")

    # 5. Robustness and error handling checks
    print("\n[5/5] Testing robustness and error handling...")
    # Empty / blank image
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    blank_res = detector.detect(blank)
    assert blank_res.is_empty(), "Blank image should produce 0 detections"
    print("  - Blank image handled correctly: 0 detections returned")

    # Invalid input
    try:
        detector.detect(None)
        print("  FAILED: detector accepted None")
        return False
    except ValueError:
        print("  - None frame correctly rejected with ValueError")

    print("\n" + "=" * 60)
    print("ALL MILESTONE 1 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
