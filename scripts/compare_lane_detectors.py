"""
Comparative Evaluation: Classical OpenCV Lane Detection vs Deep Learning Segmentation.

Compares:
1. Classical Computer Vision (OpenCVLaneDetector - HLS/HSV + Canny + Hough)
2. Deep Semantic Segmentation (SegmentationLaneDetector - LRASPP-MobileNetV3)

Generates:
- Quantitative latency and corridor telemetry: results/lane_comparison.csv
- Side-by-side visual comparison artifact:   results/lane_comparison.jpg
"""
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

BASE_DIR = Path("D:/auto_drive")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.lanes.lane_detector import OpenCVLaneDetector
from src.lanes.segmentation_lane_detector import SegmentationLaneDetector

SAMPLE_IMG_PATH = BASE_DIR / "data" / "sample_road.jpg"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = RESULTS_DIR / "lane_comparison.csv"
VIS_OUT = RESULTS_DIR / "lane_comparison.jpg"

def benchmark_detector(detector, img, iterations=15, warmup=5):
    for _ in range(warmup):
        _ = detector.detect(img)
    times = []
    res = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = detector.detect(img)
        times.append((time.perf_counter() - t0) * 1000.0)
    mean_lat = float(np.mean(times))
    median_lat = float(np.median(times))
    fps = 1000.0 / median_lat if median_lat > 0 else 0.0
    return res, mean_lat, median_lat, fps

def main():
    print("=================================================================", flush=True)
    print("EMPIRICAL COMPARISON: CLASSICAL OPENCV VS SEGMENTATION LANE MODEL", flush=True)
    print("=================================================================", flush=True)

    img = cv2.imread(str(SAMPLE_IMG_PATH))
    if img is None:
        print(f"Error: Could not load sample road image from {SAMPLE_IMG_PATH}", flush=True)
        sys.exit(1)

    print("1. Initializing Detectors...", flush=True)
    classical = OpenCVLaneDetector()
    segmentation = SegmentationLaneDetector(device="cpu")

    print("\n2. Benchmarking Classical OpenCV Lane Detector (15 iterations)...", flush=True)
    res_classic, mean_lat_c, median_lat_c, fps_c = benchmark_detector(classical, img)

    print("3. Benchmarking Deep Segmentation Lane Detector (15 iterations)...", flush=True)
    res_seg, mean_lat_s, median_lat_s, fps_s = benchmark_detector(segmentation, img)

    # Calculate corridor pixel areas
    area_c = cv2.contourArea(res_classic.lane_polygon) if res_classic.lane_polygon is not None else 0
    area_s = cv2.contourArea(res_seg.lane_polygon) if res_seg.lane_polygon is not None else 0

    print("\n--- Empirical Results Summary ---", flush=True)
    data = [
        {
            "Method": "Classical OpenCV (HLS/HSV+Canny+Hough)",
            "Model Type": "Parametric Edge Geometry",
            "Weights Size (MB)": 0.0,
            "Mean Latency (ms)": round(mean_lat_c, 2),
            "Median Latency (ms)": round(median_lat_c, 2),
            "Throughput (FPS)": round(fps_c, 2),
            "Lane Center Offset (px)": res_classic.lane_center_offset_px,
            "Corridor Area (px)": int(area_c),
            "Illumination Resilience": "Low-Moderate (Contrast Dependent)",
            "Hardware Footprint": "Minimal (Pure CPU instructions)",
        },
        {
            "Method": "Deep Segmentation (LRASPP-MobileNetV3)",
            "Model Type": "Semantic Neural Segmentation",
            "Weights Size (MB)": 12.5,
            "Mean Latency (ms)": round(mean_lat_s, 2),
            "Median Latency (ms)": round(median_lat_s, 2),
            "Throughput (FPS)": round(fps_s, 2),
            "Lane Center Offset (px)": res_seg.lane_center_offset_px,
            "Corridor Area (px)": int(area_s),
            "Illumination Resilience": "High (Learned Semantic Features)",
            "Hardware Footprint": "Moderate (Neural activation memory)",
        },
    ]

    df = pd.DataFrame(data)
    df.to_csv(CSV_OUT, index=False)
    print(df.to_string(index=False), flush=True)
    print(f"\nSaved quantitative comparison to: {CSV_OUT}", flush=True)

    # 4. Generate Visual Side-by-Side Comparison Artifact
    print("\n4. Generating Side-by-Side Visual Artifact...", flush=True)
    vis_c = classical.draw_lanes(img, res_classic)
    vis_s = segmentation.draw_lanes(img, res_seg)

    h, w = img.shape[:2]
    header_h = 60
    header_c = np.zeros((header_h, w, 3), dtype=np.uint8) + 35
    header_s = np.zeros((header_h, w, 3), dtype=np.uint8) + 35

    cv2.putText(header_c, f"Classical OpenCV (HLS/Canny/Hough) - {median_lat_c:.1f}ms ({fps_c:.1f} FPS)",
                (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 120), 2)
    cv2.putText(header_s, f"Deep Segmentation (LRASPP-MobileNetV3) - {median_lat_s:.1f}ms ({fps_s:.1f} FPS)",
                (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 200, 0), 2)

    panel_c = np.vstack([header_c, vis_c])
    panel_s = np.vstack([header_s, vis_s])

    # Resize panels if too wide to display side by side
    composite = np.hstack([panel_c, panel_s])
    composite_resized = cv2.resize(composite, (1920, int(1920 * (h + header_h) / (w * 2))))
    cv2.imwrite(str(VIS_OUT), composite_resized)
    print(f"Saved visual comparison artifact to: {VIS_OUT}", flush=True)

if __name__ == "__main__":
    main()
