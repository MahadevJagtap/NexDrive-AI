"""
Rigorous Benchmark Reconciliation Script for Autonomous Driving Perception Stack.
Separates:
  1. Pure model forward inference (640x640 tensor)
  2. Full model predict (image input + letterbox + forward + NMS)
  3. ONNX Runtime CPU forward inference
  4. End-to-end perception pipeline latency by active module
Saves unified results to results/reconciled_benchmark.csv.
"""
import csv
import gc
import platform
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import onnxruntime as ort
import psutil
import torch
from ultralytics import YOLO

from src.detection.detector import RoadObjectDetector
from src.pipeline.perception_pipeline import PerceptionPipeline, PipelineConfig

def get_process_memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)

def main():
    print("=" * 75)
    print("PERCEPTION STACK — RIGOROUS LATENCY & THROUGHPUT RECONCILIATION")
    print("=" * 75)

    pt_model_path = project_root / "models" / "best.pt"
    onnx_model_path = project_root / "models" / "best.onnx"
    sample_img_path = project_root / "data" / "sample_road.jpg"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_out = results_dir / "reconciled_benchmark.csv"

    img_bgr = cv2.imread(str(sample_img_path))
    h_orig, w_orig = img_bgr.shape[:2]
    print(f"Hardware        : Windows 11 | CPU: {platform.processor() or 'x86_64'}")
    print(f"Software        : Python {sys.version.split()[0]} | Torch {torch.__version__} | ORT {ort.__version__}")
    print(f"Sample Input    : {sample_img_path.name} ({w_orig}x{h_orig})")
    print("-" * 75)

    yolo = YOLO(str(pt_model_path))
    model_inner = yolo.model.to("cpu")
    model_inner.eval()

    dummy_tensor = torch.zeros((1, 3, 640, 640), dtype=torch.float32)

    # 1. PURE PYTORCH MODEL FORWARD PASS (No preprocessing / NMS)
    print("[1/5] Measuring Pure PyTorch Forward Pass (1x3x640x640 tensor)...")
    for _ in range(10):
        with torch.no_grad():
            _ = model_inner(dummy_tensor)

    pt_pure_latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_inner(dummy_tensor)
        pt_pure_latencies.append((time.perf_counter() - t0) * 1000.0)

    # 2. FULL PYTORCH PREDICT (Letterboxing + Forward + NMS)
    print("[2/5] Measuring Full PyTorch predict() on 2560x720 image (with NMS)...")
    detector = RoadObjectDetector(model_path=pt_model_path, imgsz=640, device="cpu")
    for _ in range(10):
        _ = detector.detect(img_bgr)

    pt_full_latencies = []
    for _ in range(30):
        res = detector.detect(img_bgr)
        pt_full_latencies.append(res.inference_time_ms)

    # 3. PURE ONNX RUNTIME FORWARD PASS (1x3x640x640 blob)
    print("[3/5] Measuring ONNX Runtime CPU session.run() (1x3x640x640 blob)...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = psutil.cpu_count(logical=False) or 4
    sess = ort.InferenceSession(str(onnx_model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    dummy_blob = np.zeros((1, 3, 640, 640), dtype=np.float32)

    for _ in range(10):
        _ = sess.run([out_name], {inp_name: dummy_blob})

    ort_latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = sess.run([out_name], {inp_name: dummy_blob})
        ort_latencies.append((time.perf_counter() - t0) * 1000.0)

    # 4. END-TO-END PIPELINE: DETECTION + TRACKING + LANES
    print("[4/5] Measuring End-to-End Pipeline (YOLO + ByteTrack + OpenCV Lanes)...")
    pipe_det_trk_lane = PerceptionPipeline(PipelineConfig(
        model_path=str(pt_model_path),
        enable_tracking=True,
        enable_lanes=True,
        enable_depth=False,
    ))
    for _ in range(5):
        _ = pipe_det_trk_lane.process_frame(img_bgr)

    e2e_latencies = []
    for i in range(25):
        res = pipe_det_trk_lane.process_frame(img_bgr, frame_index=i+1)
        e2e_latencies.append(res.total_latency_ms)

    # 5. END-TO-END PIPELINE: DETECTION + TRACKING + LANES + DEPTH
    print("[5/5] Measuring End-to-End Pipeline WITH MiDaS Depth...")
    pipe_full = PerceptionPipeline(PipelineConfig(
        model_path=str(pt_model_path),
        enable_tracking=True,
        enable_lanes=True,
        enable_depth=True,
        depth_frequency=1,  # Every frame to measure peak latency
    ))
    for _ in range(3):
        _ = pipe_full.process_frame(img_bgr)

    depth_e2e_latencies = []
    for i in range(15):
        res = pipe_full.process_frame(img_bgr, frame_index=i+1)
        depth_e2e_latencies.append(res.total_latency_ms)

    # Format Benchmark Matrix
    benchmarks = [
        ("Pure PyTorch Forward (Tensor 640x640)", pt_pure_latencies, 5.98),
        ("PyTorch Detector (Image + NMS)", pt_full_latencies, 5.98),
        ("ONNX Runtime CPU Forward (Tensor 640x640)", ort_latencies, 11.70),
        ("Pipeline (YOLO + ByteTrack + Lanes)", e2e_latencies, 5.98),
        ("Pipeline (YOLO + ByteTrack + Lanes + Depth)", depth_e2e_latencies, 5.98 + 81.8),
    ]

    print("\n" + "=" * 85)
    print(f"{'Benchmark Scope':<42} | {'Mean (ms)':<9} | {'Median':<7} | {'FPS':<6} | {'Std (ms)':<8}")
    print("-" * 85)

    csv_rows = []
    for name, lats, size_mb in benchmarks:
        mean_l = float(np.mean(lats))
        med_l = float(np.median(lats))
        std_l = float(np.std(lats))
        fps_val = 1000.0 / mean_l if mean_l > 0 else 0.0
        print(f"{name:<42} | {mean_l:<9.2f} | {med_l:<7.2f} | {fps_val:<6.1f} | {std_l:<8.2f}")
        csv_rows.append([name, round(mean_l, 2), round(med_l, 2), round(fps_val, 2), round(std_l, 2), size_mb])

    # Save to CSV
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "mean_latency_ms", "median_latency_ms", "throughput_fps", "std_latency_ms", "model_size_mb"])
        w.writerows(csv_rows)

    print("-" * 85)
    print(f"Reconciled benchmark data saved to: {csv_out}")
    print("=" * 85)

if __name__ == "__main__":
    main()
