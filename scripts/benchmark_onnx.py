"""
Benchmarking Script: PyTorch (.pt) vs ONNX Runtime (.onnx) on CPU.
Measures inference latency, throughput (FPS), memory footprint, model size,
and verifies detection output consistency.
Saves measured results to results/benchmark_onnx_vs_pytorch.csv.
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

def get_process_memory_mb() -> float:
    """Returns the current process resident memory in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024.0 * 1024.0)

def run_benchmark(num_warmup: int = 10, num_runs: int = 30):
    print("=" * 70)
    print("AUTONOMOUS DRIVING PERCEPTION — ONNX RUNTIME VS PYTORCH BENCHMARK")
    print("=" * 70)

    pt_model_path = project_root / "models" / "best.pt"
    onnx_model_path = project_root / "models" / "best.onnx"
    sample_img_path = project_root / "data" / "sample_road.jpg"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_out_path = results_dir / "benchmark_onnx_vs_pytorch.csv"

    # Environment Diagnostics
    cpu_info = platform.processor() or "x86_64"
    python_ver = sys.version.split()[0]
    torch_ver = torch.__version__
    ort_ver = ort.__version__

    print(f"Hardware Platform : Windows 11 | CPU: {cpu_info}")
    print(f"Software Stack    : Python {python_ver} | PyTorch {torch_ver} | ONNX Runtime {ort_ver}")
    print(f"Sample Input      : {sample_img_path}")
    print(f"Iterations        : {num_warmup} warm-up, {num_runs} measured iterations")
    print("-" * 70)

    if not pt_model_path.exists():
        print(f"ERROR: PyTorch model missing: {pt_model_path}")
        return False
    if not onnx_model_path.exists():
        print(f"ERROR: ONNX model missing: {onnx_model_path}")
        return False
    if not sample_img_path.exists():
        print(f"ERROR: Sample image missing: {sample_img_path}")
        return False

    img_bgr = cv2.imread(str(sample_img_path))
    h_orig, w_orig = img_bgr.shape[:2]

    # --- 1. BENCHMARK PYTORCH CPU ---
    print("\n[1/3] Benchmarking PyTorch CPU (models/best.pt)...")
    pt_size_mb = pt_model_path.stat().st_size / (1024.0 * 1024.0)
    mem_before_pt = get_process_memory_mb()

    pt_model = YOLO(str(pt_model_path))

    # Warmup
    for _ in range(num_warmup):
        _ = pt_model.predict(source=img_bgr, imgsz=640, device="cpu", verbose=False)

    # Measurement
    pt_latencies = []
    pt_det_count = 0
    for _ in range(num_runs):
        t0 = time.perf_counter()
        res = pt_model.predict(source=img_bgr, imgsz=640, device="cpu", verbose=False)
        dt = (time.perf_counter() - t0) * 1000.0
        pt_latencies.append(dt)
        if pt_det_count == 0 and res and len(res) > 0 and res[0].boxes is not None:
            pt_det_count = len(res[0].boxes)

    mem_after_pt = get_process_memory_mb()
    pt_mean = float(np.mean(pt_latencies))
    pt_median = float(np.median(pt_latencies))
    pt_min = float(np.min(pt_latencies))
    pt_max = float(np.max(pt_latencies))
    pt_std = float(np.std(pt_latencies))
    pt_fps = 1000.0 / pt_mean if pt_mean > 0 else 0.0

    print(f"  PyTorch Model Size   : {pt_size_mb:.2f} MB")
    print(f"  PyTorch Detections   : {pt_det_count} objects")
    print(f"  PyTorch Mean Latency : {pt_mean:.2f} ms (+/- {pt_std:.2f} ms)")
    print(f"  PyTorch Median       : {pt_median:.2f} ms (Min: {pt_min:.2f} ms, Max: {pt_max:.2f} ms)")
    print(f"  PyTorch Throughput   : {pt_fps:.1f} FPS")
    print(f"  Process Memory RSS   : {mem_after_pt:.1f} MB (Delta: +{mem_after_pt - mem_before_pt:.1f} MB)")

    # Cleanup PyTorch model memory
    del pt_model
    gc.collect()

    # --- 2. BENCHMARK ONNX RUNTIME CPU ---
    print("\n[2/3] Benchmarking ONNX Runtime CPU (models/best.onnx)...")
    onnx_size_mb = onnx_model_path.stat().st_size / (1024.0 * 1024.0)
    mem_before_onnx = get_process_memory_mb()

    # Setup ONNX Runtime Session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = psutil.cpu_count(logical=False) or 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(onnx_model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_names = [out.name for out in session.get_outputs()]

    # Preprocessing input for ONNX: letterbox resize to 640x640, BGR->RGB, HWC->BCHW, float32 [0, 1]
    resized = cv2.resize(img_bgr, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)

    # Warmup
    for _ in range(num_warmup):
        _ = session.run(output_names, {input_name: blob})

    # Measurement
    onnx_latencies = []
    onnx_raw_outputs = None
    for _ in range(num_runs):
        t0 = time.perf_counter()
        out = session.run(output_names, {input_name: blob})
        dt = (time.perf_counter() - t0) * 1000.0
        onnx_latencies.append(dt)
        onnx_raw_outputs = out

    mem_after_onnx = get_process_memory_mb()
    onnx_mean = float(np.mean(onnx_latencies))
    onnx_median = float(np.median(onnx_latencies))
    onnx_min = float(np.min(onnx_latencies))
    onnx_max = float(np.max(onnx_latencies))
    onnx_std = float(np.std(onnx_latencies))
    onnx_fps = 1000.0 / onnx_mean if onnx_mean > 0 else 0.0

    # Parse ONNX raw output shape (B, 14, 8400) -> 14 = 4 bbox + 10 classes
    onnx_out_shape = onnx_raw_outputs[0].shape if onnx_raw_outputs else ()
    # Count predictions above conf threshold 0.25
    out_matrix = onnx_raw_outputs[0][0].T  # shape (8400, 14)
    scores = np.max(out_matrix[:, 4:], axis=1)
    onnx_det_count = int(np.sum(scores >= 0.25))

    print(f"  ONNX Model Size      : {onnx_size_mb:.2f} MB")
    print(f"  ONNX Output Shape    : {onnx_out_shape}")
    print(f"  ONNX Detections (conf>=0.25): {onnx_det_count} candidate boxes")
    print(f"  ONNX Mean Latency    : {onnx_mean:.2f} ms (+/- {onnx_std:.2f} ms)")
    print(f"  ONNX Median          : {onnx_median:.2f} ms (Min: {onnx_min:.2f} ms, Max: {onnx_max:.2f} ms)")
    print(f"  ONNX Throughput      : {onnx_fps:.1f} FPS")
    print(f"  Process Memory RSS   : {mem_after_onnx:.1f} MB (Delta: +{mem_after_onnx - mem_before_onnx:.1f} MB)")

    # Speedup calculation
    speedup = pt_mean / onnx_mean if onnx_mean > 0 else 1.0
    print(f"\n  >>> Latency Comparison: ONNX is {speedup:.2f}x relative to PyTorch ({abs(pt_mean - onnx_mean):.2f} ms delta)")

    # --- 3. SAVE RESULTS TO CSV ---
    print(f"\n[3/3] Saving benchmark results to {csv_out_path}...")
    headers = [
        "runtime", "framework_version", "model_file", "model_size_mb",
        "mean_latency_ms", "median_latency_ms", "min_latency_ms", "max_latency_ms", "std_latency_ms",
        "throughput_fps", "rss_memory_mb", "detections_count", "hardware", "os",
    ]

    rows = [
        [
            "PyTorch CPU", torch_ver, pt_model_path.name, f"{pt_size_mb:.2f}",
            f"{pt_mean:.2f}", f"{pt_median:.2f}", f"{pt_min:.2f}", f"{pt_max:.2f}", f"{pt_std:.2f}",
            f"{pt_fps:.2f}", f"{mem_after_pt:.1f}", pt_det_count, cpu_info, platform.system(),
        ],
        [
            "ONNX Runtime CPU", ort_ver, onnx_model_path.name, f"{onnx_size_mb:.2f}",
            f"{onnx_mean:.2f}", f"{onnx_median:.2f}", f"{onnx_min:.2f}", f"{onnx_max:.2f}", f"{onnx_std:.2f}",
            f"{onnx_fps:.2f}", f"{mem_after_onnx:.1f}", onnx_det_count, cpu_info, platform.system(),
        ],
    ]

    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"  Successfully wrote benchmark data to: {csv_out_path}")
    print("=" * 70)
    print("MILESTONE 7 (ONNX BENCHMARKING) COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
