"""
Empirically compare the Baseline Checkpoint (models/best.pt) vs Fine-Tuned Checkpoint (models/finetuned_best.pt).
Generates:
1. Quantitative validation metrics (mAP50, mAP50-95, Precision, Recall).
2. CPU inference latency and throughput (ms/frame, FPS).
3. Side-by-side qualitative visual comparison saved to results/finetune_comparison.jpg.
4. Telemetry CSV saved to results/finetuning_comparison.csv.
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path("D:/auto_drive")
BASELINE_PATH = BASE_DIR / "models" / "best.pt"
FINETUNED_PATH = BASE_DIR / "models" / "finetuned_best.pt"
DATA_YAML = BASE_DIR / "data" / "bdd100k_mini" / "data.yaml"
SAMPLE_IMG_PATH = BASE_DIR / "data" / "sample_road.jpg"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = RESULTS_DIR / "finetuning_comparison.csv"
VIS_OUT = RESULTS_DIR / "finetune_comparison.jpg"

def benchmark_latency(model, img, iterations=15):
    # Warmup
    for _ in range(3):
        _ = model(img, device="cpu", verbose=False)
    
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model(img, device="cpu", verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    
    mean_latency = float(np.mean(times))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0
    return mean_latency, fps

def main():
    print("=== Empirical Comparison: Baseline vs Fine-Tuned BDD100K Checkpoints ===", flush=True)
    
    if not BASELINE_PATH.exists():
        print(f"Error: Baseline checkpoint not found at {BASELINE_PATH}", flush=True)
        sys.exit(1)
        
    if not FINETUNED_PATH.exists():
        print(f"Error: Fine-tuned checkpoint not found at {FINETUNED_PATH}", flush=True)
        sys.exit(1)

    print(f"Loading Baseline Checkpoint:   {BASELINE_PATH}", flush=True)
    base_model = YOLO(str(BASELINE_PATH))
    
    print(f"Loading Fine-Tuned Checkpoint: {FINETUNED_PATH}", flush=True)
    ft_model = YOLO(str(FINETUNED_PATH))
    
    # 1. Validation Metrics on BDD100K Mini Val Set
    print("\n--- Running Validation on Baseline Model ---", flush=True)
    base_val = base_model.val(data=str(DATA_YAML), device="cpu", verbose=False)
    base_map50 = float(base_val.box.map50)
    base_map = float(base_val.box.map)
    base_precision = float(base_val.box.mp)
    base_recall = float(base_val.box.mr)
    
    print("\n--- Running Validation on Fine-Tuned Model ---", flush=True)
    ft_val = ft_model.val(data=str(DATA_YAML), device="cpu", verbose=False)
    ft_map50 = float(ft_val.box.map50)
    ft_map = float(ft_val.box.map)
    ft_precision = float(ft_val.box.mp)
    ft_recall = float(ft_val.box.mr)

    # 2. CPU Latency Benchmark
    print("\n--- Benchmarking CPU Inference Latency ---", flush=True)
    sample_img = cv2.imread(str(SAMPLE_IMG_PATH))
    if sample_img is None:
        sample_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        
    base_lat, base_fps = benchmark_latency(base_model, sample_img)
    ft_lat, ft_fps = benchmark_latency(ft_model, sample_img)
    
    base_size_mb = BASELINE_PATH.stat().st_size / (1024 * 1024)
    ft_size_mb = FINETUNED_PATH.stat().st_size / (1024 * 1024)

    # 3. Create Comparison DataFrame
    df = pd.DataFrame([
        {
            "Model": "Baseline (dronefreak/bdd100k-yolov8n)",
            "Checkpoint": str(BASELINE_PATH),
            "Size (MB)": round(base_size_mb, 2),
            "mAP50": round(base_map50, 4),
            "mAP50-95": round(base_map, 4),
            "Precision": round(base_precision, 4),
            "Recall": round(base_recall, 4),
            "Latency (ms)": round(base_lat, 2),
            "FPS (CPU)": round(base_fps, 2),
        },
        {
            "Model": "Fine-Tuned (3-epoch mini BDD100K)",
            "Checkpoint": str(FINETUNED_PATH),
            "Size (MB)": round(ft_size_mb, 2),
            "mAP50": round(ft_map50, 4),
            "mAP50-95": round(ft_map, 4),
            "Precision": round(ft_precision, 4),
            "Recall": round(ft_recall, 4),
            "Latency (ms)": round(ft_lat, 2),
            "FPS (CPU)": round(ft_fps, 2),
        }
    ])
    
    df.to_csv(CSV_OUT, index=False)
    print(f"\nSaved quantitative comparison to: {CSV_OUT}", flush=True)
    print(df.to_string(index=False), flush=True)

    # 4. Qualitative Visual Comparison
    print("\n--- Generating Side-by-Side Visual Comparison ---", flush=True)
    res_base = base_model(sample_img, device="cpu", conf=0.25, verbose=False)[0]
    res_ft = ft_model(sample_img, device="cpu", conf=0.25, verbose=False)[0]
    
    img_base_annotated = res_base.plot()
    img_ft_annotated = res_ft.plot()
    
    # Add title headers
    h, w, _ = sample_img.shape
    header_h = 50
    header_base = np.zeros((header_h, w, 3), dtype=np.uint8) + 40
    cv2.putText(header_base, f"Baseline YOLOv8n (mAP50: {base_map50:.3f}, {base_fps:.1f} FPS)", 
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    header_ft = np.zeros((header_h, w, 3), dtype=np.uint8) + 40
    cv2.putText(header_ft, f"Fine-Tuned Checkpoint (mAP50: {ft_map50:.3f}, {ft_fps:.1f} FPS)", 
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    panel_base = np.vstack([header_base, img_base_annotated])
    panel_ft = np.vstack([header_ft, img_ft_annotated])
    
    comparison_vis = np.hstack([panel_base, panel_ft])
    cv2.imwrite(str(VIS_OUT), comparison_vis)
    print(f"Saved visual comparison artifact to: {VIS_OUT}", flush=True)
    print("\nEmpirical comparison completed successfully.", flush=True)

if __name__ == "__main__":
    main()
