"""
Execute a short CPU fine-tuning run of YOLOv8n on the mini BDD100K subset.
Saves the fine-tuned checkpoint separately as 'models/finetuned_best.pt'
without overwriting or modifying the baseline checkpoint 'models/best.pt'.
"""

import os
import sys
import shutil
import time
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path("D:/auto_drive")
BASELINE_CHECKPOINT = BASE_DIR / "models" / "best.pt"
DATA_YAML = BASE_DIR / "data" / "bdd100k_mini" / "data.yaml"
OUTPUT_DIR = BASE_DIR / "models" / "finetune_runs"
FINETUNED_CHECKPOINT = BASE_DIR / "models" / "finetuned_best.pt"

def main():
    print("=== Starting BDD100K Mini-Subset CPU Fine-Tuning Run ===", flush=True)
    
    if not BASELINE_CHECKPOINT.exists():
        print(f"Error: Baseline checkpoint not found at {BASELINE_CHECKPOINT}", flush=True)
        sys.exit(1)
        
    if not DATA_YAML.exists():
        print(f"Error: Dataset configuration not found at {DATA_YAML}", flush=True)
        sys.exit(1)

    print(f"Baseline Checkpoint: {BASELINE_CHECKPOINT}", flush=True)
    print(f"Dataset Config:      {DATA_YAML}", flush=True)
    print(f"Device:              CPU", flush=True)
    print(f"Epochs:              3", flush=True)
    print(f"Batch Size:          4", flush=True)
    print(f"Image Size:          640", flush=True)
    print(f"Run Output Dir:      {OUTPUT_DIR}", flush=True)

    # Load baseline model
    model = YOLO(str(BASELINE_CHECKPOINT))
    
    t_start = time.time()
    
    # Train for 3 epochs on CPU
    results = model.train(
        data=str(DATA_YAML),
        epochs=3,
        batch=4,
        imgsz=640,
        device="cpu",
        project=str(OUTPUT_DIR),
        name="bdd100k_subset",
        exist_ok=True,
        workers=2,
        optimizer="SGD",
        lr0=0.001,  # Gentle fine-tuning learning rate
        lrf=0.01,
        verbose=True,
        plots=True,
        save=True
    )
    
    train_duration = time.time() - t_start
    print(f"\nFine-tuning completed in {train_duration:.2f} seconds ({train_duration/60:.2f} minutes).", flush=True)
    
    # Check best.pt in run output
    run_best_pt = OUTPUT_DIR / "bdd100k_subset" / "weights" / "best.pt"
    run_last_pt = OUTPUT_DIR / "bdd100k_subset" / "weights" / "last.pt"
    
    source_pt = run_best_pt if run_best_pt.exists() else run_last_pt
    if not source_pt.exists():
        print("Error: Could not locate fine-tuned checkpoint weights.", flush=True)
        sys.exit(1)
        
    # Copy to separate checkpoint path
    shutil.copy(source_pt, FINETUNED_CHECKPOINT)
    print(f"SUCCESS: Saved fine-tuned checkpoint to: {FINETUNED_CHECKPOINT}", flush=True)
    print(f"Baseline checkpoint preserved at:        {BASELINE_CHECKPOINT}", flush=True)
    print(f"Checkpoint size: {FINETUNED_CHECKPOINT.stat().st_size / (1024*1024):.2f} MB", flush=True)

if __name__ == "__main__":
    main()
