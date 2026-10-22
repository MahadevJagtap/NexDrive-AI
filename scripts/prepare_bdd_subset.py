"""
Prepare a small, properly labeled BDD100K subset in YOLO format.
Source: dgural/bdd100k (FiftyOne BDD100K validation dataset on Hugging Face).
Only downloads the exact required slice of images (~80 images: 60 train, 20 val),
strictly adhering to CPU-feasibility without downloading the entire 100GB dataset.
Uses multi-threaded downloading for fast throughput.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download

BASE_DIR = Path("D:/auto_drive")
DATA_DIR = BASE_DIR / "data" / "bdd100k_mini"
TRAIN_IMG_DIR = DATA_DIR / "images" / "train"
VAL_IMG_DIR = DATA_DIR / "images" / "val"
TRAIN_LBL_DIR = DATA_DIR / "labels" / "train"
VAL_LBL_DIR = DATA_DIR / "labels" / "val"

for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 10 BDD100K object classes matching baseline models/best.pt
CLASS_MAP = {
    "person": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motor": 6,
    "bike": 7,
    "traffic light": 8,
    "traffic sign": 9,
}

IMG_BASE_URL = "https://huggingface.co/datasets/dgural/bdd100k/resolve/main"

def download_single(item):
    rel_path, target_img_path = item
    if target_img_path.exists() and target_img_path.stat().st_size > 0:
        return True
    url = f"{IMG_BASE_URL}/{rel_path}"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            with open(target_img_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception as e:
        print(f"Error downloading {rel_path}: {e}", flush=True)
    return False

def convert_to_yolo(detections):
    lines = []
    for d in detections:
        label = d.get("label", "").lower()
        if label not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[label]
        bbox = d.get("bounding_box", [])
        if len(bbox) != 4:
            continue
        x_min, y_min, w, h = bbox
        
        # FiftyOne format: [x_min, y_min, w, h] normalized [0, 1]
        x_center = min(max(x_min + w / 2.0, 0.0), 1.0)
        y_center = min(max(y_min + h / 2.0, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        
        if w > 0.002 and h > 0.002:
            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return lines

def main():
    print("=== Preparing BDD100K YOLO Mini-Subset ===", flush=True)
    
    samples_path = hf_hub_download(repo_id="dgural/bdd100k", filename="samples.json", repo_type="dataset")
    print(f"Reading cached samples from {samples_path}...", flush=True)
    
    with open(samples_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    all_samples = data.get("samples", [])
    print(f"Total samples in dataset index: {len(all_samples)}", flush=True)
    
    # Filter for samples that have at least 2 relevant object detections
    valid_samples = []
    for s in all_samples:
        dets = s.get("detections", {}).get("detections", [])
        relevant_dets = [d for d in dets if d.get("label", "").lower() in CLASS_MAP]
        if len(relevant_dets) >= 2:
            valid_samples.append((s, relevant_dets))
            if len(valid_samples) >= 80:
                break
                
    print(f"Selected {len(valid_samples)} candidate samples with verified annotations.", flush=True)
    
    num_train = 60
    train_slice = valid_samples[:num_train]
    val_slice = valid_samples[num_train:]
    
    download_items = []
    # Train
    for s, dets in train_slice:
        fpath = s["filepath"]
        fname = Path(fpath).name
        out_img = TRAIN_IMG_DIR / fname
        download_items.append((fpath, out_img))
        
        # Write labels
        out_txt = TRAIN_LBL_DIR / (Path(fname).stem + ".txt")
        lines = convert_to_yolo(dets)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            
    # Val
    for s, dets in val_slice:
        fpath = s["filepath"]
        fname = Path(fpath).name
        out_img = VAL_IMG_DIR / fname
        download_items.append((fpath, out_img))
        
        # Write labels
        out_txt = VAL_LBL_DIR / (Path(fname).stem + ".txt")
        lines = convert_to_yolo(dets)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    print(f"\nDownloading {len(download_items)} images using 8 parallel worker threads...", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_single, item) for item in download_items]
        completed = 0
        for f in as_completed(futures):
            completed += 1
            if completed % 20 == 0 or completed == len(download_items):
                print(f"  Downloaded: {completed}/{len(download_items)} ({time.time()-t0:.1f}s)", flush=True)

    # Write data.yaml with absolute posix path
    yaml_path = DATA_DIR / "data.yaml"
    yaml_content = f"""# BDD100K Mini Dataset Configuration for YOLOv8 CPU Fine-Tuning
path: {DATA_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: person
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motor
  7: bike
  8: traffic light
  9: traffic sign
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    train_imgs = list(TRAIN_IMG_DIR.glob("*.jpg"))
    val_imgs = list(VAL_IMG_DIR.glob("*.jpg"))
    train_lbls = list(TRAIN_LBL_DIR.glob("*.txt"))
    val_lbls = list(VAL_LBL_DIR.glob("*.txt"))

    print("\n=== Dataset Summary ===", flush=True)
    print(f"Configuration file: {yaml_path}", flush=True)
    print(f"Train samples: {len(train_imgs)} images, {len(train_lbls)} labels", flush=True)
    print(f"Val samples:   {len(val_imgs)} images, {len(val_lbls)} labels", flush=True)
    print("SUCCESS: Small properly labeled BDD100K subset is prepared and ready for YOLO training.", flush=True)

if __name__ == "__main__":
    main()
