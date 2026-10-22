r"""
Download and verify the pretrained BDD100K YOLOv8n detector from Hugging Face.
Repository: dronefreak/bdd100k-yolov8n
Checkpoint: best.pt
Target path: models/best.pt
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Step 1: Securely load environment variables
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

hf_home = os.getenv("HF_HOME", str(PROJECT_ROOT / "models" / "cache"))
os.environ["HF_HOME"] = hf_home
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.makedirs(hf_home, exist_ok=True)

models_dir = Path(os.getenv("MODELS_DIR", str(PROJECT_ROOT / "models")))
models_dir.mkdir(parents=True, exist_ok=True)
target_checkpoint = models_dir / "best.pt"

# Check HF_TOKEN (standard) or legacy HF_ACCESS_TOKEN
token = (os.getenv("HF_TOKEN", "") or os.getenv("HF_ACCESS_TOKEN", "")).strip() or None

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

print("=== Starting Model Verification & Download ===")
print(f"Target checkpoint path: {target_checkpoint}")
print(f"HF_HOME cache configured at: {hf_home}")
print(f"Hugging Face token detected: {'Yes (Length: ' + str(len(token)) + ')' if token else 'No (using public access)'}")

import huggingface_hub
from huggingface_hub import hf_hub_download, HfApi

REPO_ID = "dronefreak/bdd100k-yolov8n"
FILENAME = "best.pt"

# Download model card/README if available
try:
    print(f"\nChecking model repository: {REPO_ID}")
    api = HfApi()
    model_info = api.model_info(repo_id=REPO_ID, token=token)
    print(f"Repository confirmed accessible: {model_info.id}")
    tags = getattr(model_info, 'tags', [])
    print(f"Tags/License information: {[t for t in tags if 'license' in t or 'yolo' in t]}")
    
    # Download README.md / model card to docs
    readme_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="README.md",
        token=token,
        cache_dir=hf_home
    )
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()
    with open(docs_dir / "MODEL_CARD.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Saved model card documentation to: {docs_dir / 'MODEL_CARD.md'}")
except Exception as e:
    print(f"Warning fetching model card: {e}")

# Download best.pt if needed
if target_checkpoint.exists() and target_checkpoint.stat().st_size > 0:
    print(f"\nCheckpoint already exists: {target_checkpoint} ({target_checkpoint.stat().st_size / (1024*1024):.2f} MB)")
else:
    print(f"\nDownloading {FILENAME} from {REPO_ID} ...")
    downloaded_file = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        token=token,
        cache_dir=hf_home,
        local_dir=models_dir,
    )
    print(f"Download completed successfully: {downloaded_file}")

# Verify with Ultralytics
print("\nVerifying model loading with Ultralytics YOLO...")
from ultralytics import YOLO
import torch

try:
    model = YOLO(str(target_checkpoint))
    print("SUCCESS: Model loaded successfully with Ultralytics!")
    print(f"Model architecture type: {type(model.model).__name__}")
    print(f"Torch execution device: CPU (torch.cuda.is_available() = {torch.cuda.is_available()})")
    
    # Print class names safely
    names = model.names
    print(f"\nDetected {len(names)} classes:")
    for cls_id, cls_name in names.items():
        print(f"  Class {cls_id}: {cls_name}")
        
    print("\nModel Verification PASSED.")
except Exception as e:
    print(f"ERROR: Failed to load model with Ultralytics: {e}")
    sys.exit(1)
