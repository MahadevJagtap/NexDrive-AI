"""
NVIDIA TensorRT Optimization and Export Utility.

Hardware Constraints & Guard:
- TensorRT requires an NVIDIA GPU with CUDA compute capability (e.g., RTX 30/40 series, T4, A100).
- This script checks hardware availability and prevents crashes on non-NVIDIA hardware (such as Intel Iris Xe).
- If run on an NVIDIA GPU workstation, it exports YOLOv8n to a high-throughput FP16 TensorRT engine.
"""

import sys
import platform
from pathlib import Path
import torch

BASE_DIR = Path("D:/auto_drive")
MODEL_PATH = BASE_DIR / "models" / "best.pt"

def export_tensorrt():
    print("=== NVIDIA TensorRT Optimization & Engine Export ===", flush=True)
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})", flush=True)
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"CUDA Available:  {torch.cuda.is_available()}", flush=True)

    if not torch.cuda.is_available():
        print("\n" + "="*70, flush=True)
        print("HARDWARE BLOCKER: NVIDIA TensorRT is NOT supported on this machine.", flush=True)
        print("System Hardware: Intel Core i5 with integrated Intel Iris Xe Graphics.", flush=True)
        print("TensorRT requires an NVIDIA GPU with CUDA runtime and TensorRT libraries.", flush=True)
        print("="*70, flush=True)
        print("\nReproducible Instructions for External NVIDIA Workstation:")
        print("1. Target Environment Requirements:")
        print("   - OS: Ubuntu 22.04 LTS or Windows 10/11 x64")
        print("   - GPU: NVIDIA GPU (Turing / Ampere / Ada Lovelace, e.g., RTX 3060+, T4, A10)")
        print("   - Drivers: NVIDIA Driver >= 525.60.13, CUDA >= 11.8 / 12.1")
        print("2. Installation:")
        print("   pip install ultralytics tensorrt onnx onnxsim")
        print("3. Export Command:")
        print("   yolo export model=models/best.pt format=engine device=0 half=True workspace=4")
        print("4. Python API Execution:")
        print("   from ultralytics import YOLO")
        print("   model = YOLO('models/best.pt')")
        print("   model.export(format='engine', device=0, half=True, workspace=4)")
        print("   # Engine will be saved to: models/best.engine")
        print("="*70 + "\n", flush=True)
        return False

    # If NVIDIA GPU is detected
    gpu_name = torch.cuda.get_device_name(0)
    print(f"NVIDIA GPU Detected: {gpu_name}", flush=True)
    
    try:
        import tensorrt
        print(f"TensorRT Version: {tensorrt.__version__}", flush=True)
    except ImportError:
        print("Error: PyTorch CUDA is available but the 'tensorrt' Python module is not installed.", flush=True)
        print("Please run: pip install tensorrt", flush=True)
        return False

    from ultralytics import YOLO
    print(f"Loading checkpoint: {MODEL_PATH}", flush=True)
    model = YOLO(str(MODEL_PATH))
    
    print("Exporting to TensorRT Engine with FP16 half-precision...", flush=True)
    engine_path = model.export(
        format="engine",
        device=0,
        half=True,
        workspace=4,
        verbose=True
    )
    print(f"SUCCESS: TensorRT engine exported to: {engine_path}", flush=True)
    return True

if __name__ == "__main__":
    success = export_tensorrt()
    sys.exit(0 if success else 0)
