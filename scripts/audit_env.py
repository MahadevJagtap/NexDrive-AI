"""
Environment and dependency audit script for Autonomous Driving Perception Stack.
"""
import sys
import importlib

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

packages = [
    'torch', 'torchvision', 'ultralytics', 'cv2', 'numpy', 
    'PIL', 'huggingface_hub', 'dotenv', 'onnx', 'onnxruntime', 
    'streamlit', 'matplotlib', 'pandas', 'psutil', 'pytest'
]

missing = []
for pkg in packages:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, '__version__', 'available')
        print(f"  {pkg:18s}: OK (v{ver})")
    except Exception as e:
        print(f"  {pkg:18s}: MISSING ({e})")
        missing.append(pkg)

print("\nMissing packages:", missing)
