"""
Hardware, simulator, middleware, and accelerator environment probe.
Checks:
- System specs (CPU, RAM, Disk)
- GPU & CUDA availability (NVIDIA vs Intel Iris Xe)
- TensorRT installation
- CARLA simulator and Python API
- ROS2 installation and rclpy
"""

import sys
import os
import shutil
import platform
import psutil
import torch

def probe():
    print("=== HARDWARE & PLATFORM PROBE ===")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Architecture: {platform.machine()}")
    print(f"CPU: {platform.processor()}")
    print(f"Logical CPU cores: {psutil.cpu_count(logical=True)}, Physical cores: {psutil.cpu_count(logical=False)}")
    ram = psutil.virtual_memory()
    print(f"RAM Total: {ram.total / (1024**3):.2f} GB, Available: {ram.available / (1024**3):.2f} GB")
    disk = psutil.disk_usage("D:")
    print(f"Drive D: Free Space: {disk.free / (1024**3):.2f} GB")

    print("\n=== GPU & TENSORRT PROBE ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        print("No NVIDIA CUDA GPU detected. System runs on Intel Iris Xe Graphics (integrated).")

    try:
        import tensorrt
        print(f"TensorRT installed: {tensorrt.__version__}")
    except ImportError:
        print("TensorRT installed: NO (tensorrt module not found in environment)")

    print("\n=== CARLA SIMULATOR PROBE ===")
    try:
        import carla
        print(f"CARLA Python API installed: YES")
    except ImportError:
        print("CARLA Python API installed: NO (carla module not found in Python environment)")

    carla_paths = [
        r"C:\CARLA", r"D:\CARLA", r"C:\Program Files\CARLA", r"C:\CarlaUE4",
        os.path.expanduser(r"~\CARLA")
    ]
    found_carla = [p for p in carla_paths if os.path.exists(p)]
    print(f"CARLA filesystem paths found: {found_carla if found_carla else 'NONE'}")

    print("\n=== ROS2 ENVIRONMENT PROBE ===")
    ros_distro = os.getenv("ROS_DISTRO")
    print(f"ROS_DISTRO env var: {ros_distro}")
    ros2_cmd = shutil.which("ros2")
    print(f"ros2 executable in PATH: {ros2_cmd}")
    try:
        import rclpy
        print("rclpy installed: YES")
    except ImportError:
        print("rclpy installed: NO")

if __name__ == "__main__":
    probe()
