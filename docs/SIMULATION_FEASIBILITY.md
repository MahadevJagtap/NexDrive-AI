# CARLA, ROS2, and TensorRT Feasibility Report

## 1. Executive Summary & Hardware Context

This report evaluates the technical feasibility of running the **CARLA Autonomous Driving Simulator**, **ROS2 middleware**, and **NVIDIA TensorRT acceleration** on the target evaluation machine.

### Target Machine Specifications:
* **Operating System**: Windows 11 Home/Pro (64-bit)
* **Processor**: 11th Gen Intel Core i5-1145G7 @ 2.60 GHz
* **System Memory**: 16 GB DDR4 RAM
* **Graphics Processor**: Intel Iris Xe Integrated Graphics (Shared VRAM)
* **Dedicated GPU**: None (No NVIDIA CUDA hardware)

---

## 2. Technical Feasibility Analysis

| Technology | Requirement | Local Environment Status | Feasibility Assessment | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA TensorRT** | NVIDIA GPU (Compute Capability $\ge$ 6.0), CUDA Toolkit, cuDNN, TensorRT engine builder | Intel Iris Xe integrated graphics | ❌ **Hardware Blocked**: Cannot execute on non-NVIDIA hardware. | Exported model to **ONNX** (`models/best.onnx`) and benchmarked using **ONNX Runtime CPU**. |
| **CARLA Simulator** | Unreal Engine 4.26+, $\ge$6 GB dedicated VRAM, 8-core CPU, 50 GB disk space | Intel Iris Xe shared graphics memory | ❌ **Hardware Blocked**: Unreal Engine crashes or achieves $<3$ FPS without dedicated GPU. | Built `src/simulation/camera_adapter.py` with `CARLACameraAdapter` and `VirtualCameraStream`. |
| **ROS2 (Robot Operating System)** | Ubuntu 22.04 LTS (Jammy) for ROS2 Humble, DDS discovery middleware | Windows 11 host environment | ⚠️ **Environment Blocked**: Native Windows ROS2 lacks seamless DDS socket integration. | Provided standard ROS2 topic specifications, message schemas, and Python sensor adapter. |

---

## 3. Implemented Software Adapter (`src/simulation/`)

Rather than halting development due to hardware limitations, an architectural adapter was implemented in [`src/simulation/camera_adapter.py`](file:///d:/auto_drive/src/simulation/camera_adapter.py):

1. **`CARLACameraAdapter`**:
   - Parses the raw 4-channel BGRA byte buffer output from CARLA's `sensor.camera.rgb` sensor (`carla.Image.raw_data`).
   - Converts BGRA directly to standard OpenCV BGR `np.ndarray` without memory thrashing.
   - Preserves simulation timestamps and camera field-of-view parameters.

2. **`VirtualCameraStream`**:
   - Replays recorded driving sequences adhering strictly to the `SimulationFrame` data contract.
   - Emulates sensor arrival times and frame intervals for deterministic offline benchmarking.

3. **ROS2 Integration Blueprint**:
   The perception stack communicates via standard ROS2 message definitions:

```
[CARLA / Sensors] 
       │
       ▼ (Topic: /carla/ego_vehicle/rgb_front/image_color) [sensor_msgs/msg/Image]
┌─────────────────────────────────────────────────────────────┐
│       Autonomous Driving Perception Stack (ROS2 Node)       │
│                                                             │
│  ┌─────────────────┐   ┌────────────────┐   ┌────────────┐  │
│  │ YOLOv8 Detector │──▶│ ByteTrack (MOT)│──▶│ OpenCV Lane│  │
│  └─────────────────┘   └────────────────┘   └────────────┘  │
└─────────────────────────────────────────────────────────────┘
       │
       ├─▶ /perception/detections_2d    [vision_msgs/msg/Detection2DArray]
       ├─▶ /perception/tracks           [vision_msgs/msg/Detection2DArray]
       ├─▶ /perception/lane_boundaries  [visualization_msgs/msg/MarkerArray]
       └─▶ /perception/annotated_feed   [sensor_msgs/msg/Image]
```

---

## 4. Instructions for Deployment on Compatible NVIDIA/Linux Hardware

When deploying this repository to an NVIDIA-equipped workstation running Ubuntu 22.04 LTS with CARLA 0.9.15:

### Step 1: Install CUDA and TensorRT
```bash
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-2
pip install tensorrt tensorrt-cu12
```

### Step 2: Build TensorRT Engine from ONNX
```bash
# Using trtexec command-line tool
trtexec --onnx=models/best.onnx --saveEngine=models/best.engine --fp16
```

### Step 3: Run CARLA Simulation Bridge
```python
import carla
from src.simulation.camera_adapter import CARLACameraAdapter
from src.pipeline.perception_pipeline import PerceptionPipeline

client = carla.Client('localhost', 2000)
world = client.get_world()
# Attach camera sensor to ego vehicle and route callbacks to pipeline
```
