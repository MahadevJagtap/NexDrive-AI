# 10-Minute Technical Presentation Outline: Autonomous Driving Perception Stack

## Time Allocation: 10 Minutes Total (8 Min Presentation + 2 Min Q&A)

---

### Slide 1: Title & Problem Formulation (0:00 - 1:00)
* **Title**: Autonomous Driving Perception Stack on Edge & Resource-Constrained Hardware.
* **Problem**: Autonomous driving perception typically demands power-hungry workstation GPUs. How can we build an accurate, integrated perception stack (Detection, Tracking, Lanes, Depth) that executes reliably on standard CPU hardware without cloud dependencies?
* **Key Contribution**: A CPU-optimized perception pipeline fusing fine-tuned BDD100K YOLOv8n, ByteTrack, classical OpenCV lane geometry, and MiDaS monocular depth estimation.

---

### Slide 2: Architectural Data Flow & Design Decisions (1:00 - 2:30)
* **Visual**: System Architecture Diagram from [docs/ARCHITECTURE.md](file:///d:/auto_drive/docs/ARCHITECTURE.md).
* **Core Design Decisions**:
  - *No Duplicate Forward Passes*: Tracking (`RoadObjectTracker`) encapsulates both YOLOv8 feature extraction and ByteTrack association in one single forward pass.
  - *Decoupled Modularity*: Lane detection uses classical computer vision (HLS/HSV color thresholding + Canny + Hough), ensuring zero dependency on GPU memory or neural network weights.
  - *Frequency-Gated Depth*: Monocular depth (MiDaS_small) executes asynchronously every $N$ frames to conserve CPU cycles while still updating object proximity.

---

### Slide 3: Milestone Demonstration & Visual Results (2:30 - 4:30)
* **Visual**: Show side-by-side video clips:
  - Detection & Multi-Object Tracking (`results/tracking_demo.mp4`): 25 unique track IDs, trajectory motion trails, 100% ID persistence for foreground vehicles.
  - Lane Boundary & Drivable Corridor (`results/lane_demo.mp4`): Translucent green corridor overlay, real-time ego lane offset telemetry.
  - Relative Depth Map (`results/smoke_test_depth.jpg`): Colorized depth and proximity badges (`Rel. Prox: 0.78`).

---

### Slide 4: Empirical Benchmarking & ONNX Optimization (4:30 - 6:00)
* **Visual**: Benchmark Summary Table from [results/benchmark_onnx_vs_pytorch.csv](file:///d:/auto_drive/results/benchmark_onnx_vs_pytorch.csv).
* **Key Findings**:
  - PyTorch CPU: **87.13 ms** (11.5 FPS) | Model Size: 5.98 MB.
  - ONNX Runtime CPU: **169.70 ms** (5.9 FPS) | Model Size: 11.70 MB.
  - *Engineering Insight*: On modern Intel CPUs with oneDNN optimizations, PyTorch CPU outperforms generic ONNX Runtime `CPUExecutionProvider`. Honest, empirical reporting over assumptions.
  - *TensorRT & Hardware Analysis*: Documented why TensorRT requires NVIDIA CUDA hardware and outlined migration steps for workstation deployment.

---

### Slide 5: Simulation Feasibility & CARLA/ROS2 Bridge (6:00 - 7:15)
* **Visual**: CARLA Camera Adapter diagram and ROS2 topic architecture.
* **Adapter Pattern**:
  - `CARLACameraAdapter` converts CARLA BGRA byte arrays into standard OpenCV frames without memory copies.
  - `VirtualCameraStream` enables offline reproducible testing.
  - Full ROS2 message specification (`sensor_msgs/Image`, `vision_msgs/Detection2DArray`).

---

### Slide 6: Verification, Testing & Future Roadmap (7:15 - 8:00)
* **Test Suite**: 52 automated unit and integration tests passing (`pytest` 100% pass rate in 40.4s) across perception modules, simulation adapters, and FastAPI REST endpoints.
* **Future Work**:
  1. OpenVINO runtime acceleration for Intel Iris Xe integrated graphics.
  2. Metric camera calibration for physical distance measurement in meters.
  3. Workstation deployment with CARLA 0.9.15 and TensorRT FP16 acceleration.

---

### Q&A Session (8:00 - 10:00)
* Anticipated Questions & Pre-Prepared Answers:
  - *Why classical lane detection instead of a deep network (e.g., Ultra-Fast-Lane-Detection)?* -> Low computational footprint on CPU (~160ms), no extra weight storage, interpretable failure modes.
  - *Why is depth labeled uncalibrated?* -> Monocular models predict scale-invariant inverse depth; claiming metric meters without intrinsic/extrinsic camera calibration is physically unsound.
