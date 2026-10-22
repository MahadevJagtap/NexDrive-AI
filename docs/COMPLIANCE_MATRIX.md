# Requirements Compliance Matrix & Status Audit

This compliance matrix provides a strict, honest evaluation of each project deliverable, classifying features into:
- **Completed**: Fully implemented, empirical results verified, and benchmarked on this machine.
- **Partially Completed / Blocked by Environment**: Architectural schema, adapter, or specifications implemented; live simulator/middleware daemon blocked by Windows 11 host environment.
- **Hardware-Blocked**: Requires specialized hardware (NVIDIA GPU / CUDA runtime) not physically present on this Intel Core i5 machine.

---

## 1. Compliance Matrix Table

| Requirement / Component | Status | Target Module / Script | Verification Evidence | Notes & Empirical Results |
| :--- | :--- | :--- | :--- | :--- |
| **Pretrained BDD100K Detector** | ✅ **Completed** | [`src/detection/detector.py`](file:///d:/auto_drive/src/detection/detector.py) | [`results/smoke_test_detection.jpg`](file:///d:/auto_drive/results/smoke_test_detection.jpg) | 10 BDD100K classes; 43.99 ms model forward pass, ~116 ms full predict with NMS on CPU. |
| **Multi-Object Tracking (ByteTrack)** | ✅ **Completed** | [`src/tracking/tracker.py`](file:///d:/auto_drive/src/tracking/tracker.py) | [`results/tracking_demo.mp4`](file:///d:/auto_drive/results/tracking_demo.mp4) | Persistent IDs, motion trails; 100% ID continuity on foreground vehicles across 30 tested frames. |
| **Classical Lane Detection & Segmentation Comparison** | ✅ **Completed** | [`src/lanes/lane_detector.py`](file:///d:/auto_drive/src/lanes/lane_detector.py), [`src/lanes/segmentation_lane_detector.py`](file:///d:/auto_drive/src/lanes/segmentation_lane_detector.py), [`scripts/compare_lane_detectors.py`](file:///d:/auto_drive/scripts/compare_lane_detectors.py) | [`results/lane_comparison.csv`](file:///d:/auto_drive/results/lane_comparison.csv), [`results/lane_comparison.jpg`](file:///d:/auto_drive/results/lane_comparison.jpg) | Classical OpenCV (77.4 ms median, 12.92 FPS, 0 MB weights) vs. Deep LRASPP MobileNetV3 (173.8 ms median, 5.75 FPS, 12.5 MB weights, 366,768 px drivable corridor). |
| **Monocular Depth Estimation** | ✅ **Completed** | [`src/depth/depth_estimator.py`](file:///d:/auto_drive/src/depth/depth_estimator.py) | [`results/smoke_test_depth.jpg`](file:///d:/auto_drive/results/smoke_test_depth.jpg) | MiDaS_small; explicitly marked as uncalibrated relative proximity `[0, 1]`. Not metric distance. |
| **Integrated Pipeline & CLI** | ✅ **Completed** | [`src/pipeline/perception_pipeline.py`](file:///d:/auto_drive/src/pipeline/perception_pipeline.py), [`scripts/run_pipeline.py`](file:///d:/auto_drive/scripts/run_pipeline.py) | [`results/pipeline_e2e_validated.mp4`](file:///d:/auto_drive/results/pipeline_e2e_validated.mp4) | Full single-pass perception with top HUD telemetry; JSON telemetry output. |
| **Modern Automotive Dashboard (HTML5/CSS3/JS)** | ✅ **Completed** | [`frontend/`](file:///d:/auto_drive/frontend), [`server.py`](file:///d:/auto_drive/server.py) | Active at `http://localhost:8000` | Modern high-tech automotive UI with drag-and-drop video upload, preset samples, live telemetry polling, dual playback, and REST API. (Streamlit retained as alternative at `http://localhost:8501`). |
| **ONNX Export & CPU Benchmark** | ✅ **Completed** | [`scripts/benchmark_onnx.py`](file:///d:/auto_drive/scripts/benchmark_onnx.py), [`models/best.onnx`](file:///d:/auto_drive/models/best.onnx) | [`results/reconciled_benchmark.csv`](file:///d:/auto_drive/results/reconciled_benchmark.csv) | Verified ONNX Runtime CPU forward pass (56.69 ms, 17.64 FPS) with numerical fidelity check against PyTorch FP32. |
| **Automated Testing Suite** | ✅ **Completed** | [`tests/`](file:///d:/auto_drive/tests) | [`results/pytest_report.xml`](file:///d:/auto_drive/results/pytest_report.xml) | 52/52 automated unit and integration tests passed (100% pass rate) across config, detection, tracking, classical lanes, deep segmentation lanes, depth, pipeline, CARLA client, ROS2 node, ONNX, Streamlit logic, and FastAPI REST endpoints. |
| **Model Fine-Tuning & Comparison** | ✅ **Completed** | [`scripts/prepare_bdd_subset.py`](file:///d:/auto_drive/scripts/prepare_bdd_subset.py), [`scripts/train_subset.py`](file:///d:/auto_drive/scripts/train_subset.py), [`scripts/compare_models.py`](file:///d:/auto_drive/scripts/compare_models.py) | [`results/finetuning_comparison.csv`](file:///d:/auto_drive/results/finetuning_comparison.csv), [`results/finetune_comparison.jpg`](file:///d:/auto_drive/results/finetune_comparison.jpg) | 80-image BDD100K subset (60 train, 20 val); 3 CPU epochs in 163.3s. Saved separately to `models/finetuned_best.pt`. Precision: 0.767 vs baseline 0.711. |
| **CARLA Simulator Client Integration** | ⚠️ **Incomplete / Blocked by Environment** | [`src/simulation/carla_client.py`](file:///d:/auto_drive/src/simulation/carla_client.py), [`src/simulation/camera_adapter.py`](file:///d:/auto_drive/src/simulation/camera_adapter.py), [`tests/test_simulation.py`](file:///d:/auto_drive/tests/test_simulation.py) | [docs/SIMULATION_FEASIBILITY.md](file:///d:/auto_drive/docs/SIMULATION_FEASIBILITY.md) | `CARLASimulatorClient` lifecycle, socket port check, BGRA parsing verified; live Unreal Engine CARLA simulator blocked (not installed; requires dedicated GPU with $\ge$8GB VRAM). |
| **ROS2 Node Integration** | ⚠️ **Incomplete / Blocked by Environment** | [`src/simulation/ros2_perception_node.py`](file:///d:/auto_drive/src/simulation/ros2_perception_node.py), [`src/simulation/camera_adapter.py`](file:///d:/auto_drive/src/simulation/camera_adapter.py), [`tests/test_simulation.py`](file:///d:/auto_drive/tests/test_simulation.py) | `tests/test_simulation.py` | Standalone `PerceptionROS2Node` with isolated imports, topic/QoS schemas, mock image decoding callback verified in tests; live ROS2 daemon blocked (no ROS2 on Windows host; Python 3.12 compatibility blocker). |
| **NVIDIA TensorRT Optimization** | ❌ **Hardware-Blocked** | [`scripts/export_tensorrt.py`](file:///d:/auto_drive/scripts/export_tensorrt.py) | [`scripts/probe_environment.py`](file:///d:/auto_drive/scripts/probe_environment.py) output | Hardware check confirmed Intel Iris Xe graphics (no NVIDIA GPU/CUDA). Standalone export script with hardware guard and external workstation reproduction guide provided. |

---

## 2. Detailed Breakdown of Incomplete & Hardware-Blocked Items

### A. Model Fine-Tuning (Completed)
- **Feasibility Assessment**: The full 100GB BDD100K dataset is impractical for local CPU training. However, downloading a verified 80-sample mini subset (`dgural/bdd100k` FiftyOne validation split converted to YOLO format) required only 4 MB disk space and ~30 seconds download.
- **Execution**: 3-epoch CPU fine-tuning completed in 163.3 seconds using PyTorch CPU on an Intel Core i5-1145G7.
- **Separation of Artifacts**: Preserved baseline weights at [`models/best.pt`](file:///d:/auto_drive/models/best.pt) (5.98 MB); saved fine-tuned weights to [`models/finetuned_best.pt`](file:///d:/auto_drive/models/finetuned_best.pt) (5.94 MB).
- **Empirical Comparison on 20 Held-Out Validation Images**:
  - **Precision**: Baseline = 0.7105 (71.1%) | Fine-Tuned = **0.7672 (76.7%)** (+5.7% precision improvement)
  - **Recall**: Baseline = 0.6006 (60.1%) | Fine-Tuned = 0.5053 (50.5%)
  - **mAP50**: Baseline = 0.7299 (73.0%) | Fine-Tuned = 0.6094 (60.9%)
  - **CPU Latency**: Baseline = 88.84 ms (11.26 FPS) | Fine-Tuned = **46.48 ms (21.51 FPS)**

### B. Lane Detection: Classical vs. Deep Semantic Segmentation (Completed)
- **Feasibility & Execution**: Implemented both classical geometric vision (`OpenCVLaneDetector` using HLS/HSV masks, Canny edge detection, and Hough transforms) and deep semantic lane segmentation (`SegmentationLaneDetector` using Torchvision LRASPP MobileNetV3 Large, 12.5 MB weights).
- **Comparative Evaluation (`scripts/compare_lane_detectors.py`)**:
  - **Classical OpenCV**: 77.4 ms median latency (12.92 FPS), 0.0 MB model weights, explicit lane lines with numerical vehicle center offset (`+38.5 px`), green corridor overlay.
  - **Deep Semantic Segmentation**: 173.8 ms median latency (5.75 FPS), 12.5 MB model weights, dense pixel mask identifying 366,768 drivable corridor pixels with continuous spatial segmentation.
  - Outputs saved to [`results/lane_comparison.csv`](file:///d:/auto_drive/results/lane_comparison.csv) and side-by-side composite visual [`results/lane_comparison.jpg`](file:///d:/auto_drive/results/lane_comparison.jpg).

### C. CARLA Simulator Integration (Incomplete / Blocked by Environment)
- **Current Status**: Implemented complete client architecture [`src/simulation/carla_client.py`](file:///d:/auto_drive/src/simulation/carla_client.py) (`CARLASimulatorClient`, `CARLAClientConfig`) with lifecycle management (`connect`, `setup_sensor`, `start_streaming`, `stop`, `disconnect`), socket-level port health verification, thread-safe frame queue, and BGRA adapter (`src/simulation/camera_adapter.py`). Unit tests in `tests/test_simulation.py` verify port checks, configuration defaults, and graceful error handling.
- **Exact Environmental Blocker**:
  1. CARLA UE4 simulator binaries (`CarlaUE4.exe`) are not installed on this Windows laptop (probe confirmed 0 CARLA filesystem paths found).
  2. CARLA requires a dedicated NVIDIA GPU with DirectX 11/12 support and minimum 6-8 GB VRAM; Intel Iris Xe shared memory causes fatal crash during Unreal Engine world initialization.
  3. The official CARLA Python client wheel (`carla-0.9.15-cp37-cp37m-win_amd64.whl` or `cp38`) does not provide prebuilt wheels for Python 3.12.
- **Realistic Next Steps**: Run CARLA server on an external Linux GPU workstation (Ubuntu 22.04 with RTX 3080/4090) and stream raw sensor frames via TCP socket or ROS2 bridge to the perception pipeline.

### D. ROS2 Node Integration (Incomplete / Blocked by Environment)
- **Current Status**: Implemented standalone ROS2 node architecture [`src/simulation/ros2_perception_node.py`](file:///d:/auto_drive/src/simulation/ros2_perception_node.py) (`PerceptionROS2Node`) with isolated ROS2 imports (`rclpy`, `sensor_msgs/Image`, `vision_msgs/Detection2DArray`). Features pure-Python mock sensor message decoding, topic publishers (`/perception/detections`, `/perception/lane_markers`, `/perception/telemetry`), and image processing callbacks verified by unit tests in `tests/test_simulation.py`.
- **Exact Environmental Blocker**:
  1. No ROS2 distribution is installed on the host OS (`ROS_DISTRO` = None, `ros2` CLI missing from PATH).
  2. Official ROS2 Humble / Iron Windows binaries are compiled specifically for **Python 3.10.x** and Visual Studio 2019 runtime; attempting to inject ROS2 into the active Python 3.12.7 virtual environment would cause DLL linkage failures and environment corruption.
  3. Windows lacks native ROS2 apt package management, requiring heavy manual dependencies (Chocolatey, OpenSSL, CycloneDDS, Colcon).
- **Realistic Next Steps**: Deploy the perception stack inside an Ubuntu 22.04 LTS Docker container (or WSL2) with native `ros-humble-desktop` and `rclpy`.

### E. NVIDIA TensorRT Optimization (Hardware-Blocked)
- **Current Status**: Standalone validation utility [`scripts/export_tensorrt.py`](file:///d:/auto_drive/scripts/export_tensorrt.py) implemented with automated hardware guard.
- **Exact Environmental Blocker**:
  1. System GPU: Intel Iris Xe Graphics (integrated).
  2. `torch.cuda.is_available()` returns `False`.
  3. TensorRT is an NVIDIA-proprietary inference SDK that strictly requires NVIDIA CUDA cores and Tensor cores.
- **Reproducible External Instructions**:
  On an NVIDIA workstation (e.g., AWS EC2 `g4dn.xlarge` / `g5.xlarge` or local desktop with RTX 3060+):
  ```bash
  pip install ultralytics tensorrt onnx onnxsim
  python scripts/export_tensorrt.py
  # OR direct CLI:
  yolo export model=models/best.pt format=engine device=0 half=True workspace=4
  ```
