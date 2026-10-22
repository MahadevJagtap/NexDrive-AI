# Individual Reflection Draft: Autonomous Driving Perception Stack

> **Note for the User**: This draft documents the verified technical achievements, engineering trade-offs, and design methodologies implemented across the project. Sections requiring your personal reflections or subjective impressions are clearly highlighted in brackets `[User to confirm/customize: ...]`.

---

## 1. Technical Accomplishments & Verified Work

Throughout this project, I developed an end-to-end **Autonomous Driving Perception Stack** targeting CPU execution on an Intel Iris Xe Windows 11 machine without relying on external cloud compute or dedicated GPU infrastructure:

1. **Pretrained Road Object Detection**:
   - Integrated the `dronefreak/bdd100k-yolov8n` checkpoint fine-tuned on the Berkeley DeepDrive (BDD100K) dataset.
   - Built a robust detector wrapper in `src/detection/` achieving **43.99 ms latency (~22.7 FPS)** on CPU test frames across 10 vehicle, pedestrian, and signage classes.

2. **Persistent Multi-Object Tracking**:
   - Integrated ByteTrack into `src/tracking/` to assign persistent identity tracks across video frames.
   - Verified on a real driving video (`data/demo_banner.mp4`), achieving 100% ID persistence for foreground vehicles and rendering historical motion trajectory trails.

3. **Classical Lane Boundary Analysis**:
   - Implemented an independent geometric lane detector in `src/lanes/` combining HLS/HSV color thresholding, Canny edge detection, trapezoidal ROI masking, Hough line classification, and temporal EMA smoothing.
   - Designed a translucent green drivable corridor overlay and ego-vehicle lane departure telemetry.

4. **Monocular Relative Depth Assessment**:
   - Deployed Intel ISL's `MiDaS_small` model in `src/depth/` to evaluate relative scene depth.
   - Associated relative proximity scores with detected bounding boxes, maintaining scientific rigor by explicitly labeling outputs as uncalibrated relative proximity rather than claiming metric distances.

5. **Modern Automotive Web Interface, REST API & CLI**:
   - Engineered a high-tech HTML5/CSS3/Vanilla JS perception dashboard (`frontend/`) and asynchronous FastAPI REST server (`server.py`) delivering real-time telemetry streaming, progress tracking, and dual-view video playback without heavy frontend dependencies.
   - Preserved standalone CLI runner `scripts/run_pipeline.py` and legacy Streamlit prototype `app.py`.

6. **Empirical Benchmarking & Simulation Feasibility**:
   - Exported the model to ONNX (`models/best.onnx`) and executed an empirical benchmark comparing PyTorch CPU (87.13 ms) to ONNX Runtime CPU (169.70 ms), recording results in `results/benchmark_onnx_vs_pytorch.csv`.
   - Addressed CARLA, ROS2, and TensorRT constraints by engineering a simulation adapter (`src/simulation/camera_adapter.py`) and authoring a feasibility report (`docs/SIMULATION_FEASIBILITY.md`).

7. **Automated Verification**:
   - Authored an automated unit and integration test suite in `tests/` covering configuration security, error handling, tracking persistence, lane geometry, simulation adapters, ONNX execution, and REST endpoints, achieving a **100% test pass rate (52/52 tests passed)**.

---

## 2. Key Engineering Trade-Offs & Decisions

* **Classical Lane Detection vs. Deep Neural Segmentation**:
  * *Trade-off*: Deep neural lane detectors (e.g. SCNN, Ultra-Fast-Lane-Detection) often handle complex road wear and curved markings more smoothly, but introduce heavy memory footprints and redundant GPU forward passes.
  * *Decision*: Selected classical OpenCV geometric computer vision. This eliminated deep learning dependencies, provided deterministic mathematical interpretability, and kept lane latency bounded to ~160 ms on CPU.

* **Single-Pass Inference Architecture**:
  * *Trade-off*: Running detection first and then feeding crops into a separate tracker causes duplicate forward passes and buffer copies.
  * *Decision*: ByteTrack was coupled directly with YOLOv8 forward inference inside `RoadObjectTracker`, halving processing overhead.

* **Honest Benchmarking & TensorRT Handling**:
  * *Trade-off*: TensorRT is often assumed to always be faster, but it requires compatible NVIDIA GPU hardware.
  * *Decision*: Rather than fabricating benchmarks or attempting unsupported TensorRT installs on Intel Iris Xe graphics, I conducted an empirical benchmark of PyTorch CPU vs. ONNX Runtime CPU, documenting the exact reasons why TensorRT was hardware-blocked and providing a migration plan for NVIDIA hardware.

---

## 3. Personal Reflections & Team Collaboration

`[User to customize the following section based on your personal journey and team dynamics:]`

* **Challenges Faced & Overcome**:
  `[Describe your personal experience during setup, such as debugging virtual environments, handling Windows pathing, or managing CPU memory constraints.]`

* **What I Learned**:
  `[Reflect on your technical growth in computer vision, understanding the trade-offs between deep learning and classical geometric processing, and the importance of reproducible software engineering.]`

* **Future Enhancements**:
  `[Mention personal areas of interest for future development, such as deploying the stack on an NVIDIA Jetson edge computer or integrating physical stereo cameras for calibrated metric depth.]`
