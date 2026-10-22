# Autonomous Driving Perception Stack — API Documentation

This document describes the public Python classes, methods, data structures, and signatures implemented in the perception stack.

---

## 1. Object Detection Module (`src.detection`)

### `class RoadObjectDetector`
Detector wrapper for BDD100K road agents using YOLOv8.

```python
RoadObjectDetector(
    model_path: Union[str, Path] = "models/best.pt",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    imgsz: int = 640,
    device: str = "cpu"
)
```

#### Methods:
* **`detect(frame: Union[np.ndarray, str, Path]) -> DetectionResult`**: Runs CPU inference on an input frame or image file path. Returns a structured `DetectionResult`. Raises `ValueError` on empty or invalid frames.
* **`draw_detections(frame: np.ndarray, result: DetectionResult, color_map: Optional[Dict] = None, line_thickness: int = 2) -> np.ndarray`**: Renders bounding boxes and label badges onto a copy of the input image.

### `dataclass DetectionResult`
* `boxes`: `np.ndarray` of shape `(N, 4)` in `[x1, y1, x2, y2]` coordinates.
* `scores`: `np.ndarray` of shape `(N,)` in range `[0.0, 1.0]`.
* `class_ids`: `np.ndarray` of shape `(N,)` integer IDs.
* `class_names`: `List[str]` class names.
* `inference_time_ms`: `float` execution latency in milliseconds.
* `count`: `int` (property returning number of detections).
* `is_empty()`: `bool` (returns `True` if no objects were detected).

---

## 2. Multi-Object Tracking Module (`src.tracking`)

### `class RoadObjectTracker`
ByteTrack-based tracker maintaining persistent object identities across consecutive frames.

```python
RoadObjectTracker(
    model_path: Union[str, Path] = "models/best.pt",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    imgsz: int = 640,
    tracker_type: str = "bytetrack.yaml",
    device: str = "cpu",
    max_trail_length: int = 25
)
```

#### Methods:
* **`update(frame: Union[np.ndarray, str, Path]) -> TrackingResult`**: Performs single-pass YOLO detection and ByteTrack association. Updates internal trajectory trails.
* **`reset() -> None`**: Clears active track histories and resets internal tracking state for new video sequences.
* **`draw_tracks(frame: np.ndarray, result: TrackingResult, draw_trails: bool = True, line_thickness: int = 2) -> np.ndarray`**: Draws bounding boxes with stable ID colors, track badges (`#ID:X class conf`), and trajectory motion trails.

### `dataclass TrackingResult`
* `boxes`: `np.ndarray` of shape `(N, 4)`.
* `scores`: `np.ndarray` of shape `(N,)`.
* `class_ids`: `np.ndarray` of shape `(N,)`.
* `class_names`: `List[str]`.
* `track_ids`: `np.ndarray` of shape `(N,)` with persistent integer IDs.
* `latency_ms`: `float`.

---

## 3. Lane Detection Module (`src.lanes`)

### `class OpenCVLaneDetector`
Classical OpenCV computer vision lane boundary detector.

```python
OpenCVLaneDetector(
    roi_top_ratio: float = 0.60,
    roi_bottom_width_ratio: float = 0.95,
    roi_top_width_ratio: float = 0.20,
    smoothing_factor: float = 0.80,
    min_slope: float = 0.35,
    max_slope: float = 2.50
)
```

#### Methods:
* **`detect(frame: np.ndarray) -> LaneDetectionResult`**: Executes color masking, Canny edge detection, Hough lines, slope classification, and temporal EMA smoothing.
* **`reset() -> None`**: Clears historical smoothing state.
* **`draw_lanes(frame: np.ndarray, result: LaneDetectionResult, corridor_color = (0, 220, 100), corridor_alpha = 0.25) -> np.ndarray`**: Overlays lane boundary lines, translucent green drivable corridor, and vehicle center offset telemetry.

### `dataclass LaneDetectionResult`
* `left_line`: `Optional[Tuple[int, int, int, int]]` left boundary `(x1, y1, x2, y2)`.
* `right_line`: `Optional[Tuple[int, int, int, int]]` right boundary `(x1, y1, x2, y2)`.
* `lane_polygon`: `Optional[np.ndarray]` polygon vertices representing drivable lane area.
* `lane_center_offset_px`: `Optional[float]` deviation in pixels from ego-vehicle center.
* `confidence`: `float` detection confidence `[0.0, 1.0]`.
* `has_lanes`: `bool` property.

---

## 4. Monocular Depth Estimation Module (`src.depth`)

### `class MonocularDepthEstimator`
Relative monocular depth estimator using `MiDaS_small`.

```python
MonocularDepthEstimator(
    model_type: str = "MiDaS_small",
    device: str = "cpu",
    cache_dir: Optional[Union[str, Path]] = None
)
```

#### Methods:
* **`estimate(frame: np.ndarray) -> DepthResult`**: Computes relative depth map and inferno colorized visualization.
* **`get_object_proximity(depth_result: DepthResult, box: Tuple[int, int, int, int]) -> float`**: Computes median relative proximity index in `[0.0, 1.0]` for an object's bounding box.
* **`create_side_by_side(frame: np.ndarray, depth_result: DepthResult) -> np.ndarray`**: Returns side-by-side composite with uncalibrated disclaimer banner.

---

## 5. Perception Pipeline Module (`src.pipeline`)

### `class PerceptionPipeline`
End-to-end orchestrator fusing detection, tracking, lanes, and depth.

```python
PerceptionPipeline(config: Optional[PipelineConfig] = None)
```

#### Methods:
* **`process_frame(frame: np.ndarray, frame_index: Optional[int] = None) -> PipelineFrameResult`**: Executes perception stack and renders HUD telemetry banner.
* **`process_video(input_source, output_path = None, max_frames = None) -> Dict[str, Any]`**: Processes video file or camera stream, writes annotated output MP4, and outputs structured telemetry JSON.

---

## 6. Deep Semantic Lane Segmentation (`src.lanes`)

### `class SegmentationLaneDetector`
Deep semantic lane segmentation using Torchvision LRASPP MobileNetV3 Large.

```python
SegmentationLaneDetector(
    device: str = "cpu",
    confidence_thresh: float = 0.50,
    road_class_idx: int = 0,
    input_size: Tuple[int, int] = (520, 520)
)
```

#### Methods:
* **`detect(frame: np.ndarray) -> LaneDetectionResult`**: Runs deep semantic segmentation inference, extracts binary drivable road corridor, computes bounding contour polygon, and determines vehicle lane center offset. Raises `ValueError` on empty or invalid frames.
* **`draw_lanes(frame: np.ndarray, result: LaneDetectionResult, corridor_color = (0, 200, 255), corridor_alpha = 0.35) -> np.ndarray`**: Overlays segmented drivable road mask, contour boundary, and center offset metrics onto the frame.

---

## 7. Simulation & Middleware Integration (`src.simulation`)

### `class CARLASimulatorClient`
Client lifecycle manager for connecting to Unreal Engine CARLA simulator servers.

```python
CARLASimulatorClient(config: Optional[CARLAClientConfig] = None)
```

#### Methods:
* **`check_simulator_port(host: str = "localhost", port: int = 2000, timeout: float = 0.5) -> bool`**: Static utility testing whether the CARLA RPC port is open and listening.
* **`connect() -> bool`**: Attempts socket health check and establishes CARLA client connection. Returns `False` gracefully if simulator server is offline.
* **`setup_sensor(vehicle = None) -> bool`**: Configures RGBA camera sensor and registers frame listener callback.
* **`start_streaming() -> None`**: Begins queuing incoming simulator frames.
* **`get_latest_frame(timeout_sec: float = 0.1) -> Optional[np.ndarray]`**: Fetches most recent BGR frame from the thread-safe queue.
* **`stop() -> None`**: Stops sensor stream.
* **`disconnect() -> None`**: Safely tears down sensor actors and closes client session.

### `class CARLACameraAdapter`
Bridge parsing CARLA's raw 4-channel BGRA byte buffers into standardized 3-channel OpenCV BGR images.

```python
CARLACameraAdapter()
```

#### Methods:
* **`parse_carla_raw_image(raw_bytes: bytes, width: int = 1280, height: int = 720) -> np.ndarray`**: Converts raw memory buffer to uint8 BGR array.

### `class PerceptionROS2Node`
Isolated ROS2 Humble perception node feeding sensor frames into `PerceptionPipeline` and publishing vision results.

```python
PerceptionROS2Node(node_name: str = "autonomous_driving_perception_node", use_ros2_runtime: bool = False)
```

#### Methods:
* **`image_callback(msg: Any) -> PipelineFrameResult`**: Receives ROS2 `sensor_msgs/Image` (or mock frame), executes full perception stack, and publishes detections, lane markers, and telemetry.
* **`decode_sensor_msgs_image(msg: Any) -> np.ndarray`**: Decodes ROS2 sensor image into NumPy BGR frame with fallback pure-Python byte parsing for non-ROS2 environments.

---

## 8. REST API Endpoints (`server.py`)

The stack provides a high-throughput, non-blocking asynchronous REST API powering the HTML5/CSS3/Vanilla JS interface.

### Endpoints Reference:

| Method | Endpoint | Description | Request Parameters / Body | Response Schema |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | Serves dashboard interface | None | HTML document (`frontend/index.html`) |
| `GET` | `/api/health` | Backend status & model diagnostics | None | `{ "status": "online", "device": "cpu", "models": {...}, "active_jobs": int }` |
| `GET` | `/api/samples` | List available pre-packaged driving videos | None | `[ { "id": "demo_banner", "name": "...", "frames": 255, "path": "..." } ]` |
| `POST` | `/api/process` | Submit video or sample for perception processing | `file` (UploadFile) OR `sample_id` (str), `enable_tracking` (bool), `enable_lanes` (bool), `lane_method` ("classical"\|"segmentation"), `enable_depth` (bool), `depth_freq` (int), `conf_thresh` (float), `max_frames` (int), `model_variant` ("baseline"\|"finetuned") | `{ "job_id": "...", "status": "queued", "message": "...", "input_filename": "..." }` |
| `GET` | `/api/jobs/{id}` | Poll processing progress & live telemetry | `job_id` (path) | `{ "job_id": "...", "status": "...", "progress": float, "current_frame": int, "total_frames": int, "live_metrics": {...}, "summary": {...} }` |
| `GET` | `/api/jobs/{id}/results` | Fetch full completed results & track list | `job_id` (path) | `{ "job_id": "...", "status": "...", "summary": {...}, "detected_objects": [...], "class_counts": {...}, "total_detected": int, "total_tracked": int }` |
| `POST` | `/api/jobs/{id}/cancel` | Gracefully cancel an active job | `job_id` (path) | `{ "job_id": "...", "status": "cancelled", "message": "..." }` |
| `GET` | `/api/jobs/{id}/video` | Stream/download annotated perception MP4 | `job_id` (path) | Streaming binary `video/mp4` response |
| `GET` | `/api/jobs/{id}/original-video` | Stream original input video | `job_id` (path) | Streaming binary `video/mp4` response |
| `GET` | `/api/jobs/{id}/telemetry` | Download full structured JSON telemetry log | `job_id` (path) | JSON file download (`application/json`) |

