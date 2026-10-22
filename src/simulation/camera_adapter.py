"""
Simulation and Sensor Interface Module.
Provides CARLA camera adapter, virtual camera streamer, and ROS2 bridge specifications.
Designed for integrating synthetic sensor streams from simulators (CARLA) or replaying
test datasets into the perception pipeline.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimulationFrame:
    """
    Standardized sensor frame container mirroring CARLA/ROS2 camera messages.
    """
    frame_id: int
    timestamp_sec: float
    image: np.ndarray  # BGR format (H, W, 3)
    fov: float = 90.0
    width: int = 1280
    height: int = 720
    camera_intrinsics: Optional[Dict[str, float]] = None
    sensor_transform: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CARLACameraAdapter:
    """
    Adapter converting CARLA Python API sensor callbacks to PerceptionPipeline frames.
    
    In a live CARLA environment:
    >>> camera_bp = blueprint_library.find('sensor.camera.rgb')
    >>> camera_bp.set_attribute('image_size_x', '1280')
    >>> camera_bp.set_attribute('image_size_y', '720')
    >>> camera_bp.set_attribute('fov', '90')
    >>> camera = world.spawn_actor(camera_bp, transform, attach_to=ego_vehicle)
    >>> adapter = CARLACameraAdapter()
    >>> camera.listen(lambda image: adapter.on_carla_image(image))
    """

    def __init__(self, callback: Optional[Callable[[SimulationFrame], None]] = None):
        self.callback = callback
        self.frame_counter = 0

    def parse_carla_raw_image(self, carla_raw_data: bytes, width: int, height: int) -> np.ndarray:
        """
        Convert CARLA BGRA raw memory buffer to standard OpenCV BGR image.
        CARLA RGB sensor outputs 4-channel BGRA byte arrays.
        """
        array = np.frombuffer(carla_raw_data, dtype=np.uint8)
        array = np.reshape(array, (height, width, 4))
        # Drop Alpha channel: BGRA -> BGR
        bgr = array[:, :, :3]
        return bgr

    def on_carla_image(self, carla_image: Any) -> SimulationFrame:
        """
        Sensor listener callback receiving carla.Image objects.
        """
        self.frame_counter += 1
        bgr = self.parse_carla_raw_image(
            carla_image.raw_data,
            carla_image.width,
            carla_image.height,
        )

        sim_frame = SimulationFrame(
            frame_id=carla_image.frame if hasattr(carla_image, "frame") else self.frame_counter,
            timestamp_sec=carla_image.timestamp if hasattr(carla_image, "timestamp") else time.time(),
            image=bgr,
            fov=carla_image.fov if hasattr(carla_image, "fov") else 90.0,
            width=carla_image.width,
            height=carla_image.height,
        )

        if self.callback:
            self.callback(sim_frame)

        return sim_frame


class VirtualCameraStream:
    """
    Lightweight synthetic camera simulator for local Windows testing.
    Replays video streams or image folders adhering to the CARLA/ROS2 frame interface.
    """

    def __init__(
        self,
        video_source: Union[str, Path],
        target_fps: float = 30.0,
        loop: bool = False,
    ):
        self.video_source = Path(video_source)
        self.target_fps = target_fps
        self.loop = loop
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0.033

        if not self.video_source.exists():
            raise FileNotFoundError(f"Virtual camera source not found: {self.video_source}")

    def stream(self, max_frames: Optional[int] = None) -> Generator[SimulationFrame, None, None]:
        """
        Yields SimulationFrame objects simulating real-time camera sensor delivery.
        """
        cap = cv2.VideoCapture(str(self.video_source))
        frame_idx = 0
        start_sim_time = time.time()

        try:
            while True:
                if max_frames and frame_idx >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    if self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                frame_idx += 1
                sim_timestamp = start_sim_time + (frame_idx * self.frame_interval)
                h, w = frame.shape[:2]

                sim_frame = SimulationFrame(
                    frame_id=frame_idx,
                    timestamp_sec=sim_timestamp,
                    image=frame,
                    width=w,
                    height=h,
                    fov=90.0,
                    metadata={"source": str(self.video_source)},
                )

                yield sim_frame
        finally:
            cap.release()


# ROS2 Humble/Iron Architectural Blueprint
ROS2_SPECIFICATION = {
    "node_name": "autonomous_driving_perception_node",
    "subscribers": [
        {
            "topic": "/carla/ego_vehicle/rgb_front/image_color",
            "msg_type": "sensor_msgs/msg/Image",
            "qos": "SensorDataQoS (Best Effort)",
            "description": "Raw RGB/BGR camera stream from front bumper camera.",
        },
        {
            "topic": "/carla/ego_vehicle/rgb_front/camera_info",
            "msg_type": "sensor_msgs/msg/CameraInfo",
            "qos": "TransientLocal",
            "description": "Camera intrinsic calibration matrix (fx, fy, cx, cy) and distortion coefficients.",
        },
    ],
    "publishers": [
        {
            "topic": "/perception/detections_2d",
            "msg_type": "vision_msgs/msg/Detection2DArray",
            "qos": "Reliable",
            "description": "Bounding boxes, class hypothesis IDs, and confidence scores.",
        },
        {
            "topic": "/perception/tracks",
            "msg_type": "vision_msgs/msg/Detection2DArray",
            "qos": "Reliable",
            "description": "Persistent multi-object tracking IDs and velocities across consecutive frames.",
        },
        {
            "topic": "/perception/lane_boundaries",
            "msg_type": "visualization_msgs/msg/MarkerArray",
            "qos": "Best Effort",
            "description": "Polynomial line strips and drivable corridor polygon markers.",
        },
        {
            "topic": "/perception/annotated_feed",
            "msg_type": "sensor_msgs/msg/Image",
            "qos": "SensorDataQoS (Best Effort)",
            "description": "Annotated video stream for debugging in RViz2.",
        },
    ],
}
