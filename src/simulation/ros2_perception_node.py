"""
ROS2 Perception Node for Autonomous Driving Perception Stack.

Provides:
- PerceptionROS2Node: Subscribes to raw camera images and camera info,
  processes frames through PerceptionPipeline, and publishes Detection2DArray,
  tracking IDs, annotated visualization image feed, and telemetry JSON.
- Isolated ROS2 dependencies: Safely imports rclpy and ROS2 message types
  when available (Ubuntu 22.04 / ROS2 Humble), and provides mock containers
  and fallback interfaces for execution and testing on Windows 11.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Isolated ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image as ROSImage
    from std_msgs.msg import Header, String as ROSString
    from vision_msgs.msg import (
        BoundingBox2D,
        Detection2D,
        Detection2DArray,
        ObjectHypothesisWithPose,
    )
    ROS2_AVAILABLE = True
except ImportError:
    Node = object
    ROS2_AVAILABLE = False

from src.pipeline.perception_pipeline import PerceptionPipeline, PipelineConfig, PipelineFrameResult

logger = logging.getLogger(__name__)


class PerceptionROS2Node(Node):
    """
    ROS2 Node orchestrating road perception:
    - Subscribes: /carla/ego_vehicle/rgb_front/image_color (sensor_msgs/Image)
    - Publishes:
        /perception/detections_2d (vision_msgs/Detection2DArray)
        /perception/tracks (vision_msgs/Detection2DArray)
        /perception/annotated_feed (sensor_msgs/Image)
        /perception/telemetry (std_msgs/String)
    """

    def __init__(
        self,
        node_name: str = "autonomous_driving_perception_node",
        pipeline_config: Optional[PipelineConfig] = None,
        use_ros2_runtime: bool = True,
    ):
        self.node_name = node_name
        self.is_live_ros2 = ROS2_AVAILABLE and use_ros2_runtime

        if self.is_live_ros2:
            super().__init__(node_name)
            logger.info("Initializing live ROS2 Perception Node: %s", node_name)
            self._init_ros2_io()
        else:
            logger.info("Initializing PerceptionROS2Node in standalone/simulation mode (ROS2 unavailable)")

        # Initialize perception stack
        self.pipeline = PerceptionPipeline(config=pipeline_config)
        self.frame_counter = 0
        self.last_telemetry: Dict[str, Any] = {}

    def _init_ros2_io(self):
        """Initializes ROS2 publishers and subscribers."""
        if not self.is_live_ros2:
            return

        # QoS Profiles
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribers
        self.sub_image = self.create_subscription(
            ROSImage,
            "/carla/ego_vehicle/rgb_front/image_color",
            self.image_callback,
            sensor_qos,
        )

        # Publishers
        self.pub_detections = self.create_publisher(
            Detection2DArray, "/perception/detections_2d", reliable_qos
        )
        self.pub_tracks = self.create_publisher(
            Detection2DArray, "/perception/tracks", reliable_qos
        )
        self.pub_annotated = self.create_publisher(
            ROSImage, "/perception/annotated_feed", sensor_qos
        )
        self.pub_telemetry = self.create_publisher(
            ROSString, "/perception/telemetry", reliable_qos
        )

    def image_callback(self, msg: Any) -> PipelineFrameResult:
        """
        Callback processing incoming camera image messages.
        Works with live sensor_msgs/Image or synthetic mock message objects.
        """
        self.frame_counter += 1
        t_start = time.perf_counter()

        # Convert ROS Image to OpenCV BGR numpy array
        cv_image = self.ros_image_to_cv2(msg)

        # Process through perception pipeline
        frame_result = self.pipeline.process_frame(cv_image, frame_index=self.frame_counter)
        self.last_telemetry = frame_result.telemetry

        # Publish results if in live ROS2 mode
        if self.is_live_ros2:
            self._publish_ros2_outputs(frame_result, msg)

        return frame_result

    def ros_image_to_cv2(self, msg: Any) -> np.ndarray:
        """Converts ROS sensor_msgs/Image byte data into BGR numpy array."""
        if hasattr(msg, "data") and hasattr(msg, "height") and hasattr(msg, "width"):
            data = bytes(msg.data)
            h, w = msg.height, msg.width
            encoding = getattr(msg, "encoding", "bgr8").lower()

            if "bgra" in encoding or len(data) == h * w * 4:
                arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4))
                return arr[:, :, :3]  # Drop alpha
            elif "rgb" in encoding:
                arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                return np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
        elif isinstance(msg, np.ndarray):
            return msg
        else:
            raise ValueError(f"Unsupported image message format: {type(msg)}")

    def _publish_ros2_outputs(self, result: PipelineFrameResult, src_msg: Any):
        """Constructs and publishes ROS2 detection, tracking, annotated image, and telemetry messages."""
        if not self.is_live_ros2:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = getattr(src_msg, "header", Header()).frame_id or "camera_front_link"

        # 1. Publish Telemetry String
        tel_msg = ROSString()
        tel_msg.data = json.dumps(result.telemetry)
        self.pub_telemetry.publish(tel_msg)

        # 2. Publish 2D Detections Array
        det_array = Detection2DArray()
        det_array.header = header
        if result.detections and not result.detections.is_empty():
            for i in range(result.detections.count):
                det = Detection2D()
                det.header = header
                box = result.detections.boxes[i]
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(result.detections.class_ids[i])
                hyp.hypothesis.score = float(result.detections.scores[i])
                det.results.append(hyp)
                det.bbox = self._box_to_ros_bbox2d(box)
                det_array.detections.append(det)
        self.pub_detections.publish(det_array)

        # 3. Publish Tracking Detections Array
        track_array = Detection2DArray()
        track_array.header = header
        if result.tracking and not result.tracking.is_empty():
            for i in range(result.tracking.count):
                trk = Detection2D()
                trk.header = header
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = f"id_{result.tracking.track_ids[i]}:{result.tracking.class_names[i]}"
                hyp.hypothesis.score = float(result.tracking.scores[i])
                trk.results.append(hyp)
                trk.bbox = self._box_to_ros_bbox2d(result.tracking.boxes[i])
                track_array.detections.append(trk)
        self.pub_tracks.publish(track_array)

        # 4. Publish Annotated Image
        annotated_msg = self._cv2_to_ros_image(result.annotated_frame, header)
        self.pub_annotated.publish(annotated_msg)

    def _box_to_ros_bbox2d(self, box: np.ndarray) -> Any:
        """Converts [x1, y1, x2, y2] to vision_msgs/BoundingBox2D."""
        x1, y1, x2, y2 = box
        bbox = BoundingBox2D()
        bbox.center.position.x = float((x1 + x2) / 2.0)
        bbox.center.position.y = float((y1 + y2) / 2.0)
        bbox.size_x = float(abs(x2 - x1))
        bbox.size_y = float(abs(y2 - y1))
        return bbox

    def _cv2_to_ros_image(self, img: np.ndarray, header: Any) -> Any:
        """Converts OpenCV BGR image to sensor_msgs/Image."""
        msg = ROSImage()
        msg.header = header
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = img.tobytes()
        return msg


def main():
    """CLI launcher for the ROS2 Perception Node."""
    print("=== Launching ROS2 Perception Node ===")
    if not ROS2_AVAILABLE:
        print("\nERROR: ROS2 (rclpy) is not installed in the current Python environment.")
        print("This node is designed for execution in a ROS2 Humble environment on Ubuntu 22.04.")
        print("\nLaunch Instructions for Ubuntu 22.04 / WSL2:")
        print("  1. Source ROS2 environment: source /opt/ros/humble/setup.bash")
        print("  2. Install ROS2 packages: sudo apt-get install ros-humble-vision-msgs ros-humble-sensor-msgs")
        print("  3. Run node: python src/simulation/ros2_perception_node.py")
        return

    rclpy.init()
    node = PerceptionROS2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
