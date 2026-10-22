"""
Simulation, CARLA Sensor Adapter, and ROS2 Middleware Package.
Provides:
- CARLACameraAdapter: Raw CARLA BGRA memory decoding.
- CARLASimulatorClient: Lifecycle management for live CARLA 0.9.x world sessions.
- VirtualCameraStream: Replayer for offline deterministic sensor streams.
- PerceptionROS2Node: Real ROS2 publisher and subscriber interface.
- ROS2_SPECIFICATION: Formal topic, message, and QoS schema.
"""
from src.simulation.camera_adapter import (
    ROS2_SPECIFICATION,
    CARLACameraAdapter,
    SimulationFrame,
    VirtualCameraStream,
)
from src.simulation.async_dataflow import AsyncFrameProducer, AsyncPerceptionConsumer
from src.simulation.carla_client import CARLAClientConfig, CARLASimulatorClient
from src.simulation.ros2_perception_node import PerceptionROS2Node

__all__ = [
    "CARLACameraAdapter",
    "CARLASimulatorClient",
    "CARLAClientConfig",
    "VirtualCameraStream",
    "SimulationFrame",
    "ROS2_SPECIFICATION",
    "PerceptionROS2Node",
    "AsyncFrameProducer",
    "AsyncPerceptionConsumer",
]
