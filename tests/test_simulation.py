"""
Tests for Simulation and Sensor Interface Module (CARLA Camera Adapter and Virtual Stream).
Validates raw memory buffer parsing, format conversions, and frame data containers.
"""
import pytest
import numpy as np
from src.simulation.camera_adapter import SimulationFrame, CARLACameraAdapter, ROS2_SPECIFICATION

def test_simulation_frame_dataclass():
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame = SimulationFrame(
        frame_id=42,
        timestamp_sec=1234567.89,
        image=dummy_img,
        fov=90.0,
        width=1280,
        height=720,
    )
    assert frame.frame_id == 42
    assert frame.image.shape == (720, 1280, 3)
    assert frame.fov == 90.0

def test_carla_camera_adapter_raw_bgra_parsing():
    adapter = CARLACameraAdapter()
    width, height = 64, 48
    # Create synthetic 4-channel BGRA byte array (CARLA raw sensor format)
    bgra_raw = np.full((height, width, 4), [100, 150, 200, 255], dtype=np.uint8).tobytes()
    
    bgr = adapter.parse_carla_raw_image(bgra_raw, width=width, height=height)
    
    assert bgr.shape == (height, width, 3), "Output should be 3-channel BGR"
    assert bgr[0, 0, 0] == 100, "Blue channel preserved"
    assert bgr[0, 0, 1] == 150, "Green channel preserved"
    assert bgr[0, 0, 2] == 200, "Red channel preserved"

def test_ros2_specification_schema():
    assert "node_name" in ROS2_SPECIFICATION
    assert "subscribers" in ROS2_SPECIFICATION
    assert "publishers" in ROS2_SPECIFICATION
    assert len(ROS2_SPECIFICATION["subscribers"]) >= 2
    assert len(ROS2_SPECIFICATION["publishers"]) >= 4

def test_carla_client_port_check():
    from src.simulation.carla_client import CARLASimulatorClient
    # Port 59999 should not have a CARLA server running
    assert CARLASimulatorClient.check_simulator_port(host="127.0.0.1", port=59999, timeout=0.1) is False

def test_carla_client_config_defaults():
    from src.simulation.carla_client import CARLAClientConfig, CARLASimulatorClient
    cfg = CARLAClientConfig()
    assert cfg.host == "localhost"
    assert cfg.port == 2000
    assert cfg.image_width == 1280
    assert cfg.image_height == 720
    client = CARLASimulatorClient(config=cfg)
    assert client.is_connected is False

def test_carla_client_connect_without_server_returns_false():
    from src.simulation.carla_client import CARLASimulatorClient, CARLAClientConfig
    cfg = CARLAClientConfig(port=59998, timeout_sec=0.1)
    client = CARLASimulatorClient(config=cfg)
    # Attempting connection to closed port should gracefully return False without raising unhandled exception
    connected = client.connect()
    assert connected is False

def test_ros2_node_standalone_initialization():
    from src.simulation.ros2_perception_node import PerceptionROS2Node
    node = PerceptionROS2Node(use_ros2_runtime=False)
    assert node.node_name == "autonomous_driving_perception_node"
    assert node.pipeline is not None

def test_ros2_node_image_callback_dataflow():
    from src.simulation.ros2_perception_node import PerceptionROS2Node
    from src.pipeline.perception_pipeline import PipelineFrameResult
    node = PerceptionROS2Node(use_ros2_runtime=False)
    
    # Create mock sensor_msgs/Image object
    class MockSensorImage:
        def __init__(self, h, w):
            self.height = h
            self.width = w
            self.encoding = "bgr8"
            self.data = np.full((h, w, 3), 120, dtype=np.uint8).tobytes()
            
    mock_msg = MockSensorImage(h=360, w=640)
    res = node.image_callback(mock_msg)
    
    assert isinstance(res, PipelineFrameResult)
    assert res.annotated_frame.shape == (360, 640, 3)
    assert node.frame_counter == 1
    assert "total_latency_ms" in node.last_telemetry

def test_async_frame_producer_queueing():
    import queue
    from src.simulation.async_dataflow import AsyncFrameProducer
    q = queue.Queue(maxsize=5)
    producer = AsyncFrameProducer(frame_queue=q)
    
    # Valid frame
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    sim_frame = producer.publish_frame(img, frame_id=1)
    assert q.qsize() == 1
    assert sim_frame.frame_id == 1
    assert sim_frame.image.shape == (240, 320, 3)
    
    # Invalid frame error
    with pytest.raises(ValueError):
        producer.publish_frame(None)
    with pytest.raises(ValueError):
        producer.publish_frame(np.array([]))

def test_async_producer_consumer_dataflow():
    import queue
    import time
    from src.simulation.async_dataflow import AsyncFrameProducer, AsyncPerceptionConsumer
    from src.pipeline.perception_pipeline import PerceptionPipeline, PipelineConfig, PipelineFrameResult
    
    in_q = queue.Queue()
    out_q = queue.Queue()
    
    # Lightweight test config (no depth, fast inference)
    cfg = PipelineConfig(
        model_path="models/best.pt",
        enable_tracking=True,
        enable_lanes=True,
        enable_depth=False,
        imgsz=320,
        device="cpu",
    )
    pipeline = PerceptionPipeline(config=cfg)
    
    producer = AsyncFrameProducer(frame_queue=in_q)
    consumer = AsyncPerceptionConsumer(in_queue=in_q, out_queue=out_q, pipeline=pipeline)
    
    consumer.start()
    try:
        # Publish 2 test frames
        frame = np.full((320, 640, 3), 100, dtype=np.uint8)
        producer.publish_frame(frame, frame_id=101)
        producer.publish_frame(frame, frame_id=102)
        
        # Collect results from out_queue
        res1 = out_q.get(timeout=5.0)
        res2 = out_q.get(timeout=5.0)
        
        assert isinstance(res1, PipelineFrameResult)
        assert res1.frame_index == 101
        assert res1.total_latency_ms > 0
        
        assert isinstance(res2, PipelineFrameResult)
        assert res2.frame_index == 102
        assert consumer.processed_count == 2
    finally:
        consumer.stop()
        consumer.join(timeout=2.0)


