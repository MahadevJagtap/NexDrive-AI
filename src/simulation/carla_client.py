"""
CARLA Simulator Camera Client for Autonomous Driving Perception Stack.

Provides:
- CARLAClientConfig: Configuration for host, port, sensor dimensions, and ego attachment.
- CARLASimulatorClient: Manages full simulator lifecycle (connect, actor attachment,
  sensor callback routing, and graceful cleanup) with isolated carla imports.
- Socket health probe: Verifies simulator port connectivity independently of Python wheel.
"""
from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    carla = None
    CARLA_AVAILABLE = False

from src.simulation.camera_adapter import CARLACameraAdapter, SimulationFrame

logger = logging.getLogger(__name__)


@dataclass
class CARLAClientConfig:
    """Configuration for CARLA Simulator Client connection and sensor spawning."""
    host: str = "localhost"
    port: int = 2000
    timeout_sec: float = 5.0
    image_width: int = 1280
    image_height: int = 720
    fov: float = 90.0
    fps: int = 30
    sensor_location: Tuple[float, float, float] = (1.5, 0.0, 1.7)  # Front windshield (x, y, z)
    sensor_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Pitch, yaw, roll
    attach_to_ego: bool = True
    ego_vehicle_filter: str = "vehicle.tesla.model3"


class CARLASimulatorClient:
    """
    Manages connection to a CARLA 0.9.x Unreal Engine simulation world.
    Spawns RGB camera sensor on the ego vehicle and routes frame callbacks
    directly to the perception pipeline.
    """

    def __init__(
        self,
        config: Optional[CARLAClientConfig] = None,
        frame_callback: Optional[Callable[[SimulationFrame], None]] = None,
    ):
        self.config = config or CARLAClientConfig()
        self.frame_callback = frame_callback
        self.adapter = CARLACameraAdapter(callback=self._on_adapted_frame)

        self.client: Optional[Any] = None
        self.world: Optional[Any] = None
        self.ego_vehicle: Optional[Any] = None
        self.camera_actor: Optional[Any] = None
        self.is_connected = False
        self.is_streaming = False
        self.frame_count = 0

    @staticmethod
    def check_simulator_port(host: str = "localhost", port: int = 2000, timeout: float = 2.0) -> bool:
        """Lightweight TCP socket check to test if a CARLA server is listening on port 2000."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False

    def connect(self) -> bool:
        """Establishes connection to the CARLA RPC server."""
        if not CARLA_AVAILABLE:
            logger.warning("CARLA Python client library ('carla') is not installed in the environment.")
            return False

        if not self.check_simulator_port(self.config.host, self.config.port, timeout=self.config.timeout_sec):
            logger.warning("CARLA server unreachable at %s:%d", self.config.host, self.config.port)
            return False

        try:
            logger.info("Connecting to CARLA RPC at %s:%d...", self.config.host, self.config.port)
            self.client = carla.Client(self.config.host, self.config.port)
            self.client.set_timeout(self.config.timeout_sec)
            self.world = self.client.get_world()
            self.is_connected = True
            logger.info("Successfully connected to CARLA world: %s", self.world.get_map().name)
            return True
        except Exception as e:
            logger.error("Failed to connect to CARLA server: %s", e)
            self.is_connected = False
            return False

    def setup_sensor(self) -> bool:
        """Spawns RGB camera sensor attached to ego vehicle."""
        if not self.is_connected or self.world is None:
            logger.error("Cannot setup sensor: Not connected to CARLA world.")
            return False

        try:
            blueprint_lib = self.world.get_blueprint_library()
            camera_bp = blueprint_lib.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", str(self.config.image_width))
            camera_bp.set_attribute("image_size_y", str(self.config.image_height))
            camera_bp.set_attribute("fov", str(self.config.fov))
            camera_bp.set_attribute("sensor_tick", str(1.0 / self.config.fps))

            # Transform
            loc = carla.Location(*self.config.sensor_location)
            rot = carla.Rotation(*self.config.sensor_rotation)
            sensor_transform = carla.Transform(loc, rot)

            # Ego vehicle association
            if self.config.attach_to_ego:
                actors = self.world.get_actors().filter("vehicle.*")
                if len(actors) > 0:
                    self.ego_vehicle = actors[0]
                    self.camera_actor = self.world.spawn_actor(
                        camera_bp, sensor_transform, attach_to=self.ego_vehicle
                    )
                else:
                    logger.warning("No existing vehicle found. Spawning camera standalone at map origin.")
                    self.camera_actor = self.world.spawn_actor(camera_bp, sensor_transform)
            else:
                self.camera_actor = self.world.spawn_actor(camera_bp, sensor_transform)

            logger.info("Spawned CARLA RGB Camera Actor ID: %d", self.camera_actor.id)
            return True
        except Exception as e:
            logger.error("Failed to spawn CARLA camera actor: %s", e)
            return False

    def start_streaming(self) -> bool:
        """Starts listening to camera sensor stream."""
        if self.camera_actor is None:
            if not self.setup_sensor():
                return False

        try:
            self.camera_actor.listen(lambda img: self.adapter.on_carla_image(img))
            self.is_streaming = True
            logger.info("CARLA Camera stream listener active.")
            return True
        except Exception as e:
            logger.error("Failed to start camera listener: %s", e)
            self.is_streaming = False
            return False

    def _on_adapted_frame(self, frame: SimulationFrame):
        """Internal callback receiving parsed SimulationFrame from adapter."""
        self.frame_count += 1
        if self.frame_callback:
            self.frame_callback(frame)

    def stop(self):
        """Stops sensor listening and destroys spawned actors."""
        if self.camera_actor is not None:
            try:
                self.camera_actor.stop()
                self.camera_actor.destroy()
                logger.info("Destroyed CARLA camera actor.")
            except Exception as e:
                logger.warning("Error destroying camera actor: %s", e)
            finally:
                self.camera_actor = None
        self.is_streaming = False

    def disconnect(self):
        """Disconnects client session and cleans up."""
        self.stop()
        self.client = None
        self.world = None
        self.is_connected = False
        logger.info("Disconnected from CARLA simulator.")


def main():
    """CLI diagnostic for checking CARLA Simulator availability."""
    print("=== CARLA Simulator Client Diagnostics ===")
    print(f"CARLA Python package installed: {CARLA_AVAILABLE}")

    is_port_open = CARLASimulatorClient.check_simulator_port("localhost", 2000, timeout=1.0)
    print(f"CARLA Server reachable on localhost:2000: {is_port_open}")

    if not is_port_open:
        print("\nSTATUS: CARLA Simulator server is NOT running locally.")
        print("To run live simulation:")
        print("  1. Ensure an NVIDIA GPU workstation is available.")
        print("  2. Launch CARLA server: CarlaUE4.exe -carla-server -benchmark -fps=30")
        print("  3. Run client: python src/simulation/carla_client.py")
    else:
        print("\nCARLA Server detected! Initializing client session...")
        client = CARLASimulatorClient()
        if client.connect():
            print("Connected successfully! Stopping session.")
            client.disconnect()


if __name__ == "__main__":
    main()
