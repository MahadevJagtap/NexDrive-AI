"""
Asynchronous Producer-Consumer Perception Dataflow Module.
Provides a lightweight, thread-safe queue-based frame producer and perception consumer
as a laptop-compatible alternative to ROS2 pub/sub middleware.

DISCLAIMER:
This module demonstrates asynchronous decoupled producer-consumer dataflow in pure Python.
It is an architectural ALTERNATIVE for local Windows testing, NOT actual ROS2 DDS messaging.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from src.pipeline.perception_pipeline import PerceptionPipeline, PipelineConfig, PipelineFrameResult
from src.simulation.camera_adapter import SimulationFrame

logger = logging.getLogger(__name__)


class AsyncFrameProducer:
    """
    Simulates a camera publisher publishing frames to a thread-safe queue.
    Mimics a ROS2 sensor_msgs/Image publisher.
    """

    def __init__(self, frame_queue: queue.Queue, max_queue_size: int = 10):
        self.frame_queue = frame_queue
        self.frame_counter = 0

    def publish_frame(
        self,
        image: np.ndarray,
        frame_id: Optional[int] = None,
        timestamp_sec: Optional[float] = None,
        block: bool = True,
        timeout: Optional[float] = 1.0,
    ) -> SimulationFrame:
        """Publishes an image frame to the processing queue."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Cannot publish an invalid or empty image.")

        self.frame_counter += 1
        fid = frame_id if frame_id is not None else self.frame_counter
        ts = timestamp_sec if timestamp_sec is not None else time.time()
        h, w = image.shape[:2]

        sim_frame = SimulationFrame(
            frame_id=fid,
            timestamp_sec=ts,
            image=image,
            width=w,
            height=h,
        )

        try:
            self.frame_queue.put(sim_frame, block=block, timeout=timeout)
        except queue.Full:
            logger.warning(f"Frame queue full; dropped frame #{fid}")
            raise

        return sim_frame


class AsyncPerceptionConsumer:
    """
    Asynchronous perception worker consuming frames from input queue,
    running PerceptionPipeline inference, and publishing results to an output queue.
    Mimics a ROS2 perception node subscribing to /camera/image_raw.
    """

    def __init__(
        self,
        in_queue: queue.Queue,
        out_queue: Optional[queue.Queue] = None,
        pipeline: Optional[PerceptionPipeline] = None,
        result_callback: Optional[Callable[[PipelineFrameResult], None]] = None,
    ):
        self.in_queue = in_queue
        self.out_queue = out_queue if out_queue is not None else queue.Queue()
        self.pipeline = pipeline or PerceptionPipeline()
        self.result_callback = result_callback

        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self.processed_count = 0
        self.last_result: Optional[PipelineFrameResult] = None

    def start(self) -> None:
        """Start consumer background processing thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("AsyncPerceptionConsumer worker thread started.")

    def stop(self) -> None:
        """Signal worker thread to terminate."""
        self._stop_event.set()

    def join(self, timeout: float = 3.0) -> None:
        """Wait for worker thread to exit."""
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)

    def _worker_loop(self) -> None:
        """Internal queue consumption and processing loop."""
        while not self._stop_event.is_set():
            try:
                # Poll queue with 100ms timeout to allow checking _stop_event
                sim_frame: SimulationFrame = self.in_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # Run full perception stack
                result = self.pipeline.process_frame(
                    frame=sim_frame.image,
                    frame_index=sim_frame.frame_id,
                )
                self.processed_count += 1
                self.last_result = result

                # Push to output queue (mimicking output publisher)
                self.out_queue.put(result)

                # Optional callback notification
                if self.result_callback:
                    self.result_callback(result)
            except Exception as e:
                logger.error(f"Error processing frame #{sim_frame.frame_id}: {e}", exc_info=True)
            finally:
                self.in_queue.task_done()
