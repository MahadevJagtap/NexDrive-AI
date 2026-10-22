"""
Multi-Object Tracker (MOT) module for autonomous driving perception.
Utilizes ByteTrack integrated with YOLOv8 for persistent vehicle and pedestrian tracking.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackingResult:
    """Container for tracking outputs on a single frame."""
    boxes: np.ndarray       # Shape: (N, 4) in [x1, y1, x2, y2]
    scores: np.ndarray      # Shape: (N,) confidence scores
    class_ids: np.ndarray   # Shape: (N,) class IDs
    class_names: List[str]  # List of class names
    track_ids: np.ndarray   # Shape: (N,) persistent tracking IDs
    latency_ms: float       # Tracking + inference latency in milliseconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.boxes)

    def is_empty(self) -> bool:
        return self.count == 0


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Generates a stable, visually distinct BGR color from a track ID."""
    # Golden ratio color generation
    hue = int((track_id * 137.508) % 180)
    hsv_pixel = np.uint8([[[hue, 220, 240]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


class RoadObjectTracker:
    """
    ByteTrack-based multi-object tracker for autonomous driving perception.
    Maintains persistent object identities, tracks motion histories, and handles occlusion.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/best.pt",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        imgsz: int = 640,
        tracker_type: str = "bytetrack.yaml",
        device: str = "cpu",
        max_trail_length: int = 25,
    ):
        """
        Initialize the tracker with a YOLOv8 detection model and ByteTrack configuration.
        """
        self.model_path = Path(model_path)
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.imgsz = int(imgsz)
        self.tracker_type = tracker_type
        self.device = device.lower()
        self.max_trail_length = max_trail_length

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.model_path}")

        logger.info(f"Initializing ByteTrack with model {self.model_path} on {self.device}...")
        from ultralytics import YOLO
        self.model = YOLO(str(self.model_path))
        self.names: Dict[int, str] = self.model.names

        # Trajectory trail history: track_id -> deque of (cx, cy)
        self.trail_history: Dict[int, Deque[Tuple[int, int]]] = {}
        self.active_ids: set[int] = set()

    def reset(self) -> None:
        """Reset internal tracking state and trajectory histories."""
        self.trail_history.clear()
        self.active_ids.clear()
        logger.info("Tracker state reset.")

    def update(self, frame: Union[np.ndarray, str, Path]) -> TrackingResult:
        """
        Process a new video frame, perform detection and ByteTrack association.

        Args:
            frame: Video frame as numpy ndarray (BGR).

        Returns:
            TrackingResult with updated track IDs and bounding boxes.
        """
        if frame is None:
            raise ValueError("Frame cannot be None.")

        if isinstance(frame, (str, Path)):
            frame = cv2.imread(str(frame))
            if frame is None:
                raise ValueError("Could not read frame from path.")

        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame: Must be non-empty numpy ndarray.")

        t0 = time.perf_counter()

        # Execute YOLO with ByteTrack persistence
        results = self.model.track(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            tracker=self.tracker_type,
            persist=True,
            device=self.device,
            verbose=False,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not results or len(results) == 0 or results[0].boxes is None:
            return self._empty_result(latency_ms)

        boxes_obj = results[0].boxes
        if len(boxes_obj) == 0:
            return self._empty_result(latency_ms)

        boxes_xyxy = boxes_obj.xyxy.cpu().numpy().astype(np.float32)
        scores = boxes_obj.conf.cpu().numpy().astype(np.float32)
        class_ids = boxes_obj.cls.cpu().numpy().astype(np.int32)
        class_names = [self.names.get(int(cid), f"class_{cid}") for cid in class_ids]

        # Extract track IDs (handle unassigned / newly detected tracks)
        if boxes_obj.id is not None:
            track_ids = boxes_obj.id.cpu().numpy().astype(np.int32)
        else:
            track_ids = np.full((len(boxes_xyxy),), -1, dtype=np.int32)

        # Update motion trajectory trails
        current_frame_ids = set()
        for box, tid in zip(boxes_xyxy, track_ids):
            if tid < 0:
                continue
            current_frame_ids.add(int(tid))
            cx = int((box[0] + box[2]) / 2.0)
            cy = int(box[3])  # Base of object on the road plane
            if tid not in self.trail_history:
                self.trail_history[tid] = deque(maxlen=self.max_trail_length)
            self.trail_history[tid].append((cx, cy))

        # Cleanup stale trajectories for objects that exited the scene long ago
        stale_ids = set(self.trail_history.keys()) - current_frame_ids
        for sid in list(stale_ids):
            if len(self.trail_history[sid]) > 0:
                self.trail_history[sid].popleft()
            if len(self.trail_history[sid]) == 0:
                del self.trail_history[sid]

        self.active_ids = current_frame_ids

        return TrackingResult(
            boxes=boxes_xyxy,
            scores=scores,
            class_ids=class_ids,
            class_names=class_names,
            track_ids=track_ids,
            latency_ms=latency_ms,
        )

    def _empty_result(self, latency_ms: float) -> TrackingResult:
        return TrackingResult(
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            class_ids=np.empty((0,), dtype=np.int32),
            class_names=[],
            track_ids=np.empty((0,), dtype=np.int32),
            latency_ms=latency_ms,
        )

    def draw_tracks(
        self,
        frame: np.ndarray,
        result: TrackingResult,
        draw_trails: bool = True,
        line_thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw bounding boxes with persistent IDs and trajectory trails.

        Args:
            frame: Video frame (BGR).
            result: TrackingResult from update().
            draw_trails: If True, renders historical trajectory lines.
            line_thickness: Box stroke width.

        Returns:
            Annotated frame copy.
        """
        canvas = frame.copy()

        # 1. Draw trajectory trails
        if draw_trails:
            for tid, points in self.trail_history.items():
                if len(points) < 2:
                    continue
                color = get_color_for_id(tid)
                for i in range(1, len(points)):
                    thickness = int(np.sqrt(float(i + 1))) + 1
                    cv2.line(canvas, points[i - 1], points[i], color, thickness)

        # 2. Draw current bounding boxes and badges
        if result.is_empty():
            return canvas

        for box, score, cid, cname, tid in zip(
            result.boxes, result.scores, result.class_ids, result.class_names, result.track_ids
        ):
            x1, y1, x2, y2 = [int(v) for v in box]
            color = get_color_for_id(int(tid)) if tid >= 0 else (128, 128, 128)

            # Draw bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_thickness)

            # Badge label: ID + Class + Conf
            id_str = f"ID:{tid}" if tid >= 0 else "NEW"
            label = f"#{id_str} {cname} {score:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            font_thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            badge_y1 = max(0, y1 - th - baseline - 4)
            badge_y2 = y1
            badge_x2 = min(canvas.shape[1], x1 + tw + 6)

            cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 3, badge_y2 - baseline),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        return canvas
