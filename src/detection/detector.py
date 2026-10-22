"""
Road Object Detector Module using fine-tuned BDD100K YOLOv8.
Supports CPU inference, custom confidence thresholds, robust validation, and annotation.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Structured container for object detection outputs on a single frame."""
    boxes: np.ndarray  # Shape: (N, 4) in [x1, y1, x2, y2] format
    scores: np.ndarray  # Shape: (N,) confidence scores [0.0, 1.0]
    class_ids: np.ndarray  # Shape: (N,) integer class IDs
    class_names: List[str]  # List of string class names corresponding to class_ids
    inference_time_ms: float  # Model forward inference time in milliseconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Returns the number of detected objects."""
        return len(self.boxes)

    def is_empty(self) -> bool:
        """Returns True if no objects were detected."""
        return self.count == 0


# BDD100K 10-class color palette (BGR format for OpenCV)
CLASS_COLORS = {
    0: (230, 25, 75),     # person: red
    1: (60, 180, 75),     # rider: green
    2: (0, 130, 200),     # car: blue
    3: (245, 130, 48),    # truck: orange
    4: (145, 30, 180),    # bus: purple
    5: (70, 240, 240),    # train: cyan
    6: (240, 50, 230),    # motor: magenta
    7: (210, 245, 60),    # bike: lime
    8: (250, 190, 212),   # traffic light: pink
    9: (0, 128, 128),     # traffic sign: teal
}


class RoadObjectDetector:
    """
    Modular detector wrapper for road agents (vehicles, pedestrians, signs, lights)
    using YOLOv8 trained on BDD100K.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/best.pt",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
    ):
        """
        Initialize detector with local checkpoint.

        Args:
            model_path: Path to local YOLOv8 weights (.pt).
            conf_thresh: Detection confidence threshold (0.0 to 1.0).
            iou_thresh: Non-maximum suppression IoU threshold.
            imgsz: Input image dimension for inference.
            device: Inference device ('cpu' recommended for Intel Iris Xe / non-NVIDIA).
        """
        self.model_path = Path(model_path)
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.imgsz = int(imgsz)
        self.device = device.lower()

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {self.model_path.resolve()}\n"
                f"Please run 'python scripts/download_model.py' to download the BDD100K checkpoint."
            )

        logger.info(f"Loading YOLOv8 weights from {self.model_path} on {self.device}...")
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self.names: Dict[int, str] = self.model.names
            logger.info(f"Model loaded successfully. Available classes ({len(self.names)}): {self.names}")
        except Exception as exc:
            logger.error(f"Failed to load YOLO model: {exc}")
            raise RuntimeError(f"Failed initializing YOLO detector from {self.model_path}: {exc}") from exc

    def detect(self, frame: Union[np.ndarray, str, Path]) -> DetectionResult:
        """
        Run inference on a single image or video frame.

        Args:
            frame: Input image as numpy array (BGR/RGB) or file path.

        Returns:
            DetectionResult container with bounding boxes, scores, and class labels.
        """
        # Validate inputs
        if frame is None:
            raise ValueError("Input frame cannot be None.")

        if isinstance(frame, (str, Path)):
            frame_path = Path(frame)
            if not frame_path.exists():
                raise FileNotFoundError(f"Input image path does not exist: {frame_path}")
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"Failed reading image from path: {frame_path}")

        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame: Must be non-empty numpy ndarray.")

        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not results or len(results) == 0 or results[0].boxes is None:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_names=[],
                inference_time_ms=latency_ms,
            )

        boxes_obj = results[0].boxes
        if len(boxes_obj) == 0:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_names=[],
                inference_time_ms=latency_ms,
            )

        boxes_xyxy = boxes_obj.xyxy.cpu().numpy().astype(np.float32)
        scores = boxes_obj.conf.cpu().numpy().astype(np.float32)
        class_ids = boxes_obj.cls.cpu().numpy().astype(np.int32)
        class_names = [self.names.get(int(cid), f"class_{cid}") for cid in class_ids]

        return DetectionResult(
            boxes=boxes_xyxy,
            scores=scores,
            class_ids=class_ids,
            class_names=class_names,
            inference_time_ms=latency_ms,
        )

    def draw_detections(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
        line_thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw clean bounding boxes and class badges onto the frame.

        Args:
            frame: OpenCV image in BGR format.
            result: DetectionResult from detect().
            color_map: Optional custom BGR color mapping by class ID.
            line_thickness: Bounding box line thickness.

        Returns:
            Annotated image copy.
        """
        canvas = frame.copy()
        if result.is_empty():
            return canvas

        colors = color_map or CLASS_COLORS

        for box, score, cid, cname in zip(
            result.boxes, result.scores, result.class_ids, result.class_names
        ):
            x1, y1, x2, y2 = [int(v) for v in box]
            color = colors.get(int(cid), (0, 255, 0))

            # Draw rectangle
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_thickness)

            # Badge text
            label = f"{cname} {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Ensure badge is within frame
            badge_y1 = max(0, y1 - th - baseline - 4)
            badge_y2 = y1
            badge_x2 = min(canvas.shape[1], x1 + tw + 6)

            # Draw badge background
            cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
            # Text on badge (white or black depending on color brightness)
            text_color = (255, 255, 255)
            cv2.putText(
                canvas,
                label,
                (x1 + 3, badge_y2 - baseline),
                font,
                font_scale,
                text_color,
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        return canvas
