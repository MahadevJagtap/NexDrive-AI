"""
Classical OpenCV-based Lane Detector for Autonomous Driving Perception.
Features HLS/HSV color thresholding, Canny edge detection, trapezoidal ROI masking,
Hough transform, slope filtering, temporal smoothing, and drivable corridor overlay.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LaneDetectionResult:
    """Container for lane detection outputs on a single frame."""
    left_line: Optional[Tuple[int, int, int, int]]   # (x1, y1, x2, y2)
    right_line: Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    lane_polygon: Optional[np.ndarray]               # Shape (N, 1, 2) corridor points
    lane_center_offset_px: Optional[float]           # Ego vehicle offset from lane center
    confidence: float                                # Detection confidence [0.0, 1.0]
    latency_ms: float                                # Processing latency in ms
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_lanes(self) -> bool:
        return self.left_line is not None and self.right_line is not None


class OpenCVLaneDetector:
    """
    Classical computer vision lane detector optimized for real-time CPU execution.
    Combines color segmentation, Canny edges, Hough lines, and temporal smoothing.
    """

    def __init__(
        self,
        roi_top_ratio: float = 0.60,
        roi_bottom_width_ratio: float = 0.95,
        roi_top_width_ratio: float = 0.20,
        smoothing_factor: float = 0.80,
        min_slope: float = 0.35,
        max_slope: float = 2.50,
    ):
        """
        Initialize lane detector parameters.

        Args:
            roi_top_ratio: Horizon height ratio (0.6 = start at 60% down the frame).
            roi_bottom_width_ratio: Trapezoid bottom width fraction.
            roi_top_width_ratio: Trapezoid top width fraction.
            smoothing_factor: Temporal exponential moving average alpha [0.0, 1.0].
            min_slope: Minimum absolute line slope to filter horizontal lines.
            max_slope: Maximum absolute line slope to filter vertical artifacts.
        """
        self.roi_top_ratio = roi_top_ratio
        self.roi_bottom_width_ratio = roi_bottom_width_ratio
        self.roi_top_width_ratio = roi_top_width_ratio
        self.smoothing_factor = smoothing_factor
        self.min_slope = min_slope
        self.max_slope = max_slope

        # Historical smoothed lines: (x1, y1, x2, y2)
        self.prev_left: Optional[Tuple[float, float, float, float]] = None
        self.prev_right: Optional[Tuple[float, float, float, float]] = None
        self.consecutive_misses = 0
        self.max_misses_before_reset = 15

    def reset(self) -> None:
        """Reset temporal state."""
        self.prev_left = None
        self.prev_right = None
        self.consecutive_misses = 0

    def detect(self, frame: np.ndarray) -> LaneDetectionResult:
        """
        Perform classical lane line detection on a frame.

        Args:
            frame: OpenCV BGR image.

        Returns:
            LaneDetectionResult containing fitted lines, drivable area polygon, and offset.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame for lane detection.")

        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # 1. Color Masking (White & Yellow road markings)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        # White mask: high lightness
        white_mask = cv2.inRange(hls, np.array([0, 190, 0]), np.array([180, 255, 255]))

        # Yellow mask: yellow hue with adequate saturation
        yellow_mask = cv2.inRange(hls, np.array([15, 30, 100]), np.array([38, 204, 255]))
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # 2. Grayscale & Canny Edge Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Combine color edges with structural edges
        combined_edges = cv2.bitwise_or(edges, color_mask)

        # 3. Region of Interest (Trapezoidal Mask)
        y_top = int(h * self.roi_top_ratio)
        x_center = w // 2
        top_half_w = int(w * self.roi_top_width_ratio / 2.0)
        bottom_half_w = int(w * self.roi_bottom_width_ratio / 2.0)

        roi_vertices = np.array([
            [
                (x_center - bottom_half_w, h),
                (x_center - top_half_w, y_top),
                (x_center + top_half_w, y_top),
                (x_center + bottom_half_w, h),
            ]
        ], dtype=np.int32)

        mask = np.zeros_like(combined_edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(combined_edges, mask)

        # 4. Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=25,
            minLineLength=30,
            maxLineGap=20,
        )

        left_candidates = []
        right_candidates = []

        if lines is not None:
            for line in lines:
                coords = line.reshape(-1)
                if len(coords) < 4:
                    continue
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                if x2 == x1:
                    continue  # skip strictly vertical lines
                slope = (y2 - y1) / float(x2 - x1)

                if abs(slope) < self.min_slope or abs(slope) > self.max_slope:
                    continue

                # OpenCV image origin is top-left:
                # Left lane slopes down-right (negative slope in math, negative (y2-y1)/(x2-x1))
                if slope < 0 and x1 < x_center and x2 < x_center:
                    left_candidates.append((x1, y1, x2, y2, slope))
                elif slope > 0 and x1 > x_center and x2 > x_center:
                    right_candidates.append((x1, y1, x2, y2, slope))

        # 5. Fit & Extrapolate Left and Right Lines
        raw_left = self._fit_line(left_candidates, y_bottom=h, y_top=y_top)
        raw_right = self._fit_line(right_candidates, y_bottom=h, y_top=y_top)

        # 6. Temporal Smoothing & Miss Handling
        left_line, right_line, confidence = self._temporal_smooth(raw_left, raw_right)

        # 7. Drivable Corridor Polygon & Lane Center Offset
        polygon = None
        lane_offset = None

        if left_line is not None and right_line is not None:
            lx1, ly1, lx2, ly2 = left_line
            rx1, ry1, rx2, ry2 = right_line

            # Polygon: bottom-left -> top-left -> top-right -> bottom-right
            pts = np.array([
                [lx1, ly1],
                [lx2, ly2],
                [rx2, ry2],
                [rx1, ry1],
            ], dtype=np.int32).reshape((-1, 1, 2))
            polygon = pts

            # Calculate vehicle lane center offset (assuming camera mounted at image center)
            lane_bottom_center = (lx1 + rx1) / 2.0
            image_center = w / 2.0
            lane_offset = image_center - lane_bottom_center

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return LaneDetectionResult(
            left_line=left_line,
            right_line=right_line,
            lane_polygon=polygon,
            lane_center_offset_px=lane_offset,
            confidence=confidence,
            latency_ms=latency_ms,
        )

    def _fit_line(
        self,
        candidates: List[Tuple[int, int, int, int, float]],
        y_bottom: int,
        y_top: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Fit single representative line using linear regression on candidate points."""
        if not candidates:
            return None

        x_coords = []
        y_coords = []
        weights = []

        for x1, y1, x2, y2, _ in candidates:
            length = np.hypot(x2 - x1, y2 - y1)
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
            weights.extend([length, length])

        if len(x_coords) < 2:
            return None

        # Polynomial fit x = f(y) (degree 1) to avoid divide-by-zero on steep slopes
        poly = np.polyfit(y_coords, x_coords, 1, w=weights)
        fit_fn = np.poly1d(poly)

        x_bottom = int(fit_fn(y_bottom))
        x_top = int(fit_fn(y_top))

        return (x_bottom, y_bottom, x_top, y_top)

    def _temporal_smooth(
        self,
        raw_left: Optional[Tuple[int, int, int, int]],
        raw_right: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]], float]:
        """Smooth fitted lines using Exponential Moving Average across frames."""
        alpha = self.smoothing_factor

        # Left line smoothing
        if raw_left is not None:
            if self.prev_left is None:
                curr_left = tuple(float(v) for v in raw_left)
            else:
                curr_left = tuple(
                    alpha * prev + (1.0 - alpha) * float(raw)
                    for prev, raw in zip(self.prev_left, raw_left)
                )
            self.prev_left = curr_left
        else:
            curr_left = self.prev_left

        # Right line smoothing
        if raw_right is not None:
            if self.prev_right is None:
                curr_right = tuple(float(v) for v in raw_right)
            else:
                curr_right = tuple(
                    alpha * prev + (1.0 - alpha) * float(raw)
                    for prev, raw in zip(self.prev_right, raw_right)
                )
            self.prev_right = curr_right
        else:
            curr_right = self.prev_right

        # Assess confidence
        if raw_left is not None and raw_right is not None:
            self.consecutive_misses = 0
            conf = 0.95
        elif raw_left is not None or raw_right is not None:
            self.consecutive_misses = max(0, self.consecutive_misses)
            conf = 0.65
        else:
            self.consecutive_misses += 1
            conf = max(0.0, 0.50 - (self.consecutive_misses * 0.05))

        if self.consecutive_misses > self.max_misses_before_reset:
            self.prev_left = None
            self.prev_right = None
            return None, None, 0.0

        out_left = tuple(int(round(v)) for v in curr_left) if curr_left else None
        out_right = tuple(int(round(v)) for v in curr_right) if curr_right else None

        return out_left, out_right, conf

    def draw_lanes(
        self,
        frame: np.ndarray,
        result: LaneDetectionResult,
        corridor_color: Tuple[int, int, int] = (0, 220, 100),
        corridor_alpha: float = 0.25,
        line_thickness: int = 4,
    ) -> np.ndarray:
        """
        Render detected lane markings and highlighted drivable corridor onto frame.

        Args:
            frame: Video frame (BGR).
            result: LaneDetectionResult from detect().
            corridor_color: BGR fill color for drivable lane area.
            corridor_alpha: Translucency alpha for corridor overlay.
            line_thickness: Stroke thickness for boundary lines.

        Returns:
            Annotated frame copy.
        """
        canvas = frame.copy()

        # 1. Draw translucent drivable corridor
        if result.lane_polygon is not None:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [result.lane_polygon], corridor_color)
            canvas = cv2.addWeighted(overlay, corridor_alpha, canvas, 1.0 - corridor_alpha, 0)

        # 2. Draw Left Lane Boundary (Cyan/Blue)
        if result.left_line is not None:
            lx1, ly1, lx2, ly2 = result.left_line
            cv2.line(canvas, (lx1, ly1), (lx2, ly2), (255, 200, 0), line_thickness, cv2.LINE_AA)

        # 3. Draw Right Lane Boundary (Red/Magenta)
        if result.right_line is not None:
            rx1, ry1, rx2, ry2 = result.right_line
            cv2.line(canvas, (rx1, ry1), (rx2, ry2), (0, 100, 255), line_thickness, cv2.LINE_AA)

        # 4. Draw Departure & Offset Telemetry
        if result.lane_center_offset_px is not None:
            offset = result.lane_center_offset_px
            direction = "CENTERED" if abs(offset) < 20 else ("RIGHT" if offset > 0 else "LEFT")
            hud_text = f"Lane: {direction} ({abs(offset):.1f}px offset) | Conf: {result.confidence*100:.0f}%"
            cv2.putText(canvas, hud_text, (20, canvas.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return canvas
