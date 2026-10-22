"""
Integrated Perception Pipeline Orchestrator.
Fuses Object Detection, ByteTrack Multi-Object Tracking, OpenCV Lane Detection,
and optional Monocular Depth Estimation into a single high-performance pipeline.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import psutil

from src.depth.depth_estimator import DepthResult, MonocularDepthEstimator
from src.detection.detector import CLASS_COLORS, DetectionResult, RoadObjectDetector
from src.lanes.lane_detector import LaneDetectionResult, OpenCVLaneDetector
from src.tracking.tracker import RoadObjectTracker, TrackingResult, get_color_for_id

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration options for Perception Pipeline."""
    model_path: str = "models/best.pt"
    enable_tracking: bool = True
    enable_lanes: bool = True
    lane_method: str = "classical"  # 'classical' (OpenCV) or 'segmentation' (LRASPP)
    enable_depth: bool = False
    depth_frequency: int = 5         # Compute depth every N frames to conserve CPU
    conf_thresh: float = 0.25
    iou_thresh: float = 0.45
    imgsz: int = 640
    device: str = "cpu"
    draw_trails: bool = True
    draw_hud: bool = True
    enable_hazard_assessment: bool = True
    hazard_roi_top: float = 0.55
    hazard_roi_bottom: float = 0.95
    hazard_roi_width_ratio: float = 0.35


@dataclass
class PipelineFrameResult:
    """Telemetry and processed visual outputs for a single frame."""
    frame_index: int
    annotated_frame: np.ndarray
    detections: Optional[DetectionResult] = None
    tracks: Optional[TrackingResult] = None
    lanes: Optional[LaneDetectionResult] = None
    depth: Optional[DepthResult] = None
    total_latency_ms: float = 0.0
    effective_fps: float = 0.0
    hazard_status: str = "CLEAR"
    telemetry: Dict[str, Any] = field(default_factory=dict)


class PerceptionPipeline:
    """
    End-to-End Autonomous Driving Perception Orchestrator.
    Handles frame acquisition, module execution, telemetry collection, and HUD rendering.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        logger.info(f"Initializing PerceptionPipeline with config: {self.config}")

        # 1. Detection / Tracking: If tracking is enabled, RoadObjectTracker handles both
        # YOLO inference and ByteTrack association in one single forward pass (no duplicate inference).
        if self.config.enable_tracking:
            self.tracker = RoadObjectTracker(
                model_path=self.config.model_path,
                conf_thresh=self.config.conf_thresh,
                iou_thresh=self.config.iou_thresh,
                imgsz=self.config.imgsz,
                device=self.config.device,
            )
            self.detector = None
        else:
            self.tracker = None
            self.detector = RoadObjectDetector(
                model_path=self.config.model_path,
                conf_thresh=self.config.conf_thresh,
                iou_thresh=self.config.iou_thresh,
                imgsz=self.config.imgsz,
                device=self.config.device,
            )

        # 2. Lane Detector
        if self.config.enable_lanes:
            if getattr(self.config, "lane_method", "classical") == "segmentation":
                from src.lanes.segmentation_lane_detector import SegmentationLaneDetector
                self.lane_detector = SegmentationLaneDetector(device=self.config.device)
            else:
                self.lane_detector = OpenCVLaneDetector()
        else:
            self.lane_detector = None

        # 3. Depth Estimator (optional)
        self.depth_estimator: Optional[MonocularDepthEstimator] = None
        if self.config.enable_depth:
            self.depth_estimator = MonocularDepthEstimator(
                model_type="MiDaS_small",
                device=self.config.device,
            )

        self.last_depth_result: Optional[DepthResult] = None
        self.frame_counter = 0

    def reset(self) -> None:
        """Reset internal pipeline tracking and temporal states."""
        self.frame_counter = 0
        self.last_depth_result = None
        if self.tracker:
            self.tracker.reset()
        if self.lane_detector and hasattr(self.lane_detector, "reset"):
            self.lane_detector.reset()

    def process_frame(self, frame: np.ndarray, frame_index: Optional[int] = None) -> PipelineFrameResult:
        """
        Execute full perception stack on a single video frame.

        Args:
            frame: OpenCV BGR image (numpy ndarray).
            frame_index: Optional sequential frame number.

        Returns:
            PipelineFrameResult with rendered image and complete telemetry.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame passed to process_frame().")

        if frame_index is None:
            self.frame_counter += 1
            idx = self.frame_counter
        else:
            idx = frame_index

        t_start = time.perf_counter()
        telemetry: Dict[str, Any] = {"frame": idx}

        # 1. Lane Detection (independent of object detection)
        lane_res: Optional[LaneDetectionResult] = None
        if self.lane_detector:
            lane_res = self.lane_detector.detect(frame)
            telemetry["lane_latency_ms"] = round(lane_res.latency_ms, 2)
            telemetry["lane_offset_px"] = round(lane_res.lane_center_offset_px, 1) if lane_res.lane_center_offset_px is not None else None
            telemetry["lane_conf"] = round(lane_res.confidence, 2)

        # 2. Tracking or Detection (single forward pass)
        track_res: Optional[TrackingResult] = None
        det_res: Optional[DetectionResult] = None

        if self.tracker:
            track_res = self.tracker.update(frame)
            telemetry["perception_latency_ms"] = round(track_res.latency_ms, 2)
            telemetry["object_count"] = track_res.count
            telemetry["active_tracks"] = len(self.tracker.active_ids)
        elif self.detector:
            det_res = self.detector.detect(frame)
            telemetry["perception_latency_ms"] = round(det_res.inference_time_ms, 2)
            telemetry["object_count"] = det_res.count

        # 3. Optional Depth Estimation
        depth_res: Optional[DepthResult] = None
        if self.depth_estimator:
            # Check depth frequency
            if idx % self.config.depth_frequency == 0 or self.last_depth_result is None:
                depth_res = self.depth_estimator.estimate(frame)
                self.last_depth_result = depth_res
                telemetry["depth_latency_ms"] = round(depth_res.latency_ms, 2)
            else:
                depth_res = self.last_depth_result
                telemetry["depth_latency_ms"] = 0.0

        # 4. Unified Scene Rendering
        annotated = frame.copy()

        # Step 4a: Render lanes
        if lane_res and self.lane_detector:
            annotated = self.lane_detector.draw_lanes(annotated, lane_res)

        # Step 4b: Render tracked objects
        if track_res and self.tracker:
            annotated = self.tracker.draw_tracks(
                annotated,
                track_res,
                draw_trails=self.config.draw_trails,
            )
            # Annotate relative proximity if depth is active
            if depth_res and self.depth_estimator:
                for box, tid in zip(track_res.boxes, track_res.track_ids):
                    if tid >= 0:
                        prox = self.depth_estimator.get_object_proximity(depth_res, tuple(box))
                        x1, y1 = int(box[0]), int(box[1])
                        cv2.putText(
                            annotated,
                            f"Prox: {prox:.2f}",
                            (x1, max(35, y1 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

        elif det_res and self.detector:
            annotated = self.detector.draw_detections(annotated, det_res)

        # Total Frame Time
        total_latency_ms = (time.perf_counter() - t_start) * 1000.0
        effective_fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0.0

        telemetry["total_latency_ms"] = round(total_latency_ms, 2)
        telemetry["fps"] = round(effective_fps, 1)

        # Step 4c: Obstacle Hazard Assessment (Demonstration Rule)
        hazard_status = "STATUS: CLEAR"
        hazard_color = (0, 220, 100)
        hazard_roi = None
        if self.config.enable_hazard_assessment:
            hazard_status, hazard_color, hazard_roi = self._assess_hazard(
                frame.shape,
                track_res or det_res,
            )
            telemetry["hazard_status"] = hazard_status

            # Draw subtle safety corridor boundary on road
            if hazard_roi is not None:
                rx1, ry1, rx2, ry2 = hazard_roi
                cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 165, 255), 1, cv2.LINE_AA)

        # Step 4d: Render HUD Telemetry Banner
        if self.config.draw_hud:
            annotated = self._render_hud(annotated, telemetry, hazard_status, hazard_color)

        return PipelineFrameResult(
            frame_index=idx,
            annotated_frame=annotated,
            detections=det_res,
            tracks=track_res,
            lanes=lane_res,
            depth=depth_res,
            total_latency_ms=total_latency_ms,
            effective_fps=effective_fps,
            hazard_status=hazard_status,
            telemetry=telemetry,
        )

    def _assess_hazard(
        self,
        frame_shape: Tuple[int, ...],
        objects: Optional[Union[TrackingResult, DetectionResult]],
    ) -> Tuple[str, Tuple[int, int, int], Tuple[int, int, int, int]]:
        """
        Rule-based demonstration obstacle-response indicator.
        
        Evaluates tracked obstacles overlapping the forward safety corridor.
        DISCLAIMER: This is a demonstration heuristic rule, NOT safety-certified vehicle control.
        Uncalibrated relative depth is never interpreted as metric distance in meters.
        """
        h, w = frame_shape[:2]
        y_min = int(h * self.config.hazard_roi_top)
        y_max = int(h * self.config.hazard_roi_bottom)
        x_min = int(w * (0.5 - self.config.hazard_roi_width_ratio / 2.0))
        x_max = int(w * (0.5 + self.config.hazard_roi_width_ratio / 2.0))
        roi = (x_min, y_min, x_max, y_max)

        if objects is None or len(objects.boxes) == 0:
            return "STATUS: CLEAR", (0, 220, 100), roi

        hazard_level = "STATUS: CLEAR"
        hazard_color = (0, 220, 100)
        obstacle_classes = {"car", "truck", "bus", "person", "rider", "motor", "bike"}
        mid_y = y_min + (y_max - y_min) * 0.45

        for box, name in zip(objects.boxes, objects.class_names):
            if name.lower() not in obstacle_classes:
                continue

            bx1, by1, bx2, by2 = box
            box_cx = (bx1 + bx2) / 2.0
            box_bottom = by2

            if x_min <= box_cx <= x_max and y_min <= box_bottom <= y_max:
                if box_bottom >= mid_y:
                    return "HAZARD: OBSTACLE IN PATH", (0, 0, 255), roi
                else:
                    hazard_level = "SLOW DOWN: OBSTACLE AHEAD"
                    hazard_color = (0, 165, 255)

        return hazard_level, hazard_color, roi

    def _render_hud(
        self,
        canvas: np.ndarray,
        telemetry: Dict[str, Any],
        hazard_status: str = "STATUS: CLEAR",
        hazard_color: Tuple[int, int, int] = (0, 220, 100),
    ) -> np.ndarray:
        """Draw an advanced HUD telemetry dashboard overlay."""
        h, w = canvas.shape[:2]
        hud_h = 42

        # Translucent top banner
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
        canvas = cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0)

        # HUD Text components
        fps = telemetry.get("fps", 0.0)
        latency = telemetry.get("total_latency_ms", 0.0)
        tracks = telemetry.get("active_tracks", telemetry.get("object_count", 0))
        cpu_pct = psutil.cpu_percent()

        hud_str = (
            f"PERCEPTION STACK | "
            f"FPS: {fps:.1f} | "
            f"Latency: {latency:.1f}ms | "
            f"Tracks: {tracks} | "
            f"CPU: {cpu_pct:.0f}%"
        )

        cv2.putText(
            canvas,
            hud_str,
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )

        # Status pills for active modules
        pill_x = w - 380
        modules = [
            ("DET", True),
            ("TRK", self.config.enable_tracking),
            ("LANE", self.config.enable_lanes),
            ("DEPTH", self.config.enable_depth),
        ]

        for mod_name, active in modules:
            bg_col = (40, 160, 40) if active else (80, 80, 80)
            cv2.rectangle(canvas, (pill_x, 8), (pill_x + 65, 34), bg_col, -1)
            cv2.putText(canvas, mod_name, (pill_x + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            pill_x += 75

        # Render Hazard Badge below top banner if enabled
        if self.config.enable_hazard_assessment:
            badge_x = 18
            badge_y = 48
            badge_w = 330
            badge_h = 28
            badge_overlay = canvas.copy()
            cv2.rectangle(badge_overlay, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (20, 20, 20), -1)
            canvas = cv2.addWeighted(badge_overlay, 0.75, canvas, 0.25, 0)
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), hazard_color, 1)

            cv2.putText(
                canvas,
                hazard_status,
                (badge_x + 10, badge_y + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                hazard_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "[DEMO RULE - NOT VEHICLE CONTROL]",
                (badge_x + badge_w + 10, badge_y + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

        return canvas

    def process_video(
        self,
        input_source: Union[str, Path, int],
        output_path: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        callback: Optional[Callable[[PipelineFrameResult], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline over an entire video file or webcam stream.

        Args:
            input_source: File path or integer index (e.g., 0 for webcam).
            output_path: Destination path for annotated MP4 video.
            max_frames: Max frames to process (None for entire video).
            callback: Optional per-frame hook.

        Returns:
            Dictionary containing overall statistics and saved telemetry.
        """
        self.reset()
        source_str = str(input_source) if not isinstance(input_source, int) else input_source
        cap = cv2.VideoCapture(source_str)

        if not cap.isOpened():
            raise RuntimeError(f"Failed opening video source: {input_source}")

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_in_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            for codec in ("avc1", "H264", "mp4v"):
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    w = cv2.VideoWriter(str(out_p), fourcc, fps_in, (width, height))
                    if w.isOpened():
                        writer = w
                        break
                    w.release()
                except Exception:
                    pass
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_p), fourcc, fps_in, (width, height))

        frame_count = 0
        latencies: List[float] = []
        telemetry_log: List[Dict[str, Any]] = []
        t0_all = time.perf_counter()

        logger.info(f"Processing video: {input_source} ({width}x{height} @ {fps_in:.1f} FPS)...")

        try:
            while True:
                if max_frames and frame_count >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                res = self.process_frame(frame, frame_index=frame_count)
                latencies.append(res.total_latency_ms)
                telemetry_log.append(res.telemetry)

                if writer:
                    writer.write(res.annotated_frame)

                if callback:
                    callback(res)

        finally:
            cap.release()
            if writer:
                writer.release()

        total_wall_time = time.perf_counter() - t0_all
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        overall_fps = frame_count / total_wall_time if total_wall_time > 0 else 0.0

        summary = {
            "input_source": str(input_source),
            "output_path": str(output_path) if output_path else None,
            "processed_frames": frame_count,
            "total_video_frames": total_in_frames,
            "wall_clock_seconds": round(total_wall_time, 2),
            "average_latency_ms": round(avg_latency, 2),
            "effective_fps": round(overall_fps, 2),
            "active_modules": {
                "detection": True,
                "tracking": self.config.enable_tracking,
                "lanes": self.config.enable_lanes,
                "depth": self.config.enable_depth,
            },
        }

        # Save telemetry log to JSON if results folder exists
        results_dir = Path("D:/auto_drive/results")
        if results_dir.exists():
            log_path = results_dir / "pipeline_telemetry.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "frames": telemetry_log}, f, indent=2)
            summary["telemetry_log"] = str(log_path)

        logger.info(f"Pipeline finished: {frame_count} frames processed in {total_wall_time:.2f}s ({overall_fps:.1f} FPS)")
        return summary
