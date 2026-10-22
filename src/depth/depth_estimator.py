"""
Monocular Depth Estimation module for Autonomous Driving Perception.
Uses pretrained MiDaS (MiDaS_small) for real-time CPU relative depth estimation.
Outputs are clearly labeled as relative (uncalibrated) depth to prevent misinterpreting
as metric distances.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    """Container for relative monocular depth estimation outputs."""
    raw_depth: np.ndarray          # Inverse relative depth map (H, W), float32
    normalized_depth: np.ndarray   # Normalized [0.0, 1.0] relative proximity (1.0 = closest)
    colored_map: np.ndarray        # BGR colorized depth visualization (H, W, 3)
    latency_ms: float              # Inference latency in ms
    is_calibrated_metric: bool = False  # Explicitly False: uncalibrated relative depth
    label: str = "Relative Depth (Uncalibrated)"
    metadata: Dict[str, Any] = field(default_factory=dict)


class MonocularDepthEstimator:
    """
    Lightweight monocular depth estimator based on MiDaS_small.
    Optimized for CPU inference with optional frame downsampling and object association.
    """

    def __init__(
        self,
        model_type: str = "MiDaS_small",
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize MiDaS model and preprocessing transform.
        """
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        # Configure PyTorch Hub cache
        if cache_dir:
            hub_dir = Path(cache_dir).resolve()
        else:
            PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
            hub_dir = (PROJECT_ROOT / "models" / "cache" / "hub").resolve()

        hub_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(hub_dir))

        logger.info(f"Loading {model_type} from torch.hub on {self.device} (cache: {hub_dir})...")
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            self.model.to(self.device)
            self.model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            logger.info("MiDaS Depth Estimator successfully initialized.")
        except Exception as exc:
            logger.error(f"Failed to load MiDaS model: {exc}")
            raise RuntimeError(f"Could not initialize MiDaS depth estimator: {exc}") from exc

    def estimate(self, frame: np.ndarray) -> DepthResult:
        """
        Predict relative depth map for a single BGR frame.

        Args:
            frame: OpenCV BGR image (numpy ndarray).

        Returns:
            DepthResult container with normalized map and color visualization.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame for depth estimation.")

        t0 = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            # Resize to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth_np = prediction.cpu().numpy()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # MiDaS outputs inverse relative depth (higher value = closer)
        # Normalize to [0.0, 1.0]
        d_min = float(depth_np.min())
        d_max = float(depth_np.max())
        if d_max > d_min:
            norm_depth = (depth_np - d_min) / (d_max - d_min)
        else:
            norm_depth = np.zeros_like(depth_np, dtype=np.float32)

        # Generate colorized colormap (INFERNO: bright yellow = close, dark purple = far)
        depth_u8 = (norm_depth * 255).astype(np.uint8)
        colored_map = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)

        return DepthResult(
            raw_depth=depth_np,
            normalized_depth=norm_depth,
            colored_map=colored_map,
            latency_ms=latency_ms,
            is_calibrated_metric=False,
            label="Relative Proximity (Uncalibrated)",
        )

    def get_object_proximity(
        self,
        depth_result: DepthResult,
        box: Tuple[int, int, int, int],
    ) -> float:
        """
        Calculate relative proximity score for an object's bounding box.

        Args:
            depth_result: Output from estimate().
            box: (x1, y1, x2, y2) bounding box.

        Returns:
            Float proximity score in [0.0, 1.0] (1.0 = nearest, 0.0 = distant).
        """
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = depth_result.normalized_depth.shape

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Sample lower 50% of the bounding box to capture the object base on the road
        y_mid = (y1 + y2) // 2
        patch = depth_result.normalized_depth[y_mid:y2, x1:x2]

        if patch.size == 0:
            return 0.0

        return float(np.median(patch))

    def create_side_by_side(
        self,
        frame: np.ndarray,
        depth_result: DepthResult,
        banner_text: str = "Relative Depth (Uncalibrated - Not Meters)",
    ) -> np.ndarray:
        """
        Create a side-by-side composite of original driving frame and depth map.
        """
        h, w = frame.shape[:2]
        colored = depth_result.colored_map

        # Add watermark / label banner on depth map
        depth_view = colored.copy()
        cv2.putText(
            depth_view,
            banner_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        composite = np.hstack([frame, depth_view])
        return composite
