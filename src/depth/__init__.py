"""
Autonomous Driving Depth Estimation Package.
Provides MiDaS monocular relative depth estimation with explicit non-metric calibration labeling.
"""
from src.depth.depth_estimator import DepthResult, MonocularDepthEstimator

__all__ = ["MonocularDepthEstimator", "DepthResult"]
