"""
Autonomous Driving Lane Detection Package.
Provides:
1. Classical OpenCV lane detector (OpenCVLaneDetector) with HLS/HSV color thresholding & Hough lines.
2. Deep learning segmentation lane detector (SegmentationLaneDetector) using MobileNetV3-LRASPP.
"""
from src.lanes.lane_detector import LaneDetectionResult, OpenCVLaneDetector
from src.lanes.segmentation_lane_detector import SegmentationLaneDetector

__all__ = ["OpenCVLaneDetector", "SegmentationLaneDetector", "LaneDetectionResult"]
