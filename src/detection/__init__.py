"""
Autonomous Driving Detection Package.
Provides pretrained YOLOv8 detector for BDD100K road scene objects.
"""
from src.detection.detector import CLASS_COLORS, DetectionResult, RoadObjectDetector

__all__ = ["RoadObjectDetector", "DetectionResult", "CLASS_COLORS"]
