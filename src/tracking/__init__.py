"""
Autonomous Driving Tracking Package.
Provides ByteTrack-based persistent multi-object tracking.
"""
from src.tracking.tracker import RoadObjectTracker, TrackingResult, get_color_for_id

__all__ = ["RoadObjectTracker", "TrackingResult", "get_color_for_id"]
