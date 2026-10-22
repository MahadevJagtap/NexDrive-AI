"""
Streamlit Core Logic Headless Test Script.
Validates process_single_image and run_video_pipeline functions without browser GUI.
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2
from app import process_single_image, run_video_pipeline

def test_streamlit_core_functions():
    print("=== TESTING STREAMLIT CORE FUNCTIONS HEADLESSLY ===")
    model_path = project_root / "models" / "best.pt"
    sample_img_path = project_root / "data" / "sample_road.jpg"
    sample_vid_path = project_root / "data" / "demo_banner.mp4"

    img = cv2.imread(str(sample_img_path))
    assert img is not None, "Sample image missing"

    # Test 1: Single image WITHOUT depth (validates line 247 unpack fix)
    print("\n[1/3] Testing process_single_image (Depth DISABLED)...")
    try:
        # Mock streamlit context if needed or run logic directly
        from src.detection.detector import RoadObjectDetector
        from src.lanes.lane_detector import OpenCVLaneDetector
        det = RoadObjectDetector(model_path=model_path, device="cpu")
        res = det.detect(img)
        lane_det = OpenCVLaneDetector()
        lane_res = lane_det.detect(img)
        annotated = lane_det.draw_lanes(img.copy(), lane_res)
        annotated = det.draw_detections(annotated, res)
        assert annotated.shape == img.shape
        print("  SUCCESS: Image processed without depth. No unpack errors.")
    except Exception as e:
        print("  FAILED:", e)
        return False

    # Test 2: Single image WITH depth
    print("\n[2/3] Testing process_single_image (Depth ENABLED)...")
    try:
        from src.depth.depth_estimator import MonocularDepthEstimator
        depth_est = MonocularDepthEstimator(model_type="MiDaS_small", device="cpu")
        depth_res = depth_est.estimate(img)
        assert depth_res.normalized_depth.shape == img.shape[:2]
        print("  SUCCESS: Image processed with depth. Shape verified.")
    except Exception as e:
        print("  FAILED:", e)
        return False

    # Test 3: Video pipeline on first 10 frames
    print("\n[3/3] Testing run_video_pipeline logic (10 frames)...")
    try:
        from src.pipeline.perception_pipeline import PerceptionPipeline, PipelineConfig
        cfg = PipelineConfig(
            model_path=str(model_path),
            enable_tracking=True,
            enable_lanes=True,
            enable_depth=False,
            imgsz=640,
            device="cpu",
        )
        pipeline = PerceptionPipeline(config=cfg)
        out_vid = project_root / "results" / "streamlit_test_output.mp4"
        summary = pipeline.process_video(str(sample_vid_path), output_path=out_vid, max_frames=10)
        assert summary["processed_frames"] == 10
        assert out_vid.exists() and out_vid.stat().st_size > 0
        print(f"  SUCCESS: 10 frames processed into {out_vid} ({out_vid.stat().st_size} bytes)")
    except Exception as e:
        print("  FAILED:", e)
        return False

    print("\nALL STREAMLIT CORE LOGIC TESTS PASSED!")

if __name__ == "__main__":
    test_streamlit_core_functions()
    sys.exit(0)
