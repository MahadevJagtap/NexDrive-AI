"""
CLI Entry Point for Autonomous Driving Perception Pipeline.
Processes input video or camera stream, runs full perception stack,
and produces annotated video and telemetry diagnostics.
"""
import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.perception_pipeline import PerceptionPipeline, PipelineConfig

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Driving Perception Stack CLI Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/demo_banner.mp4",
        help="Input video file path or camera index (e.g. 0)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/pipeline_output.mp4",
        help="Destination path for annotated video",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="models/best.pt",
        help="Path to YOLOv8 detection weights (.pt)",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable ByteTrack multi-object tracking",
    )
    parser.add_argument(
        "--no-lanes",
        action="store_true",
        help="Disable OpenCV lane detection",
    )
    parser.add_argument(
        "--enable-depth",
        action="store_true",
        help="Enable MiDaS monocular relative depth estimation",
    )
    parser.add_argument(
        "--depth-freq",
        type=int,
        default=5,
        help="Compute depth every N frames when depth is enabled",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Object detection confidence threshold",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=90,
        help="Maximum frames to process (0 for full video)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device ('cpu' recommended)",
    )

    args = parser.parse_args()

    # Resolve paths
    input_path = Path(args.input)
    if not input_path.exists() and not args.input.isdigit():
        print(f"ERROR: Input file does not exist: {input_path}")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model checkpoint does not exist: {model_path}")
        print("Please run 'python scripts/download_model.py' first.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        model_path=str(model_path),
        enable_tracking=not args.no_tracking,
        enable_lanes=not args.no_lanes,
        enable_depth=args.enable_depth,
        depth_frequency=args.depth_freq,
        conf_thresh=args.conf,
        device=args.device,
    )

    print("=" * 65)
    print("AUTONOMOUS DRIVING PERCEPTION PIPELINE — CLI EXECUTION")
    print("=" * 65)
    print(f"  Input Source     : {args.input}")
    print(f"  Output Video     : {output_path}")
    print(f"  Model Checkpoint : {model_path}")
    print(f"  Tracking Active  : {config.enable_tracking}")
    print(f"  Lanes Active     : {config.enable_lanes}")
    print(f"  Depth Active     : {config.enable_depth} (every {config.depth_frequency} frames)")
    print(f"  Device           : {config.device}")
    print(f"  Max Frames       : {args.max_frames if args.max_frames > 0 else 'All'}")
    print("-" * 65)

    pipeline = PerceptionPipeline(config=config)

    max_f = args.max_frames if args.max_frames > 0 else None
    t0 = time.perf_counter()

    summary = pipeline.process_video(
        input_source=str(input_path) if not args.input.isdigit() else int(args.input),
        output_path=output_path,
        max_frames=max_f,
    )

    elapsed = time.perf_counter() - t0

    print("-" * 65)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"  Frames Processed : {summary['processed_frames']}")
    print(f"  Total Duration   : {elapsed:.2f} s")
    print(f"  Average Latency  : {summary['average_latency_ms']:.1f} ms/frame")
    print(f"  Effective FPS    : {summary['effective_fps']:.1f} FPS")
    print(f"  Annotated Video  : {summary['output_path']}")
    if "telemetry_log" in summary:
        print(f"  Telemetry Log    : {summary['telemetry_log']}")
    print("=" * 65)

if __name__ == "__main__":
    main()
