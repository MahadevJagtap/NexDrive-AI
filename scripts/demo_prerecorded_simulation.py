"""
Prerecorded Video Simulation Demo (CARLA Laptop-Compatible Alternative).

Demonstrates the perception pipeline operating on synthetic camera sensor streams
via VirtualCameraStream and SimulationFrame, processing real road frames without
requiring a full Unreal Engine CARLA simulator installation.

NOTICE:
This is an offline playback simulation alternative using prerecorded driving video.
It is NOT a live CARLA Unreal Engine simulation.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.perception_pipeline import PerceptionPipeline, PipelineConfig
from src.simulation.camera_adapter import VirtualCameraStream


def run_prerecorded_simulation_demo(
    video_path: str = "data/demo_banner.mp4",
    output_path: str = "results/prerecorded_simulation_demo.mp4",
    model_path: str = "models/best.pt",
    max_frames: int = 60,
    enable_depth: bool = True,
    depth_freq: int = 5,
    device: str = "cpu",
) -> None:
    print("=" * 70)
    print("PRERECORDED VIDEO SIMULATION DEMO (CARLA ALTERNATIVE)")
    print("=" * 70)
    print("  [DISCLAIMER] This script uses VirtualCameraStream to replay driving frames")
    print("  through the CARLA/ROS2 sensor message container (SimulationFrame).")
    print("  It is a reproducible LAPTOP-COMPATIBLE ALTERNATIVE and NOT live CARLA UE4.")
    print("-" * 70)
    print(f"  Input Source    : {video_path}")
    print(f"  Output Video    : {output_path}")
    print(f"  Model Path      : {model_path}")
    print(f"  Max Frames      : {max_frames}")
    print(f"  Depth Enabled   : {enable_depth} (freq={depth_freq})")
    print(f"  Device          : {device}")
    print("=" * 70)

    in_file = Path(video_path)
    if not in_file.exists():
        raise FileNotFoundError(f"Input video not found: {in_file}")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Virtual Camera Stream (mirrors sensor.camera.rgb stream)
    streamer = VirtualCameraStream(video_source=str(in_file), target_fps=30.0)

    # 2. Configure Perception Pipeline
    config = PipelineConfig(
        model_path=str(model_path),
        enable_tracking=True,
        enable_lanes=True,
        enable_depth=enable_depth,
        depth_frequency=depth_freq,
        device=device,
        draw_hud=True,
    )
    pipeline = PerceptionPipeline(config=config)

    # Setup video writer
    writer = None
    frames_processed = 0
    total_latency_ms = 0.0
    t_start = time.perf_counter()

    for sim_frame in streamer.stream(max_frames=max_frames):
        # Process the simulated camera frame
        result = pipeline.process_frame(sim_frame.image, frame_index=sim_frame.frame_id)
        frames_processed += 1
        total_latency_ms += result.total_latency_ms

        annotated = result.annotated_frame.copy()

        # Add simulation metadata watermark
        h, w = annotated.shape[:2]
        watermark = f"SIM-STREAM #{sim_frame.frame_id:04d} | t={sim_frame.timestamp_sec:.2f}s | [PRERECORDED ALT]"
        cv2.putText(
            annotated,
            watermark,
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_file), fourcc, 20.0, (w, h))

        writer.write(annotated)

        if frames_processed % 15 == 0 or frames_processed == max_frames:
            print(f"  Processed {frames_processed}/{max_frames} frames | Frame Latency: {result.total_latency_ms:.1f}ms ({result.effective_fps:.1f} FPS)")

    if writer is not None:
        writer.release()

    total_time = time.perf_counter() - t_start
    avg_latency = total_latency_ms / frames_processed if frames_processed > 0 else 0.0
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    print("-" * 70)
    print("SIMULATION DEMO COMPLETED SUCCESSFULLY!")
    print(f"  Total Frames  : {frames_processed}")
    print(f"  Total Time    : {total_time:.2f} s")
    print(f"  Mean Latency  : {avg_latency:.1f} ms/frame")
    print(f"  Mean FPS      : {avg_fps:.1f} FPS")
    print(f"  Saved Output  : {out_file} ({out_file.stat().st_size} bytes)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Prerecorded Video Simulation Demo (CARLA Alternative)")
    parser.add_argument("--input", "-i", type=str, default="data/demo_banner.mp4")
    parser.add_argument("--output", "-o", type=str, default="results/prerecorded_simulation_demo.mp4")
    parser.add_argument("--model", "-m", type=str, default="models/best.pt")
    parser.add_argument("--max-frames", "-n", type=int, default=30)
    parser.add_argument("--enable-depth", action="store_true", default=False)
    parser.add_argument("--depth-freq", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_prerecorded_simulation_demo(
        video_path=args.input,
        output_path=args.output,
        model_path=args.model,
        max_frames=args.max_frames,
        enable_depth=args.enable_depth,
        depth_freq=args.depth_freq,
        device=args.device,
    )


if __name__ == "__main__":
    main()
