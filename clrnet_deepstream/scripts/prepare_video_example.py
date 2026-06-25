#!/usr/bin/env python3
"""Create an MP4 test video from data/video_example.zip."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract data/video_example.zip and encode it as MP4."
    )
    parser.add_argument(
        "--zip",
        default=str(PROJECT_ROOT / "data/video_example.zip"),
        help="Input zip containing JPG frames.",
    )
    parser.add_argument(
        "--frames-dir",
        default=str(PROJECT_ROOT / "clrnet_deepstream/outputs/video_example_frames"),
        help="Directory where frames are extracted.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "clrnet_deepstream/outputs/video_example.mp4"),
        help="Output MP4 path.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N sorted frames. Useful for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output video.",
    )
    parser.add_argument(
        "--codec",
        choices=("mp4v", "h264"),
        default="h264",
        help="Use raw H.264 for DeepStream smoke tests.",
    )
    return parser.parse_args()


def extract_frames(zip_path: Path, frames_dir: Path) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    if not any(frames_dir.rglob("*.jpg")):
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(frames_dir)
    return sorted(frames_dir.rglob("*.jpg"))


def write_video(frames: list[Path], output_path: Path, fps: float) -> None:
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"failed to read first frame: {frames[0]}")

    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output_path}")

    written = 0
    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"failed to read frame: {frame_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    print(f"frames={written}")
    print(f"size={width}x{height}")
    print(f"fps={fps}")
    print(f"video={output_path}")


def link_sequence(frames: list[Path], sequence_dir: Path) -> str:
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for old in sequence_dir.glob("frame_*.jpg"):
        old.unlink()
    for index, frame in enumerate(frames):
        target = sequence_dir / f"frame_{index:06d}.jpg"
        os.symlink(frame.resolve(), target)
    return str((sequence_dir / "frame_%06d.jpg").resolve())


def write_h264_video(frames: list[Path], output_path: Path, fps: float) -> None:
    if shutil.which("gst-launch-1.0") is None:
        raise RuntimeError("gst-launch-1.0 is required to write H.264 video")

    sequence_dir = output_path.parent / f"{output_path.stem}_sequence"
    pattern = link_sequence(frames, sequence_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fps_num = int(round(fps))
    command = [
        "gst-launch-1.0",
        "-e",
        "multifilesrc",
        f"location={pattern}",
        "index=0",
        f"caps=image/jpeg,framerate=(fraction){fps_num}/1",
        "!",
        "jpegdec",
        "!",
        "videoconvert",
        "!",
        "video/x-raw,format=I420",
        "!",
        "x264enc",
        "byte-stream=true",
        "tune=zerolatency",
        "speed-preset=ultrafast",
        "bitrate=8000",
        f"key-int-max={fps_num}",
        "!",
        "filesink",
        f"location={output_path.resolve()}",
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError("command=" + " ".join(command) + "\n" + completed.stdout)

    first = cv2.imread(str(frames[0]))
    height, width = first.shape[:2]
    print(f"frames={len(frames)}")
    print(f"size={width}x{height}")
    print(f"fps={fps}")
    print(f"codec=h264")
    print(f"video={output_path}")


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip)
    frames_dir = Path(args.frames_dir)
    output_path = Path(args.output)

    if not zip_path.exists():
        raise FileNotFoundError(f"video example zip not found: {zip_path}")
    if output_path.exists() and not args.overwrite:
        print(f"video_exists={output_path}")
        return 0

    frames = extract_frames(zip_path, frames_dir)
    if not frames:
        raise RuntimeError(f"no JPG frames found under: {frames_dir}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        frames = frames[: args.limit]
    if args.codec == "h264":
        write_h264_video(frames, output_path, args.fps)
    else:
        write_video(frames, output_path, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
