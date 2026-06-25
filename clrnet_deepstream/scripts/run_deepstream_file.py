#!/usr/bin/env python3
"""Run a DeepStream GPU smoke pipeline for CLRNet."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLRNet DeepStream smoke test.")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "clrnet_deepstream/outputs/video_example_300.h264"),
    )
    parser.add_argument(
        "--nvinfer-config",
        default=str(PROJECT_ROOT / "clrnet_deepstream/configs/nvinfer_clrnet.txt"),
    )
    parser.add_argument(
        "--sink",
        choices=("fakesink", "display", "file"),
        default="fakesink",
        help="Use fakesink for SSH/headless validation.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "clrnet_deepstream/outputs/deepstream_output.mp4"),
    )
    parser.add_argument("--width", type=int, default=1640)
    parser.add_argument("--height", type=int, default=590)
    return parser.parse_args()


def require_deepstream() -> None:
    if shutil.which("gst-launch-1.0") is None:
        raise SystemExit("gst-launch-1.0 is not installed")

    for plugin in ("nvv4l2decoder", "nvstreammux", "nvinfer"):
        completed = subprocess.run(
            ["gst-inspect-1.0", plugin],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise SystemExit(
                f"{plugin} is not available. Run inside the DeepStream container."
            )


def build_pipeline(
    input_path: Path,
    config_path: Path,
    sink: str,
    output_path: Path,
    width: int,
    height: int,
) -> list[str]:
    if sink == "fakesink":
        tail = ["fakesink", "sync=false"]
    elif sink == "display":
        tail = ["nveglglessink"]
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tail = [
            "nvvideoconvert",
            "!",
            "video/x-raw,format=I420",
            "!",
            "x264enc",
            "tune=zerolatency",
            "speed-preset=ultrafast",
            "bitrate=8000",
            "!",
            "h264parse",
            "!",
            "qtmux",
            "!",
            "filesink",
            f"location={output_path}",
        ]
    return [
        "gst-launch-1.0",
        "-e",
        "filesrc",
        f"location={input_path}",
        "!",
        "h264parse",
        "!",
        "nvv4l2decoder",
        "!",
        "nvvideoconvert",
        "!",
        "video/x-raw(memory:NVMM),format=NV12",
        "!",
        "m.sink_0",
        "nvstreammux",
        "name=m",
        "batch-size=1",
        f"width={width}",
        f"height={height}",
        "batched-push-timeout=40000",
        "!",
        "nvinfer",
        f"config-file-path={config_path}",
        "!",
        *tail,
    ]


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    config_path = Path(args.nvinfer_config)
    if not input_path.exists():
        raise FileNotFoundError(f"input video not found: {input_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"nvinfer config not found: {config_path}")

    require_deepstream()

    command = build_pipeline(
        input_path=input_path,
        config_path=config_path,
        sink=args.sink,
        output_path=Path(args.output),
        width=args.width,
        height=args.height,
    )
    print("command=" + " ".join(command))
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
