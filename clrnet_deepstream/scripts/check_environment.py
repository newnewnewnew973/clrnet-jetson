#!/usr/bin/env python3
"""Check the local Jetson environment for CLRNet DeepStream deployment."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        return 127, "command not found"
    return completed.returncode, completed.stdout.strip()


def command_status(command: list[str]) -> dict[str, object]:
    code, output = run(command)
    return {
        "command": " ".join(command),
        "ok": code == 0,
        "returncode": code,
        "output": output,
    }


def gst_plugin_status(name: str) -> dict[str, object]:
    code, output = run(["gst-inspect-1.0", name])
    return {
        "plugin": name,
        "ok": code == 0,
        "returncode": code,
        "summary": output.splitlines()[0] if output else "",
    }


def hardware_decode_status(input_path: Path) -> dict[str, object]:
    command = [
        "gst-launch-1.0",
        "filesrc",
        f"location={input_path}",
        "!",
        "h264parse",
        "!",
        "nvv4l2decoder",
        "!",
        "fakesink",
        "sync=false",
    ]
    code, output = run(command)
    return {
        "command": " ".join(command),
        "ok": code == 0,
        "returncode": code,
        "summary": output.splitlines()[-1] if output else "",
    }


def module_import_status(name: str) -> dict[str, object]:
    spec_found = importlib.util.find_spec(name) is not None
    if not spec_found:
        return {"found": False, "import_ok": False, "error": "module not found"}
    try:
        __import__(name)
    except Exception as exc:
        return {"found": True, "import_ok": False, "error": str(exc)}
    return {"found": True, "import_ok": True, "error": ""}


def main() -> int:
    h264_input = PROJECT_ROOT / "clrnet_deepstream/outputs/video_example_300.h264"
    checks = {
        "paths": {
            "workspace": str(PROJECT_ROOT),
            "onnx_exists": (
                PROJECT_ROOT / "clrnet_deepstream/outputs/clrnet_dla34.onnx"
            ).exists(),
            "deepstream_engine_exists": (
                PROJECT_ROOT
                / "clrnet_deepstream/outputs/clrnet_dla34.onnx_b1_gpu0_fp16.engine"
            ).exists(),
            "video_zip_exists": (PROJECT_ROOT / "data/video_example.zip").exists(),
            "h264_input_exists": h264_input.exists(),
        },
        "commands": {
            "deepstream_app": command_status(["deepstream-app", "--version-all"]),
            "gst_launch": command_status(["gst-launch-1.0", "--version"]),
            "lsmod": command_status(["lsmod"]),
        },
        "gst_plugins": {
            "nvinfer": gst_plugin_status("nvinfer"),
            "nvstreammux": gst_plugin_status("nvstreammux"),
            "nvvideoconvert": gst_plugin_status("nvvideoconvert"),
            "nvv4l2decoder": gst_plugin_status("nvv4l2decoder"),
            "avdec_h264": gst_plugin_status("avdec_h264"),
            "x264enc": gst_plugin_status("x264enc"),
        },
        "python": {
            "gi": module_import_status("gi"),
            "pyds": module_import_status("pyds"),
            "numpy": module_import_status("numpy"),
        },
        "smoke": {
            "hardware_decode": hardware_decode_status(h264_input),
        },
    }
    print(json.dumps(checks, indent=2))

    required_ok = (
        checks["paths"]["onnx_exists"]
        and checks["paths"]["video_zip_exists"]
        and checks["paths"]["h264_input_exists"]
        and checks["commands"]["gst_launch"]["ok"]
        and checks["commands"]["lsmod"]["ok"]
        and checks["python"]["numpy"]["import_ok"]
        and checks["python"]["pyds"]["import_ok"]
    )
    deepstream_ok = (
        checks["gst_plugins"]["nvinfer"]["ok"]
        and checks["gst_plugins"]["nvstreammux"]["ok"]
        and checks["gst_plugins"]["nvvideoconvert"]["ok"]
        and checks["gst_plugins"]["nvv4l2decoder"]["ok"]
        and checks["gst_plugins"]["avdec_h264"]["ok"]
        and checks["gst_plugins"]["x264enc"]["ok"]
        and checks["smoke"]["hardware_decode"]["ok"]
    )
    return 0 if required_ok and deepstream_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
