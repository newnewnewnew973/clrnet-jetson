#!/usr/bin/env python3
"""Run CLRNet DeepStream inference and draw lane overlays to a video file."""

from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from clrnet_postprocess import CulaneLayout, decode_lane_points


PREDICTION_SHAPE = (1, 192, 78)
MAX_LINES_PER_DISPLAY_META = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepStream CLRNet and write a lane-overlay video."
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "clrnet_deepstream/outputs/video_example_300.h264"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "clrnet_deepstream/outputs/deepstream_lanes.mp4"),
    )
    parser.add_argument(
        "--nvinfer-config",
        default=str(PROJECT_ROOT / "clrnet_deepstream/configs/nvinfer_clrnet.txt"),
    )
    parser.add_argument("--width", type=int, default=1640)
    parser.add_argument("--height", type=int, default=590)
    parser.add_argument("--crop-top", type=int, default=270)
    parser.add_argument("--conf-threshold", type=float, default=0.4)
    parser.add_argument("--max-lanes", type=int, default=4)
    parser.add_argument(
        "--compose-full-frame",
        action="store_true",
        help="Use nvcompositor to restore the original full frame inside the pipeline.",
    )
    parser.add_argument(
        "--decoder",
        choices=("auto", "hardware", "software"),
        default="auto",
        help="Use hardware decode when available, otherwise fall back to avdec_h264.",
    )
    parser.add_argument("--decoder-probe-timeout", type=int, default=8)
    return parser.parse_args()


def import_deepstream_modules():
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst
    import pyds

    return GLib, Gst, pyds


def layer_to_numpy(pyds, layer, shape: tuple[int, int, int]) -> np.ndarray:
    element_count = int(np.prod(shape))
    ptr = ctypes.cast(
        pyds.get_ptr(layer.buffer),
        ctypes.POINTER(ctypes.c_float),
    )
    array = np.ctypeslib.as_array(ptr, shape=(element_count,))
    return array.reshape(shape).copy()


def iter_meta_list(meta_list):
    item = meta_list
    while item is not None:
        yield item.data
        item = item.next


def find_prediction_layer(pyds, tensor_meta):
    fallback = None
    for index in range(tensor_meta.num_output_layers):
        layer = pyds.get_nvds_LayerInfo(tensor_meta, index)
        if fallback is None:
            fallback = layer
        layer_name = layer.layerName
        if isinstance(layer_name, bytes):
            layer_name = layer_name.decode("utf-8")
        if layer_name == "predictions":
            return layer
    return fallback


def add_lane_lines(
    pyds,
    batch_meta,
    frame_meta,
    lanes: list[np.ndarray],
    y_offset: int,
) -> None:
    display_meta = None
    line_index = 0

    def ensure_display_meta():
        nonlocal display_meta, line_index
        if display_meta is None or line_index >= MAX_LINES_PER_DISPLAY_META:
            if display_meta is not None:
                display_meta.num_lines = line_index
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            line_index = 0

    for points in lanes:
        if len(points) < 2:
            continue
        for start, end in zip(points[:-1], points[1:]):
            ensure_display_meta()
            line = display_meta.line_params[line_index]
            line.x1 = int(start[0])
            line.y1 = int(start[1] - y_offset)
            line.x2 = int(end[0])
            line.y2 = int(end[1] - y_offset)
            line.line_width = 4
            line.line_color.set(0.0, 1.0, 0.0, 1.0)
            line_index += 1

    if display_meta is not None:
        display_meta.num_lines = line_index
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)


def count_segments(lanes: list[np.ndarray]) -> int:
    return sum(max(len(points) - 1, 0) for points in lanes)


def make_probe(layout: CulaneLayout, Gst, pyds, stats: dict[str, int], y_offset: int):
    def probe(_pad, info, _user_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        for frame_data in iter_meta_list(batch_meta.frame_meta_list):
            frame_meta = pyds.NvDsFrameMeta.cast(frame_data)
            stats["frames"] += 1
            for user_data in iter_meta_list(frame_meta.frame_user_meta_list):
                user_meta = pyds.NvDsUserMeta.cast(user_data)
                if user_meta.base_meta.meta_type != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    continue
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                layer = find_prediction_layer(pyds, tensor_meta)
                if layer is None:
                    continue
                predictions = layer_to_numpy(pyds, layer, PREDICTION_SHAPE)
                lanes = decode_lane_points(predictions, layout)
                stats["lanes"] += len(lanes)
                stats["segments"] += count_segments(lanes)
                add_lane_lines(pyds, batch_meta, frame_meta, lanes, y_offset)
        return Gst.PadProbeReturn.OK

    return probe


def build_pipeline(
    input_path: Path,
    output_path: Path,
    config_path: Path,
    width: int,
    height: int,
    crop_top: int,
    decoder: str,
    compose_full_frame: bool,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cropped_height = height - crop_top
    if decoder == "hardware":
        decode_stage = [
            "h264parse",
            "!",
            "nvv4l2decoder",
            "!",
            "nvvideoconvert",
            "!",
            "video/x-raw(memory:NVMM),format=NV12",
        ]
    else:
        decode_stage = [
            "h264parse",
            "!",
            "avdec_h264",
            "!",
            "videoconvert",
            "!",
            "nvvideoconvert",
            "!",
            "video/x-raw(memory:NVMM),format=NV12",
        ]

    if compose_full_frame:
        if decoder != "hardware":
            raise ValueError("--compose-full-frame currently expects --decoder hardware")
        return " ".join(
            [
                f"filesrc location={input_path}",
                "!",
                "h264parse",
                "!",
                "nvv4l2decoder",
                "!",
                "nvvideoconvert",
                "!",
                f"video/x-raw(memory:NVMM),format=NV12,width={width},height={height}",
                "!",
                "tee name=t",
                "nvcompositor name=comp",
                "sink_0::xpos=0 sink_0::ypos=0",
                f"sink_0::width={width} sink_0::height={height} sink_0::zorder=0",
                f"sink_1::xpos=0 sink_1::ypos={crop_top}",
                f"sink_1::width={width} sink_1::height={cropped_height} sink_1::zorder=1",
                "!",
                "nvvideoconvert",
                "!",
                "video/x-raw,format=I420",
                "!",
                "x264enc tune=zerolatency speed-preset=ultrafast bitrate=8000",
                "!",
                "h264parse",
                "!",
                "qtmux",
                "!",
                f"filesink location={output_path}",
                "t.",
                "!",
                "queue",
                "!",
                "comp.sink_0",
                "t.",
                "!",
                "queue",
                "!",
                "nvvideoconvert",
                "!",
                "video/x-raw,format=NV12",
                "!",
                f"videocrop top={crop_top}",
                "!",
                "nvvideoconvert",
                "!",
                "video/x-raw(memory:NVMM),format=NV12",
                "!",
                "m.sink_0",
                f"nvstreammux name=m batch-size=1 width={width} height={cropped_height} batched-push-timeout=40000",
                "!",
                f"nvinfer name=clrnet config-file-path={config_path}",
                "!",
                "nvdsosd",
                "!",
                "queue",
                "!",
                "comp.sink_1",
            ]
        )

    streammux_height = cropped_height
    pre_mux_stage = [
        "nvvideoconvert",
        "!",
        "video/x-raw,format=NV12",
        "!",
        f"videocrop top={crop_top}",
        "!",
        "nvvideoconvert",
        "!",
        "video/x-raw(memory:NVMM),format=NV12",
        "!",
    ]

    return " ".join(
        [
            f"filesrc location={input_path}",
            "!",
            *decode_stage,
            "!",
            *pre_mux_stage,
            "m.sink_0",
            f"nvstreammux name=m batch-size=1 width={width} height={streammux_height} batched-push-timeout=40000",
            "!",
            f"nvinfer name=clrnet config-file-path={config_path}",
            "!",
            "nvdsosd",
            "!",
            "nvvideoconvert",
            "!",
            "video/x-raw,format=I420",
            "!",
            "x264enc tune=zerolatency speed-preset=ultrafast bitrate=8000",
            "!",
            "h264parse",
            "!",
            "qtmux",
            "!",
            f"filesink location={output_path}",
        ]
    )


def hardware_decoder_is_usable(input_path: Path, timeout_sec: int) -> bool:
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
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def resolve_decoder(requested: str, input_path: Path, timeout_sec: int) -> str:
    if requested != "auto":
        return requested
    if hardware_decoder_is_usable(input_path, timeout_sec):
        return "hardware"
    print("decoder_fallback=hardware_decode_unavailable_using_software_decode")
    return "software"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    config_path = Path(args.nvinfer_config)
    if not input_path.exists():
        raise FileNotFoundError(f"input video not found: {input_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"nvinfer config not found: {config_path}")
    decoder = resolve_decoder(args.decoder, input_path, args.decoder_probe_timeout)

    GLib, Gst, pyds = import_deepstream_modules()
    Gst.init(None)
    layout = CulaneLayout(
        ori_img_w=args.width,
        ori_img_h=args.height,
        conf_threshold=args.conf_threshold,
        max_lanes=args.max_lanes,
    )

    pipeline_text = build_pipeline(
        input_path,
        output_path,
        config_path,
        args.width,
        args.height,
        args.crop_top,
        decoder,
        args.compose_full_frame,
    )
    print(f"decoder={decoder}")
    print(f"compose_full_frame={args.compose_full_frame}")
    print(f"pipeline={pipeline_text}")
    pipeline = Gst.parse_launch(pipeline_text)

    nvinfer = pipeline.get_by_name("clrnet")
    src_pad = nvinfer.get_static_pad("src")
    stats = {"frames": 0, "lanes": 0, "segments": 0}
    src_pad.add_probe(
        Gst.PadProbeType.BUFFER,
        make_probe(layout, Gst, pyds, stats, args.crop_top),
        None,
    )

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(_bus, message):
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            print(f"error={error}")
            print(f"debug={debug}")
            loop.quit()
        elif message.type == Gst.MessageType.EOS:
            print(f"output={output_path}")
            print(
                f"overlay_frames={stats['frames']} "
                f"overlay_lanes={stats['lanes']} "
                f"overlay_segments={stats['segments']}"
            )
            loop.quit()

    bus.connect("message", on_message)
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
