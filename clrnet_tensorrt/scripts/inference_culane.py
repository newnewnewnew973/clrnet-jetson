import argparse
import importlib
from pathlib import Path

import cv2
import torch

from runtime import (
    PROJECT_ROOT,
    configure_import_paths,
    ensure_numpy_bool_alias,
    load_checkpoint_for_inference,
    nms_build_message,
    resolve_culane_model_args,
    resolve_device,
)
from clrnet_common.image import (
    draw_lanes,
    find_first_image,
    lane_to_culane_line,
    preprocess_bgr,
)
from clrnet_tensorrt.tensorrt_runner import TensorRTEngine


ensure_numpy_bool_alias()
configure_import_paths()

try:
    importlib.import_module("clrnet.models")
    from clrnet.models.registry import build_net
    from clrnet.utils.config import Config
except ImportError as exc:
    if "nms_impl" in str(exc):
        raise SystemExit(nms_build_message(include_arch=False)) from exc
    raise


def parse_args():
    parser = argparse.ArgumentParser(description="Run TensorRT CLRNet on one CULane image.")
    parser.add_argument("--model", default="dla34")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--engine", default=None)
    parser.add_argument(
        "--image",
        default=None,
        help="CULane test image. If omitted, the script searches under data/.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "clrnet_tensorrt/outputs/single_image"),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name, args.config, args.checkpoint = resolve_culane_model_args(
        args.model,
        args.config,
        args.checkpoint,
    )
    cfg = Config.fromfile(args.config)
    cfg.backbone.pretrained = False

    engine_path = Path(args.engine) if args.engine else (
        PROJECT_ROOT / f"clrnet_tensorrt/outputs/engine/clrnet_{model_name}.engine"
    )
    checkpoint_path = Path(args.checkpoint)
    image_path = Path(args.image) if args.image else find_first_image(PROJECT_ROOT / "data")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("TensorRT inference requires --device cuda")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    # Build the official model only for its tested CLRNet lane decoder/NMS path.
    decoder_model = build_net(cfg).to(device)
    load_checkpoint_for_inference(decoder_model, checkpoint_path)
    decoder_model.eval()

    engine = TensorRTEngine(engine_path)
    tensor = preprocess_bgr(image, cfg).to(device)
    with torch.inference_mode():
        outputs = engine.infer(tensor)
        torch.cuda.synchronize()
        predictions = outputs["predictions"]
        lanes = decoder_model.heads.get_lanes(predictions)[0]

    stem = image_path.stem
    vis_path = output_dir / f"{stem}_clrnet_{model_name}_trt.jpg"
    txt_path = output_dir / f"{stem}.lines.txt"

    vis = draw_lanes(image, lanes, cfg)
    cv2.imwrite(str(vis_path), vis)

    culane_lines = [lane_to_culane_line(lane, cfg) for lane in lanes]
    culane_lines = [line for line in culane_lines if line]
    txt_path.write_text("\n".join(culane_lines), encoding="utf-8")

    print(f"image={image_path}")
    print(f"engine={engine_path}")
    print(f"model_name={model_name}")
    print(f"device={device}")
    print(f"lanes={len(lanes)}")
    print(f"visualization={vis_path}")
    print(f"culane_output={txt_path}")


if __name__ == "__main__":
    main()
