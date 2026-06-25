import argparse
import importlib
from pathlib import Path

import onnx
import torch

from runtime import (
    PROJECT_ROOT,
    configure_import_paths,
    default_device_arg,
    ensure_numpy_bool_alias,
    load_checkpoint_for_inference,
    resolve_culane_model_args,
    resolve_device,
)
from clrnet_tensorrt.export import patch_model_for_onnx_export


ensure_numpy_bool_alias()
configure_import_paths()

importlib.import_module("clrnet.models")
from clrnet.models.registry import build_net
from clrnet.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Export CLRNet CULane model to ONNX.")
    parser.add_argument("--model", default="dla34")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Output ONNX path. Defaults to clrnet_tensorrt/outputs/onnx/<model>.onnx.",
    )
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device is None:
        args.device = default_device_arg()

    model_name, args.config, args.checkpoint = resolve_culane_model_args(
        args.model,
        args.config,
        args.checkpoint,
    )
    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / f"clrnet_tensorrt/outputs/onnx/clrnet_{model_name}.onnx"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    cfg = Config.fromfile(str(config_path))
    cfg.backbone.pretrained = False
    device = resolve_device(args.device)

    model = build_net(cfg).to(device)
    load_checkpoint_for_inference(model, checkpoint_path)
    model.eval()
    patch_model_for_onnx_export(model)

    dummy = torch.zeros(1, 3, cfg.img_h, cfg.img_w, dtype=torch.float32, device=device)
    with torch.inference_mode():
        output = model(dummy)
    print(f"pytorch_output_shape={tuple(output.shape)}")

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["predictions"],
        opset_version=args.opset,
        do_constant_folding=True,
        external_data=False,
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"model_name={model_name}")
    print(f"config={config_path}")
    print(f"checkpoint={checkpoint_path}")
    print(f"onnx={output_path}")
    print(f"opset={args.opset}")


if __name__ == "__main__":
    main()
