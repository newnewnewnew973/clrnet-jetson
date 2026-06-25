import argparse
import importlib
from pathlib import Path

from runtime import (
    PROJECT_ROOT,
    configure_import_paths,
    ensure_numpy_bool_alias,
    resolve_culane_model_args,
)
from clrnet_common.culane import build_culane_dataset
from clrnet_tensorrt.calibration import CULaneEntropyCalibrator
from clrnet_tensorrt.tensorrt_runner import build_engine_from_onnx


ensure_numpy_bool_alias()
configure_import_paths()

try:
    importlib.import_module("clrnet.datasets")
    from clrnet.datasets.registry import build_dataset
    from clrnet.utils.config import Config
except ImportError:
    build_dataset = None
    Config = None


def parse_args():
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from CLRNet ONNX.")
    parser.add_argument("--model", default="dla34")
    parser.add_argument("--config", default=None)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--engine", default=None)
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "clrnet_inference/data/CULane"),
    )
    parser.add_argument("--workspace-gb", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--calib-samples", type=int, default=512)
    parser.add_argument("--calib-cache", default=None)
    return parser.parse_args()


def build_calibrator(args: argparse.Namespace, model_name: str):
    if not args.int8:
        return None
    if Config is None or build_dataset is None:
        raise RuntimeError("CLRNet dataset imports are required for INT8 calibration")
    if args.calib_samples < 1:
        raise ValueError("--calib-samples must be >= 1")

    _, config_path, _ = resolve_culane_model_args(args.model, args.config, None)
    data_root = Path(args.data_root)
    cache_path = Path(args.calib_cache) if args.calib_cache else (
        PROJECT_ROOT / f"clrnet_tensorrt/outputs/calibration/clrnet_{model_name}_int8.cache"
    )
    if not Path(config_path).exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    cfg = Config.fromfile(config_path)
    cfg.backbone.pretrained = False
    dataset = build_culane_dataset(build_dataset, cfg, data_root)
    print(f"calibration_config={config_path}")
    print(f"calibration_data_root={dataset.data_root}")
    print(f"calibration_dataset_total={len(dataset)}")
    print(f"calibration_samples={min(args.calib_samples, len(dataset))}")
    print(f"calibration_cache={cache_path}")
    return CULaneEntropyCalibrator(dataset, cache_path, args.calib_samples)


def main():
    args = parse_args()
    model_name, _, _ = resolve_culane_model_args(args.model, None, None)
    onnx_path = Path(args.onnx) if args.onnx else (
        PROJECT_ROOT / f"clrnet_tensorrt/outputs/onnx/clrnet_{model_name}.onnx"
    )
    engine_path = Path(args.engine) if args.engine else (
        PROJECT_ROOT / f"clrnet_tensorrt/outputs/engine/clrnet_{model_name}.engine"
    )
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    calibrator = build_calibrator(args, model_name)
    build_engine_from_onnx(
        onnx_path,
        engine_path,
        workspace_gb=args.workspace_gb,
        fp16=args.fp16,
        int8=args.int8,
        calibrator=calibrator,
    )
    print(f"model_name={model_name}")
    print(f"onnx={onnx_path}")
    print(f"engine={engine_path}")
    print(f"fp16={args.fp16}")
    print(f"int8={args.int8}")


if __name__ == "__main__":
    main()
