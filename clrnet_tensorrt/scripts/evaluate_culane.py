import argparse
import importlib
from pathlib import Path

import torch
from tqdm import tqdm

from runtime import (
    PROJECT_ROOT,
    configure_import_paths,
    ensure_numpy_bool_alias,
    load_checkpoint_for_inference,
    nms_build_message,
    resolve_culane_model_args,
    resolve_device,
)
from clrnet_common.culane import (
    build_culane_dataset,
    resolve_eval_output_dir,
    write_culane_prediction,
)
from clrnet_tensorrt.tensorrt_runner import TensorRTEngine


ensure_numpy_bool_alias()
configure_import_paths()

try:
    importlib.import_module("clrnet.datasets")
    importlib.import_module("clrnet.models")
    from clrnet.datasets.registry import build_dataset
    from clrnet.models.registry import build_net
    from clrnet.utils.config import Config
except ImportError as exc:
    if "nms_impl" in str(exc):
        raise SystemExit(nms_build_message()) from exc
    raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CULane predictions with a TensorRT CLRNet engine."
    )
    parser.add_argument("--model", default="dla34")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--engine", default=None)
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "clrnet_inference/data/CULane"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for quick smoke tests. Omit for full CULane eval.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1 when provided")


def main():
    args = parse_args()
    validate_args(args)
    model_name, args.config, args.checkpoint = resolve_culane_model_args(
        args.model,
        args.config,
        args.checkpoint,
    )
    cfg = Config.fromfile(args.config)
    cfg.backbone.pretrained = False

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data_root)
    engine_path = Path(args.engine) if args.engine else (
        PROJECT_ROOT / f"clrnet_tensorrt/outputs/engine/clrnet_{model_name}.engine"
    )
    output_dir = resolve_eval_output_dir(
        PROJECT_ROOT,
        "clrnet_tensorrt",
        model_name,
        args.output_dir,
    )

    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")
    if not engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("TensorRT CULane evaluation requires --device cuda")

    dataset = build_culane_dataset(build_dataset, cfg, data_root)
    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    if total < 1:
        raise RuntimeError("dataset is empty; cannot run TensorRT CULane eval")

    # Build the official model only for its tested CLRNet lane decoder/NMS path.
    decoder_model = build_net(cfg).to(device)
    load_checkpoint_for_inference(decoder_model, checkpoint_path)
    decoder_model.eval()
    engine = TensorRTEngine(engine_path)

    print(f"data_root={dataset.data_root}")
    print(f"config={config_path}")
    print(f"checkpoint={checkpoint_path}")
    print(f"engine={engine_path}")
    print(f"dataset_class={dataset.__class__.__name__}")
    print(f"dataset_total={len(dataset)}")
    print(f"evaluated_samples={total}")
    print(f"eval_list={dataset.list_path}")
    print(f"model_name={model_name}")
    print("batch_size=1")
    print(f"output_dir={output_dir}")

    with torch.inference_mode():
        for idx in tqdm(range(total), desc=f"{model_name}-tensorrt-eval"):
            sample = dataset[idx]
            tensor = sample["img"].unsqueeze(0).contiguous().to(device)
            outputs = engine.infer(tensor)
            predictions = outputs["predictions"]
            lanes = decoder_model.heads.get_lanes(predictions)[0]
            write_culane_prediction(dataset, idx, lanes, output_dir)

    torch.cuda.synchronize()
    print(f"predictions_written={total}")
    print(
        "metric_command="
        "python clrnet_tensorrt/scripts/measure_culane_metric.py "
        f"--model {model_name} --pred-dir {output_dir} "
        f"--data-root {data_root} --workers 4"
    )


if __name__ == "__main__":
    main()
