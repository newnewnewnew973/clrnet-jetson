import argparse
import importlib
from pathlib import Path

import torch
from tqdm import tqdm

from runtime import (
    PROJECT_ROOT,
    configure_import_paths,
    default_device_arg,
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
        description="Generate CULane predictions for an official CLRNet checkpoint."
    )
    parser.add_argument(
        "--model",
        default="dla34",
        help="Model preset name. Defaults to dla34 and maps to clr_<model>_culane.py.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override official CLRNet config path.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override trained CLRNet checkpoint path.",
    )
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "clrnet_inference/data/CULane"),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of CULane samples per forward pass. Use 1 for baseline parity.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for quick smoke tests. Omit for full CULane eval.",
    )
    parser.add_argument(
        "--run-official-evaluate",
        action="store_true",
        help=(
            "Also run the official dataset.evaluate() after writing predictions. "
            "This uses Pool(cpu_count()) inside CLRNet and is not recommended on Jetson."
        ),
    )
    return parser.parse_args()


def iter_batches(total: int, batch_size: int):
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield list(range(start, end))


def main():
    args = parse_args()
    if args.device is None:
        args.device = default_device_arg()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1 when provided")
    model_name, args.config, args.checkpoint = resolve_culane_model_args(
        args.model,
        args.config,
        args.checkpoint,
    )
    config = Path(args.config)
    if not config.exists():
        raise FileNotFoundError(f"config not found: {config}")
    cfg = Config.fromfile(args.config)
    cfg.backbone.pretrained = False

    data_root = Path(args.data_root)
    checkpoint = Path(args.checkpoint)
    output_dir = resolve_eval_output_dir(
        PROJECT_ROOT,
        "clrnet_inference",
        model_name,
        args.output_dir,
    )
    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_culane_dataset(build_dataset, cfg, data_root)
    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    if total < 1:
        raise RuntimeError("dataset is empty; cannot run CULane eval")

    device = resolve_device(args.device)
    model = build_net(cfg).to(device)
    load_checkpoint_for_inference(model, checkpoint)
    model.eval()

    print(f"data_root={dataset.data_root}")
    print(f"config={config}")
    print(f"checkpoint={checkpoint}")
    print(f"dataset_class={dataset.__class__.__name__}")
    print(f"dataset_total={len(dataset)}")
    print(f"evaluated_samples={total}")
    print(f"eval_list={dataset.list_path}")
    print(f"model_name={model_name}")
    print(f"batch_size={args.batch_size}")
    print(f"output_dir={output_dir}")

    predictions = [] if args.run_official_evaluate else None
    num_batches = (total + args.batch_size - 1) // args.batch_size
    with torch.inference_mode():
        for indices in tqdm(
            iter_batches(total, args.batch_size),
            total=num_batches,
            desc=f"{model_name}-dataset-eval",
        ):
            samples = [dataset[idx] for idx in indices]
            tensor = torch.stack([sample["img"] for sample in samples])
            tensor = tensor.contiguous().to(device)
            output = model(tensor)
            lanes_batch = model.heads.get_lanes(output)
            if len(lanes_batch) != len(indices):
                raise RuntimeError(
                    f"batch output size mismatch: {len(lanes_batch)} vs {len(indices)}"
                )
            for idx, lanes in zip(indices, lanes_batch):
                write_culane_prediction(dataset, idx, lanes, output_dir)
                if predictions is not None:
                    predictions.append(lanes)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"predictions_written={total}")

    if args.run_official_evaluate:
        # Official CLRNet keeps all predictions in memory and then calls
        # dataset.evaluate(). That path also invokes multiprocessing internally,
        # so it remains opt-in on Jetson.
        f1 = dataset.evaluate(predictions, str(output_dir))
        print(f"official_dataset_evaluate_F1={f1:.6f}")
    else:
        print(
            "official_dataset_evaluate_skipped=True "
            "(use scripts/measure_culane_metric.py --workers 4 for Jetson metric)"
        )


if __name__ == "__main__":
    main()
