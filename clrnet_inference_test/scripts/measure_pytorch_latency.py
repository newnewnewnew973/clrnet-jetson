import argparse
import json
import statistics
import time
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
from clrnet_common.latency import (
    MEASUREMENT_DESCRIPTIONS_KO,
    measure_device_ms,
    summarize_ms,
    synchronize,
)


ensure_numpy_bool_alias()
configure_import_paths()

try:
    import clrnet.datasets  # noqa: F401,E402 - registers official datasets/processes
    import clrnet.models  # noqa: F401,E402 - registers official model modules
    from clrnet.datasets.registry import build_dataset  # noqa: E402
    from clrnet.models.registry import build_net  # noqa: E402
    from clrnet.utils.config import Config  # noqa: E402
except ImportError as exc:
    if "nms_impl" in str(exc):
        raise SystemExit(nms_build_message()) from exc
    raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure PyTorch CLRNet CULane inference latency/FPS."
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
        default=str(PROJECT_ROOT / "clrnet_inference_test/data/CULane"),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--block-warmup", type=int, default=10)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Fail fast on invalid benchmark sizes before loading the model."""
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.block_warmup < 0:
        raise ValueError("--block-warmup must be >= 0")


def build_culane_dataset(cfg, data_root: Path):
    cfg.dataset.test.data_root = str(data_root)
    return build_dataset(cfg.dataset.test, cfg)


def tensor_from_dataset(dataset, idx: int, device: torch.device):
    sample = dataset[idx]
    return sample["img"].unsqueeze(0).contiguous().to(device)


def run_forward_postprocess(model, tensor):
    output = model(tensor)
    return model.heads.get_lanes(output)[0]


def warmup_pipeline(model, dataset, count: int, device: torch.device) -> None:
    if count <= 0:
        return
    for idx in range(count):
        tensor = tensor_from_dataset(dataset, idx, device)
        _ = run_forward_postprocess(model, tensor)
    synchronize(device)


def warmup_tensor(model, tensor, count: int, device: torch.device) -> None:
    if count <= 0:
        return
    for _ in range(count):
        _ = run_forward_postprocess(model, tensor)
    synchronize(device)


def measure_wall_clock_breakdown(model, dataset, total: int, device: torch.device):
    metrics = {
        "dataset_preprocess": [],
        "h2d_copy": [],
        "forward": [],
        "postprocess": [],
        "total": [],
    }
    lane_counts = []

    for idx in tqdm(range(total), desc="pytorch-wall-clock-breakdown"):
        synchronize(device)
        start_total = time.perf_counter()

        start = time.perf_counter()
        sample = dataset[idx]
        cpu_tensor = sample["img"].unsqueeze(0).contiguous()
        metrics["dataset_preprocess"].append((time.perf_counter() - start) * 1000.0)

        synchronize(device)
        start = time.perf_counter()
        tensor = cpu_tensor.to(device)
        synchronize(device)
        metrics["h2d_copy"].append((time.perf_counter() - start) * 1000.0)

        synchronize(device)
        start = time.perf_counter()
        output = model(tensor)
        synchronize(device)
        metrics["forward"].append((time.perf_counter() - start) * 1000.0)

        synchronize(device)
        start = time.perf_counter()
        lanes = model.heads.get_lanes(output)[0]
        synchronize(device)
        metrics["postprocess"].append((time.perf_counter() - start) * 1000.0)
        metrics["total"].append((time.perf_counter() - start_total) * 1000.0)
        lane_counts.append(len(lanes))

    return metrics, lane_counts


def measure_model_latency(model, tensor, total: int, device: torch.device):
    forward_event_ms = []
    forward_postprocess_event_ms = []
    forward_postprocess_wall_ms = []

    for _ in tqdm(range(total), desc="pytorch-cuda-event-model"):
        _, elapsed_ms = measure_device_ms(device, lambda: model(tensor))
        forward_event_ms.append(elapsed_ms)

        _, elapsed_ms = measure_device_ms(
            device,
            lambda: run_forward_postprocess(model, tensor),
        )
        forward_postprocess_event_ms.append(elapsed_ms)

        synchronize(device)
        start = time.perf_counter()
        _ = run_forward_postprocess(model, tensor)
        synchronize(device)
        forward_postprocess_wall_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "forward_event": forward_event_ms,
        "forward_postprocess_event": forward_postprocess_event_ms,
        "forward_postprocess_wall": forward_postprocess_wall_ms,
    }


def measure_continuous_forward(model, tensor, total: int, device: torch.device) -> float:
    synchronize(device)
    start = time.perf_counter()
    for _ in range(total):
        _ = model(tensor)
    synchronize(device)
    return time.perf_counter() - start


def measure_continuous_forward_postprocess(
    model,
    tensor,
    total: int,
    device: torch.device,
) -> float:
    synchronize(device)
    start = time.perf_counter()
    for _ in range(total):
        _ = run_forward_postprocess(model, tensor)
    synchronize(device)
    return time.perf_counter() - start


def measure_continuous_e2e(model, dataset, total: int, device: torch.device) -> float:
    synchronize(device)
    start = time.perf_counter()
    for idx in range(total):
        tensor = tensor_from_dataset(dataset, idx, device)
        _ = run_forward_postprocess(model, tensor)
    synchronize(device)
    return time.perf_counter() - start


def write_report(path: Path, result: dict):
    wall = result["latency_ms"]["wall_clock_breakdown"]
    event = result["latency_ms"]["cuda_event"]
    model_post_wall = result["latency_ms"]["wall_clock_model_postprocess"]
    lines = [
        f"# PyTorch CLRNet {result['model_name']} Latency Baseline",
        "",
        "This latency measurement uses the official CLRNet CULane dataset class and the",
        "local external CUDA NMS extension.",
        "",
        "## Summary",
        "",
        f"- Device: `{result['device']}`",
        f"- Samples: `{result['samples']}`",
        f"- Warmup: `{result['warmup']}`",
        f"- Block warmup: `{result['block_warmup']}`",
        f"- Continuous end-to-end FPS: `{result['fps_continuous_e2e']:.2f}`",
        f"- Continuous end-to-end mean latency: `{result['latency_ms']['continuous_e2e']:.2f} ms`",
        f"- Wall-clock breakdown total mean latency: `{wall['total']['mean']:.2f} ms`",
        "- Dataset/preprocess wall-clock mean latency: "
        f"`{wall['dataset_preprocess']['mean']:.2f} ms`",
        f"- H2D copy wall-clock mean latency: `{wall['h2d_copy']['mean']:.2f} ms`",
        f"- Forward wall-clock mean latency: `{wall['forward']['mean']:.2f} ms`",
        f"- Postprocess wall-clock mean latency: `{wall['postprocess']['mean']:.2f} ms`",
        f"- Model forward CUDA-event FPS: `{result['fps_cuda_event_forward']:.2f}`",
        f"- Model forward CUDA-event mean latency: `{event['forward']['mean']:.2f} ms`",
        f"- Model forward CUDA-event p95/p99/max: "
        f"`{event['forward']['p95']:.2f} / "
        f"{event['forward']['p99']:.2f} / "
        f"{event['forward']['max']:.2f} ms`",
        f"- Forward + postprocess CUDA-event FPS: "
        f"`{result['fps_cuda_event_forward_postprocess']:.2f}`",
        f"- Forward + postprocess CUDA-event mean latency: "
        f"`{event['forward_postprocess']['mean']:.2f} ms`",
        f"- Forward + postprocess wall-clock mean latency: "
        f"`{model_post_wall['mean']:.2f} ms`",
        "",
        "## Korean Notes",
        "",
        f"- Wall-clock breakdown: {MEASUREMENT_DESCRIPTIONS_KO['wall_clock_breakdown']}",
        f"- CUDA event forward: {MEASUREMENT_DESCRIPTIONS_KO['cuda_event_forward']}",
        f"- CUDA event forward + postprocess: "
        f"{MEASUREMENT_DESCRIPTIONS_KO['cuda_event_forward_postprocess']}",
        f"- Wall-clock forward + postprocess: "
        f"{MEASUREMENT_DESCRIPTIONS_KO['wall_clock_model_postprocess']}",
        f"- Continuous forward: {MEASUREMENT_DESCRIPTIONS_KO['continuous_forward']}",
        f"- Continuous forward + postprocess: "
        f"{MEASUREMENT_DESCRIPTIONS_KO['continuous_forward_postprocess']}",
        f"- Continuous e2e: {MEASUREMENT_DESCRIPTIONS_KO['continuous_e2e']}",
        f"- Percentiles: {MEASUREMENT_DESCRIPTIONS_KO['percentiles']}",
        "",
        "## Notes",
        "",
        "- Timing excludes image/result file writing.",
        "- Model timings use CUDA events on CUDA devices, matching standard GPU "
        "benchmark practice.",
        "- Wall-clock breakdown separates dataset/preprocess, host-to-device copy, "
        "forward, and postprocess.",
        "- Dataset/preprocess and continuous end-to-end timings use wall-clock timing "
        "because they include CPU work.",
        "- Continuous end-to-end FPS synchronizes only before and after the measured loop.",
        "- Postprocess includes `model.heads.get_lanes()`, including CLRNet NMS.",
        "- This is the PyTorch latency/FPS baseline used before ONNX/TensorRT comparison.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def print_korean_measurement_notes():
    print("measurement_notes_ko:")
    print(f"  wall_clock_breakdown: {MEASUREMENT_DESCRIPTIONS_KO['wall_clock_breakdown']}")
    print(f"  cuda_event_forward: {MEASUREMENT_DESCRIPTIONS_KO['cuda_event_forward']}")
    print(
        "  cuda_event_forward_postprocess: "
        f"{MEASUREMENT_DESCRIPTIONS_KO['cuda_event_forward_postprocess']}"
    )
    print(
        "  wall_clock_model_postprocess: "
        f"{MEASUREMENT_DESCRIPTIONS_KO['wall_clock_model_postprocess']}"
    )
    print(f"  continuous_e2e: {MEASUREMENT_DESCRIPTIONS_KO['continuous_e2e']}")


def write_outputs(output_dir: Path, model_name: str, result: dict) -> tuple[Path, Path]:
    json_path = output_dir / f"pytorch_latency_{model_name}.json"
    report_path = output_dir / "PYTORCH_LATENCY_BASELINE.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(report_path, result)
    return json_path, report_path


def print_summary(result: dict, json_path: Path, report_path: Path) -> None:
    print(f"fps_continuous_e2e={result['fps_continuous_e2e']:.2f}")
    print(f"fps_continuous_forward={result['fps_continuous_forward']:.2f}")
    print(
        "fps_continuous_forward_postprocess="
        f"{result['fps_continuous_forward_postprocess']:.2f}"
    )
    print(f"fps_cuda_event_forward={result['fps_cuda_event_forward']:.2f}")
    print(
        "fps_cuda_event_forward_postprocess="
        f"{result['fps_cuda_event_forward_postprocess']:.2f}"
    )
    print(f"latency_continuous_e2e_ms={result['latency_ms']['continuous_e2e']:.2f}")
    print(
        "latency_wall_breakdown_total_mean_ms="
        f"{result['latency_ms']['wall_clock_breakdown']['total']['mean']:.2f}"
    )
    print(
        "latency_cuda_event_forward_mean_ms="
        f"{result['latency_ms']['cuda_event']['forward']['mean']:.2f}"
    )
    print(
        "latency_cuda_event_forward_p95_ms="
        f"{result['latency_ms']['cuda_event']['forward']['p95']:.2f}"
    )
    print(
        "latency_cuda_event_forward_p99_ms="
        f"{result['latency_ms']['cuda_event']['forward']['p99']:.2f}"
    )
    print(
        "latency_forward_postprocess_mean_ms="
        f"{result['latency_ms']['wall_clock_model_postprocess']['mean']:.2f}"
    )
    print_korean_measurement_notes()
    print(f"latency_json={json_path}")
    print(f"latency_report={report_path}")


def main():
    args = parse_args()
    validate_args(args)
    if args.device is None:
        args.device = default_device_arg()
    model_name, args.config, args.checkpoint = resolve_culane_model_args(
        args.model,
        args.config,
        args.checkpoint,
    )
    if args.output_dir is None:
        args.output_dir = str(
            PROJECT_ROOT / f"clrnet_inference_test/outputs/latency/pytorch_{model_name}"
        )
    cfg = Config.fromfile(args.config)
    cfg.backbone.pretrained = False

    data_root = Path(args.data_root)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    torch.backends.cudnn.benchmark = device.type == "cuda"
    dataset = build_culane_dataset(cfg, data_root)
    total = min(args.limit, len(dataset))
    if total < 1:
        raise RuntimeError("dataset is empty; cannot measure latency")
    warmup = min(args.warmup, total)
    block_warmup = min(args.block_warmup, total)

    model = build_net(cfg).to(device)
    load_checkpoint_for_inference(model, checkpoint)
    model.eval()

    print(f"data_root={dataset.data_root}")
    print(f"dataset_total={len(dataset)}")
    print(f"measured_samples={total}")
    print(f"warmup={warmup}")
    print(f"block_warmup={block_warmup}")
    print(f"model_name={model_name}")
    print(f"device={device}")
    print(f"cudnn_benchmark={torch.backends.cudnn.benchmark}")
    print(f"output_dir={output_dir}")

    with torch.inference_mode():
        warmup_pipeline(model, dataset, warmup, device)

        warmup_pipeline(model, dataset, block_warmup, device)
        wall_metrics, lane_counts = measure_wall_clock_breakdown(
            model,
            dataset,
            total,
            device,
        )

        event_tensor = tensor_from_dataset(dataset, 0, device)
        warmup_tensor(model, event_tensor, block_warmup, device)
        model_metrics = measure_model_latency(model, event_tensor, total, device)

        warmup_tensor(model, event_tensor, block_warmup, device)
        continuous_forward_sec = measure_continuous_forward(
            model,
            event_tensor,
            total,
            device,
        )

        warmup_tensor(model, event_tensor, block_warmup, device)
        continuous_forward_postprocess_sec = measure_continuous_forward_postprocess(
            model,
            event_tensor,
            total,
            device,
        )

        warmup_pipeline(model, dataset, block_warmup, device)
        continuous_e2e_sec = measure_continuous_e2e(model, dataset, total, device)

    continuous_e2e_ms = continuous_e2e_sec * 1000.0 / total
    continuous_forward_ms = continuous_forward_sec * 1000.0 / total
    continuous_forward_postprocess_ms = continuous_forward_postprocess_sec * 1000.0 / total

    result = {
        "device": str(device),
        "model_name": model_name,
        "samples": total,
        "warmup": warmup,
        "block_warmup": block_warmup,
        "data_root": str(data_root),
        "checkpoint": str(checkpoint),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "input_shape": list(event_tensor.shape),
        },
        "timing_method": {
            "wall_clock_breakdown": "wall_clock_with_per_stage_synchronize",
            "cuda_event": "cuda_event" if device.type == "cuda" else "wall_clock",
            "wall_clock_model_postprocess": "wall_clock_with_start_end_synchronize",
            "continuous_e2e": "wall_clock_with_start_end_synchronize",
            "continuous_forward": "wall_clock_with_start_end_synchronize",
        },
        "fps_continuous_e2e": total / continuous_e2e_sec,
        "fps_continuous_forward": total / continuous_forward_sec,
        "fps_continuous_forward_postprocess": total / continuous_forward_postprocess_sec,
        "fps_cuda_event_forward": (
            1000.0 / statistics.fmean(model_metrics["forward_event"])
        ),
        "fps_cuda_event_forward_postprocess": (
            1000.0 / statistics.fmean(model_metrics["forward_postprocess_event"])
        ),
        "latency_ms": {
            "wall_clock_breakdown": {
                "dataset_preprocess": summarize_ms(wall_metrics["dataset_preprocess"]),
                "h2d_copy": summarize_ms(wall_metrics["h2d_copy"]),
                "forward": summarize_ms(wall_metrics["forward"]),
                "postprocess": summarize_ms(wall_metrics["postprocess"]),
                "total": summarize_ms(wall_metrics["total"]),
            },
            "cuda_event": {
                "forward": summarize_ms(model_metrics["forward_event"]),
                "forward_postprocess": summarize_ms(
                    model_metrics["forward_postprocess_event"]
                ),
            },
            "wall_clock_model_postprocess": summarize_ms(
                model_metrics["forward_postprocess_wall"]
            ),
            "continuous_e2e": continuous_e2e_ms,
            "continuous_forward": continuous_forward_ms,
            "continuous_forward_postprocess": continuous_forward_postprocess_ms,
        },
        "lane_count": {
            "mean": statistics.fmean(lane_counts),
            "min": min(lane_counts),
            "max": max(lane_counts),
        },
    }

    json_path, report_path = write_outputs(output_dir, model_name, result)
    print_summary(result, json_path, report_path)


if __name__ == "__main__":
    main()
