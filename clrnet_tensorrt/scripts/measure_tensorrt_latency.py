import argparse
import importlib
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
from clrnet_common.culane import build_culane_dataset
from clrnet_common.latency import (
    MEASUREMENT_DESCRIPTIONS_KO,
    measure_device_ms,
    summarize_ms,
    synchronize,
)
from clrnet_tensorrt.tensorrt_runner import TensorRTEngine, trt_dtype_to_torch


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


PRECISION_ORDER = ("fp32", "fp16", "int8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure TensorRT CLRNet CULane inference latency/FPS."
    )
    parser.add_argument("--model", default="dla34")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "clrnet_inference/data/CULane"),
    )
    parser.add_argument(
        "--precision",
        nargs="+",
        default=["all"],
        choices=("all", "fp32", "fp16", "int8"),
        help="TensorRT precision(s) to measure. Defaults to all.",
    )
    parser.add_argument("--fp32-engine", default=None)
    parser.add_argument("--fp16-engine", default=None)
    parser.add_argument("--int8-engine", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--block-warmup", type=int, default=10)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.block_warmup < 0:
        raise ValueError("--block-warmup must be >= 0")


def selected_precisions(args: argparse.Namespace) -> list[str]:
    if "all" in args.precision:
        return list(PRECISION_ORDER)
    return [precision for precision in PRECISION_ORDER if precision in args.precision]


def first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_engine_paths(args: argparse.Namespace, model_name: str) -> dict[str, Path]:
    engine_dir = PROJECT_ROOT / "clrnet_tensorrt/outputs/engine"
    return {
        "fp32": Path(args.fp32_engine) if args.fp32_engine else first_existing_path(
            [
                engine_dir / f"clrnet_{model_name}.engine",
                engine_dir / f"clrnet_{model_name}_debug.engine",
            ]
        ),
        "fp16": Path(args.fp16_engine)
        if args.fp16_engine
        else engine_dir / f"clrnet_{model_name}_fp16.engine",
        "int8": Path(args.int8_engine)
        if args.int8_engine
        else engine_dir / f"clrnet_{model_name}_int8.engine",
    }


def tensor_from_dataset(
    dataset,
    idx: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    sample = dataset[idx]
    return sample["img"].unsqueeze(0).contiguous().to(device=device, dtype=dtype)


class PreallocatedEngineCall:
    """Reusable TensorRT bindings for measuring engine execution without allocation."""

    def __init__(self, engine: TensorRTEngine, input_tensor: torch.Tensor):
        if input_tensor.dtype != engine.input_dtype:
            raise ValueError(
                "TensorRT input dtype mismatch: "
                f"engine expects {engine.input_dtype}, got {input_tensor.dtype}"
            )
        self.engine = engine
        self.input_tensor = input_tensor.contiguous()
        self.outputs = {}
        self._bind_once()

    def _bind_once(self) -> None:
        context = self.engine.context
        input_name = self.engine.input_name
        context.set_input_shape(input_name, tuple(self.input_tensor.shape))
        context.set_tensor_address(input_name, self.input_tensor.data_ptr())

        for name in self.engine.output_names:
            shape = tuple(context.get_tensor_shape(name))
            dtype = trt_dtype_to_torch(self.engine.engine.get_tensor_dtype(name))
            output = torch.empty(shape, dtype=dtype, device=self.input_tensor.device)
            context.set_tensor_address(name, output.data_ptr())
            self.outputs[name] = output

    def execute(self) -> dict[str, torch.Tensor]:
        stream = torch.cuda.current_stream().cuda_stream
        if not self.engine.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return self.outputs


def run_forward(engine: TensorRTEngine, tensor: torch.Tensor):
    return engine.infer(tensor)["predictions"]


def run_forward_postprocess(decoder_model, engine: TensorRTEngine, tensor: torch.Tensor):
    predictions = run_forward(engine, tensor)
    return decoder_model.heads.get_lanes(predictions)[0]


def warmup_pipeline(
    decoder_model,
    engine: TensorRTEngine,
    dataset,
    count: int,
    device: torch.device,
) -> None:
    if count <= 0:
        return
    for idx in range(count):
        tensor = tensor_from_dataset(dataset, idx, device, engine.input_dtype)
        _ = run_forward_postprocess(decoder_model, engine, tensor)
    synchronize(device)


def warmup_tensor(
    decoder_model,
    engine: TensorRTEngine,
    tensor: torch.Tensor,
    count: int,
    device: torch.device,
) -> None:
    if count <= 0:
        return
    for _ in range(count):
        _ = run_forward_postprocess(decoder_model, engine, tensor)
    synchronize(device)


def measure_wall_clock_breakdown(
    decoder_model,
    engine: TensorRTEngine,
    dataset,
    total: int,
    device: torch.device,
    precision: str,
):
    metrics = {
        "dataset_preprocess": [],
        "h2d_copy": [],
        "forward": [],
        "postprocess": [],
        "total": [],
    }
    lane_counts = []

    for idx in tqdm(range(total), desc=f"trt-{precision}-wall-clock-breakdown"):
        synchronize(device)
        start_total = time.perf_counter()

        start = time.perf_counter()
        sample = dataset[idx]
        cpu_tensor = sample["img"].unsqueeze(0).contiguous()
        metrics["dataset_preprocess"].append((time.perf_counter() - start) * 1000.0)

        synchronize(device)
        start = time.perf_counter()
        tensor = cpu_tensor.to(device=device, dtype=engine.input_dtype)
        synchronize(device)
        metrics["h2d_copy"].append((time.perf_counter() - start) * 1000.0)

        synchronize(device)
        start = time.perf_counter()
        predictions = run_forward(engine, tensor)
        synchronize(device)
        metrics["forward"].append((time.perf_counter() - start) * 1000.0)

        synchronize(device)
        start = time.perf_counter()
        lanes = decoder_model.heads.get_lanes(predictions)[0]
        synchronize(device)
        metrics["postprocess"].append((time.perf_counter() - start) * 1000.0)
        metrics["total"].append((time.perf_counter() - start_total) * 1000.0)
        lane_counts.append(len(lanes))

    return metrics, lane_counts


def measure_model_latency(
    decoder_model,
    engine: TensorRTEngine,
    tensor: torch.Tensor,
    total: int,
    device: torch.device,
    precision: str,
):
    forward_event_ms = []
    forward_postprocess_event_ms = []
    forward_postprocess_wall_ms = []

    for _ in tqdm(range(total), desc=f"trt-{precision}-cuda-event-model"):
        _, elapsed_ms = measure_device_ms(device, lambda: run_forward(engine, tensor))
        forward_event_ms.append(elapsed_ms)

        _, elapsed_ms = measure_device_ms(
            device,
            lambda: run_forward_postprocess(decoder_model, engine, tensor),
        )
        forward_postprocess_event_ms.append(elapsed_ms)

        synchronize(device)
        start = time.perf_counter()
        _ = run_forward_postprocess(decoder_model, engine, tensor)
        synchronize(device)
        forward_postprocess_wall_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "forward_event": forward_event_ms,
        "forward_postprocess_event": forward_postprocess_event_ms,
        "forward_postprocess_wall": forward_postprocess_wall_ms,
    }


def measure_pure_engine_latency(
    preallocated_call: PreallocatedEngineCall,
    total: int,
    device: torch.device,
    precision: str,
):
    event_ms = []

    for _ in tqdm(range(total), desc=f"trt-{precision}-pure-engine-event"):
        _, elapsed_ms = measure_device_ms(device, preallocated_call.execute)
        event_ms.append(elapsed_ms)

    return event_ms


def measure_continuous_forward(
    preallocated_call: PreallocatedEngineCall,
    total: int,
    device: torch.device,
) -> float:
    synchronize(device)
    start = time.perf_counter()
    for _ in range(total):
        _ = preallocated_call.execute()
    synchronize(device)
    return time.perf_counter() - start


def measure_continuous_forward_postprocess(
    decoder_model,
    engine: TensorRTEngine,
    tensor: torch.Tensor,
    total: int,
    device: torch.device,
) -> float:
    synchronize(device)
    start = time.perf_counter()
    for _ in range(total):
        _ = run_forward_postprocess(decoder_model, engine, tensor)
    synchronize(device)
    return time.perf_counter() - start


def measure_continuous_e2e(
    decoder_model,
    engine: TensorRTEngine,
    dataset,
    total: int,
    device: torch.device,
) -> float:
    synchronize(device)
    start = time.perf_counter()
    for idx in range(total):
        tensor = tensor_from_dataset(dataset, idx, device, engine.input_dtype)
        _ = run_forward_postprocess(decoder_model, engine, tensor)
    synchronize(device)
    return time.perf_counter() - start


def measure_precision(
    precision: str,
    engine_path: Path,
    decoder_model,
    dataset,
    total: int,
    warmup: int,
    block_warmup: int,
    device: torch.device,
) -> dict:
    if not engine_path.exists():
        raise FileNotFoundError(f"{precision} TensorRT engine not found: {engine_path}")

    engine = TensorRTEngine(engine_path)
    print(f"{precision}_engine_input_dtype={engine.input_dtype}")
    with torch.inference_mode():
        warmup_pipeline(decoder_model, engine, dataset, warmup, device)

        warmup_pipeline(decoder_model, engine, dataset, block_warmup, device)
        wall_metrics, lane_counts = measure_wall_clock_breakdown(
            decoder_model,
            engine,
            dataset,
            total,
            device,
            precision,
        )

        event_tensor = tensor_from_dataset(dataset, 0, device, engine.input_dtype)
        preallocated_call = PreallocatedEngineCall(engine, event_tensor)
        warmup_tensor(decoder_model, engine, event_tensor, block_warmup, device)
        for _ in range(block_warmup):
            _ = preallocated_call.execute()
        synchronize(device)
        pure_engine_event_ms = measure_pure_engine_latency(
            preallocated_call,
            total,
            device,
            precision,
        )

        warmup_tensor(decoder_model, engine, event_tensor, block_warmup, device)
        model_metrics = measure_model_latency(
            decoder_model,
            engine,
            event_tensor,
            total,
            device,
            precision,
        )

        warmup_tensor(decoder_model, engine, event_tensor, block_warmup, device)
        for _ in range(block_warmup):
            _ = preallocated_call.execute()
        synchronize(device)
        continuous_forward_sec = measure_continuous_forward(
            preallocated_call,
            total,
            device,
        )

        warmup_tensor(decoder_model, engine, event_tensor, block_warmup, device)
        continuous_forward_postprocess_sec = measure_continuous_forward_postprocess(
            decoder_model,
            engine,
            event_tensor,
            total,
            device,
        )

        warmup_pipeline(decoder_model, engine, dataset, block_warmup, device)
        continuous_e2e_sec = measure_continuous_e2e(
            decoder_model,
            engine,
            dataset,
            total,
            device,
        )

    continuous_e2e_ms = continuous_e2e_sec * 1000.0 / total
    continuous_forward_ms = continuous_forward_sec * 1000.0 / total
    continuous_forward_postprocess_ms = continuous_forward_postprocess_sec * 1000.0 / total

    return {
        "precision": precision,
        "engine": str(engine_path),
        "engine_size_mb": engine_path.stat().st_size / (1024 * 1024),
        "engine_input_dtype": str(engine.input_dtype),
        "fps_pure_engine_cuda_event": (
            1000.0 / statistics.fmean(pure_engine_event_ms)
        ),
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
                "pure_engine_forward": summarize_ms(pure_engine_event_ms),
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


def write_report(path: Path, result: dict):
    lines = [
        f"# TensorRT CLRNet {result['model_name']} Latency",
        "",
        "This report measures TensorRT engines with the same CULane dataset pipeline",
        "and CLRNet lane decoder/NMS path used by the PyTorch baseline.",
        "",
        "## Summary",
        "",
        f"- Device: `{result['device']}`",
        f"- Samples: `{result['samples']}`",
        f"- Warmup: `{result['warmup']}`",
        f"- Block warmup: `{result['block_warmup']}`",
        "",
        "### Pure TensorRT Engine",
        "",
        "| Precision | Input dtype | Engine MB | Pure engine FPS | Pure engine ms | Pure engine p95/p99/max ms |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for precision in PRECISION_ORDER:
        item = result["results"].get(precision)
        if item is None:
            continue
        pure = item["latency_ms"]["cuda_event"]["pure_engine_forward"]
        lines.append(
            "| "
            f"{precision.upper()} | "
            f"`{item['engine_input_dtype']}` | "
            f"{item['engine_size_mb']:.1f} | "
            f"{item['fps_pure_engine_cuda_event']:.2f} | "
            f"{pure['mean']:.2f} | "
            f"{pure['p95']:.2f}/{pure['p99']:.2f}/{pure['max']:.2f} |"
        )

    lines.extend(
        [
            "",
            "### Pipeline-Compatible Timing",
            "",
            "| Precision | Runner forward ms | Runner forward p95/p99/max ms | Forward+post FPS | E2E FPS | E2E ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for precision in PRECISION_ORDER:
        item = result["results"].get(precision)
        if item is None:
            continue
        forward = item["latency_ms"]["cuda_event"]["forward"]
        lines.append(
            "| "
            f"{precision.upper()} | "
            f"{forward['mean']:.2f} | "
            f"{forward['p95']:.2f}/{forward['p99']:.2f}/{forward['max']:.2f} | "
            f"{item['fps_cuda_event_forward_postprocess']:.2f} | "
            f"{item['fps_continuous_e2e']:.2f} | "
            f"{item['latency_ms']['continuous_e2e']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Korean Notes",
            "",
            f"- Wall-clock breakdown: {MEASUREMENT_DESCRIPTIONS_KO['wall_clock_breakdown']}",
            f"- CUDA event forward: {MEASUREMENT_DESCRIPTIONS_KO['cuda_event_forward']}",
            f"- CUDA event forward + postprocess: "
            f"{MEASUREMENT_DESCRIPTIONS_KO['cuda_event_forward_postprocess']}",
            f"- Wall-clock forward + postprocess: "
            f"{MEASUREMENT_DESCRIPTIONS_KO['wall_clock_model_postprocess']}",
            f"- Continuous e2e: {MEASUREMENT_DESCRIPTIONS_KO['continuous_e2e']}",
            f"- Percentiles: {MEASUREMENT_DESCRIPTIONS_KO['percentiles']}",
            "",
            "## Notes",
            "",
            "- Timing excludes prediction file writing.",
            "- Forward timing measures TensorRT `execute_async_v3` with CUDA events.",
            "- Forward + postprocess includes the tested CLRNet `heads.get_lanes()` path.",
            "- End-to-end timing includes dataset access, preprocessing, H2D copy, TensorRT forward, and CLRNet lane decode/NMS.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(output_dir: Path, model_name: str, result: dict) -> tuple[Path, Path]:
    json_path = output_dir / f"tensorrt_latency_{model_name}.json"
    report_path = output_dir / "TENSORRT_LATENCY.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(report_path, result)
    return json_path, report_path


def print_summary(result: dict, json_path: Path, report_path: Path) -> None:
    for precision, item in result["results"].items():
        pure = item["latency_ms"]["cuda_event"]["pure_engine_forward"]
        forward = item["latency_ms"]["cuda_event"]["forward"]
        print(f"{precision}_engine={item['engine']}")
        print(f"{precision}_engine_size_mb={item['engine_size_mb']:.2f}")
        print(f"{precision}_engine_input_dtype={item['engine_input_dtype']}")
        print(
            f"{precision}_fps_pure_engine_cuda_event="
            f"{item['fps_pure_engine_cuda_event']:.2f}"
        )
        print(
            f"{precision}_latency_pure_engine_cuda_event_mean_ms="
            f"{pure['mean']:.2f}"
        )
        print(f"{precision}_fps_cuda_event_forward={item['fps_cuda_event_forward']:.2f}")
        print(f"{precision}_latency_cuda_event_forward_mean_ms={forward['mean']:.2f}")
        print(f"{precision}_latency_cuda_event_forward_p95_ms={forward['p95']:.2f}")
        print(f"{precision}_latency_cuda_event_forward_p99_ms={forward['p99']:.2f}")
        print(f"{precision}_fps_continuous_e2e={item['fps_continuous_e2e']:.2f}")
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
            PROJECT_ROOT / f"clrnet_tensorrt/outputs/latency/tensorrt_{model_name}"
        )

    cfg = Config.fromfile(args.config)
    cfg.backbone.pretrained = False

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("TensorRT latency measurement requires --device cuda")

    torch.backends.cudnn.benchmark = True
    dataset = build_culane_dataset(build_dataset, cfg, data_root)
    total = min(args.limit, len(dataset))
    warmup = min(args.warmup, total)
    block_warmup = min(args.block_warmup, total)

    decoder_model = build_net(cfg).to(device)
    load_checkpoint_for_inference(decoder_model, checkpoint_path)
    decoder_model.eval()

    engine_paths = resolve_engine_paths(args, model_name)
    precisions = selected_precisions(args)
    for precision in precisions:
        if not engine_paths[precision].exists():
            raise FileNotFoundError(
                f"{precision} TensorRT engine not found: {engine_paths[precision]}"
            )

    print(f"data_root={dataset.data_root}")
    print(f"dataset_total={len(dataset)}")
    print(f"measured_samples={total}")
    print(f"warmup={warmup}")
    print(f"block_warmup={block_warmup}")
    print(f"model_name={model_name}")
    print(f"device={device}")
    print(f"cudnn_benchmark={torch.backends.cudnn.benchmark}")
    print(f"output_dir={output_dir}")

    results = {}
    for precision in precisions:
        print(f"measure_precision={precision}")
        results[precision] = measure_precision(
            precision,
            engine_paths[precision],
            decoder_model,
            dataset,
            total,
            warmup,
            block_warmup,
            device,
        )

    result = {
        "device": str(device),
        "model_name": model_name,
        "samples": total,
        "warmup": warmup,
        "block_warmup": block_warmup,
        "data_root": str(data_root),
        "checkpoint": str(checkpoint_path),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "timing_method": {
            "wall_clock_breakdown": "wall_clock_with_per_stage_synchronize",
            "cuda_event": "cuda_event",
            "wall_clock_model_postprocess": "wall_clock_with_start_end_synchronize",
            "continuous_e2e": "wall_clock_with_start_end_synchronize",
            "continuous_forward": "wall_clock_with_start_end_synchronize",
        },
        "results": results,
    }

    json_path, report_path = write_outputs(output_dir, model_name, result)
    print_summary(result, json_path, report_path)


if __name__ == "__main__":
    main()
