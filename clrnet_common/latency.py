"""Latency measurement primitives shared by CLRNet experiment scripts."""

import statistics
import time
from collections.abc import Callable
from typing import TypeVar

import torch


T = TypeVar("T")


MEASUREMENT_DESCRIPTIONS_KO = {
    "wall_clock_breakdown": (
        "실제 파이프라인을 단계별로 강제 동기화하며 잰 값입니다. "
        "dataset/preprocess, H2D copy, model forward, postprocess 중 어디가 병목인지 볼 때 사용합니다."
    ),
    "cuda_event_forward": (
        "이미 GPU에 올라간 입력 텐서에 대해 model(tensor)만 CUDA event로 잰 값입니다. "
        "공식 논문 FPS나 TensorRT 변환 후 모델 자체 성능과 비교할 때 가장 중요한 기준입니다."
    ),
    "cuda_event_forward_postprocess": (
        "이미 GPU에 올라간 입력 텐서에 대해 model(tensor)와 get_lanes()를 합쳐 CUDA event로 잰 값입니다. "
        "GPU 관점에서 모델과 CUDA NMS/후처리 비용을 함께 볼 때 사용합니다."
    ),
    "wall_clock_model_postprocess": (
        "이미 GPU에 올라간 입력 텐서에 대해 model(tensor)와 get_lanes()를 wall-clock으로 잰 값입니다. "
        "get_lanes() 내부의 Python/CPU 오버헤드까지 포함한 실제 모델+후처리 비용입니다."
    ),
    "continuous_forward": (
        "같은 GPU 입력 텐서로 model(tensor)를 연속 실행하고 루프 시작/끝에서만 동기화한 throughput입니다. "
        "프레임 단위 GPU forward 처리량을 볼 때 사용합니다."
    ),
    "continuous_forward_postprocess": (
        "같은 GPU 입력 텐서로 model(tensor)+get_lanes()를 연속 실행한 throughput입니다. "
        "모델과 후처리를 포함한 반복 처리량을 볼 때 사용합니다."
    ),
    "continuous_e2e": (
        "dataset[idx]부터 H2D copy, model forward, get_lanes()까지 전체를 연속 실행한 throughput입니다. "
        "현재 PyTorch 평가 파이프라인의 실제 end-to-end 처리량에 가장 가깝습니다."
    ),
    "percentiles": (
        "p50/p90/p95/p99/max는 평균으로 숨겨지는 느린 프레임을 확인하기 위한 값입니다. "
        "실시간 시스템에서는 평균보다 p95, p99, max가 deadline 위반 여부를 보는 데 중요합니다."
    ),
}


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA work when timing a CUDA device."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def percentile(values: list[float], pct: float) -> float:
    """Return a linearly interpolated percentile for a non-empty value list."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_ms(values: list[float]) -> dict[str, float]:
    """Summarize latency values in milliseconds."""
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def measure_device_ms(device: torch.device, fn: Callable[[], T]) -> tuple[T, float]:
    """Measure a callable with CUDA events on CUDA devices, wall-clock otherwise."""
    if device.type != "cuda":
        start = time.perf_counter()
        result = fn()
        return result, (time.perf_counter() - start) * 1000.0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    result = fn()
    end_event.record()
    torch.cuda.synchronize()
    return result, start_event.elapsed_time(end_event)
