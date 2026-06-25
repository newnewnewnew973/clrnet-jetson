from pathlib import Path

import pytest

from clrnet_common.culane import resolve_eval_output_dir
from clrnet_common.latency import percentile, summarize_ms
from clrnet_common.runtime import (
    INFERENCE_PROJECT_ROOT,
    PROJECT_ROOT,
    resolve_culane_model_args,
)


def test_resolve_culane_model_args_uses_workspace_defaults():
    model_name, config, checkpoint = resolve_culane_model_args("dla34", None, None)

    assert model_name == "dla34"
    assert config == str(PROJECT_ROOT / "clrnet/configs/clrnet/clr_dla34_culane.py")
    assert checkpoint == str(PROJECT_ROOT / "weights/culane_dla34.pth")


def test_resolve_culane_model_args_preserves_explicit_paths():
    model_name, config, checkpoint = resolve_culane_model_args(
        "resnet18",
        "/tmp/custom.py",
        "/tmp/custom.pth",
    )

    assert model_name == "resnet18"
    assert config == "/tmp/custom.py"
    assert checkpoint == "/tmp/custom.pth"


def test_resolve_eval_output_dir_uses_explicit_path():
    output_dir = resolve_eval_output_dir(
        PROJECT_ROOT,
        "clrnet_inference",
        "dla34",
        "/tmp/predictions",
    )

    assert output_dir == Path("/tmp/predictions")


def test_resolve_eval_output_dir_uses_namespace_prefix():
    output_dir = resolve_eval_output_dir(
        PROJECT_ROOT,
        "clrnet_inference",
        "dla34",
        None,
    )

    assert output_dir.parent == INFERENCE_PROJECT_ROOT / "outputs/eval"
    assert output_dir.name.startswith("dla34_")


@pytest.mark.parametrize(
    ("pct", "expected"),
    [
        (50, 20.0),
        (90, 28.0),
        (100, 30.0),
    ],
)
def test_percentile_interpolates_values(pct, expected):
    assert percentile([10.0, 20.0, 30.0], pct) == pytest.approx(expected)


def test_summarize_ms_returns_expected_keys():
    summary = summarize_ms([1.0, 2.0, 3.0])

    assert summary["mean"] == pytest.approx(2.0)
    assert set(summary) == {"mean", "p50", "p90", "p95", "p99", "min", "max"}
