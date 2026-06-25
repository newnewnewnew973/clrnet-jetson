"""CULane dataset/output helpers shared by PyTorch and TensorRT scripts."""

from datetime import datetime
from pathlib import Path
import pickle


def build_culane_dataset(build_dataset, cfg, data_root: Path):
    """Build the official CULane test dataset with the repo-local data root."""
    cfg.dataset.test.data_root = str(data_root)
    remove_stale_culane_cache(data_root, cfg.dataset.test.split)
    return build_dataset(cfg.dataset.test, cfg)


def remove_stale_culane_cache(data_root: Path, split: str) -> None:
    """Remove official CLRNet's CULane cache when it points at another root."""
    cache_path = Path("cache") / f"culane_{split}.pkl"
    if not cache_path.exists():
        return

    try:
        with cache_path.open("rb") as cache_file:
            data_infos = pickle.load(cache_file)
    except (OSError, pickle.PickleError, EOFError):
        cache_path.unlink(missing_ok=True)
        return

    if not data_infos:
        return

    expected_root = data_root.resolve()
    cached_path = Path(data_infos[0].get("img_path", ""))
    try:
        cached_path.resolve().relative_to(expected_root)
    except (OSError, ValueError):
        cache_path.unlink(missing_ok=True)


def resolve_eval_output_dir(
    project_root: Path,
    namespace: str,
    model_name: str,
    output_dir: str | None,
) -> Path:
    """Use an explicit output dir as-is, otherwise isolate eval runs by timestamp."""
    if output_dir:
        return Path(output_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / f"{namespace}/outputs/eval/{model_name}_{run_id}"


def write_culane_prediction(dataset, idx: int, pred, output_dir: Path) -> None:
    """Write one prediction using the same relative path layout as CULane."""
    relative_img = Path(dataset.data_infos[idx]["img_name"])
    pred_dir = output_dir / relative_img.parent
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_file = pred_dir / f"{relative_img.stem}.lines.txt"
    pred_file.write_text(dataset.get_prediction_string(pred), encoding="utf-8")
