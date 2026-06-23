import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

if "bool" not in np.__dict__:
    np.bool = bool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PROJECT_ROOT = PROJECT_ROOT / "clrnet_inference_test"
OFFICIAL_CLRNET_ROOT = PROJECT_ROOT / "clrnet"

if str(OFFICIAL_CLRNET_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CLRNET_ROOT))
if str(LOCAL_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PROJECT_ROOT))

from clrnet.utils.culane_metric import culane_metric, load_culane_data  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure CULane F1 from generated CLRNet prediction files with "
            "limited worker processes and progress logging."
        )
    )
    parser.add_argument(
        "--pred-dir",
        default=str(PROJECT_ROOT / "clrnet_inference_test/outputs/eval/dla34_official"),
        help="Directory containing generated CULane *.lines.txt prediction files.",
    )
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "clrnet_inference_test/data/CULane"),
        help="CULane dataset root containing images, GT .lines.txt files, and list/.",
    )
    parser.add_argument(
        "--list-path",
        default=None,
        help="CULane list file. Defaults to <data-root>/list/test.txt.",
    )
    parser.add_argument(
        "--output-json",
        default=str(PROJECT_ROOT / "clrnet_inference_test/outputs/eval/dla34_official_metric_0_5.json"),
        help="JSON file for final and intermediate metric results.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-interval", type=int, default=500)
    return parser.parse_args()


def calc_one(args):
    pred, anno, threshold = args
    result = culane_metric(
        pred,
        anno,
        iou_thresholds=[threshold],
        official=True,
        img_shape=(590, 1640, 3),
    )
    return result[threshold]


def make_payload(processed, total, tp, fp, fn, elapsed_sec):
    precision = float(tp) / (tp + fp) if tp else 0.0
    recall = float(tp) / (tp + fn) if tp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {
        "processed": processed,
        "total": total,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "elapsed_sec": elapsed_sec,
    }


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    data_root = Path(args.data_root)
    list_path = Path(args.list_path) if args.list_path else data_root / "list/test.txt"
    output_json = Path(args.output_json)

    if not pred_dir.exists():
        raise FileNotFoundError(f"prediction directory not found: {pred_dir}")
    if not data_root.exists():
        raise FileNotFoundError(f"CULane data root not found: {data_root}")
    if not list_path.exists():
        raise FileNotFoundError(f"CULane list file not found: {list_path}")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    output_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"pred_dir={pred_dir}", flush=True)
    print(f"data_root={data_root}", flush=True)
    print(f"list_path={list_path}", flush=True)
    print(f"iou_threshold={args.iou_threshold}", flush=True)
    print(f"workers={args.workers}", flush=True)

    print("loading_predictions", flush=True)
    predictions = load_culane_data(str(pred_dir), str(list_path))
    print(f"predictions={len(predictions)}", flush=True)

    print("loading_annotations", flush=True)
    annotations = load_culane_data(str(data_root), str(list_path))
    print(f"annotations={len(annotations)}", flush=True)

    if len(predictions) != len(annotations):
        raise RuntimeError(
            f"prediction/annotation count mismatch: {len(predictions)} vs {len(annotations)}"
        )

    total = len(predictions)
    tp = fp = fn = 0
    start = time.time()
    jobs = ((pred, anno, args.iou_threshold) for pred, anno in zip(predictions, annotations))

    print("metric_start", flush=True)
    with mp.Pool(processes=args.workers) as pool:
        for idx, values in enumerate(pool.imap(calc_one, jobs, chunksize=64), 1):
            tp += values[0]
            fp += values[1]
            fn += values[2]
            if idx % args.progress_interval == 0 or idx == total:
                payload = make_payload(idx, total, tp, fp, fn, time.time() - start)
                output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(json.dumps(payload), flush=True)

    print(f"metric_json={output_json}", flush=True)


if __name__ == "__main__":
    main()
