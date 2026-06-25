"""Minimal JSON/YAML helpers for the local MMCV compatibility layer.

Only the formats needed by the CLRNet inference/evaluation path are supported.
Unsupported formats fail explicitly instead of pretending to match full MMCV.
"""

import json
from pathlib import Path


def load(filename):
    path = Path(filename)
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix == ".json":
            return json.load(f)
        if suffix in (".yml", ".yaml"):
            import yaml

            return yaml.safe_load(f)
    raise IOError(f"Unsupported file type: {suffix}")


def dump(obj, file=None, file_format=None):
    if file is None:
        if file_format != "json":
            raise IOError(f"Unsupported file format: {file_format}")
        return json.dumps(obj, indent=2)

    path = Path(file)
    suffix = path.suffix.lower()
    with path.open("w", encoding="utf-8") as f:
        if suffix == ".json":
            json.dump(obj, f, indent=2)
            return None
    raise IOError(f"Unsupported file type: {suffix}")
