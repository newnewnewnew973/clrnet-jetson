"""Compatibility imports for existing CLRNet inference-test scripts."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clrnet_common.runtime import (  # noqa: E402,F401
    LOCAL_PROJECT_ROOT,
    OFFICIAL_CLRNET_ROOT,
    configure_import_paths,
    default_device_arg,
    ensure_numpy_bool_alias,
    load_checkpoint_for_inference,
    nms_build_message,
    resolve_culane_model_args,
    resolve_device,
)
