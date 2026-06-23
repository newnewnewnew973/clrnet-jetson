"""Proxy package for using official CLRNet with local test-only overrides.

This package is intentionally named ``clrnet`` because the official CLRNet code
imports modules with paths such as ``clrnet.ops`` and ``clrnet.models``.

Import behavior:
1. Python first searches this local package:
   ``/home/newnew/workspace/clrnet_inference_test/clrnet``
2. If a requested submodule is not here, ``__path__`` below also lets Python
   search the official CLRNet package:
   ``/home/newnew/workspace/clrnet/clrnet``

Result:
- ``clrnet.ops.nms`` is loaded from this project, so it can use the external
  CUDA NMS extension under ``clrnet_inference_test/extensions/nms``.
- ``clrnet.models``, ``clrnet.utils``, and other unchanged modules are loaded
  from the official CLRNet source tree.

If the official CLRNet folder moves, update ``OFFICIAL_PACKAGE`` below so it
points to the new official Python package directory, the one that contains
``models/``, ``ops/``, and ``utils/``.
"""

from pathlib import Path


OFFICIAL_PACKAGE = Path(__file__).resolve().parents[2] / "clrnet" / "clrnet"
if OFFICIAL_PACKAGE.exists():
    __path__.append(str(OFFICIAL_PACKAGE))
