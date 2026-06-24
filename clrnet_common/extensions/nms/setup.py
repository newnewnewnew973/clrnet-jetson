from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

setup(
    name="clrnet_external_nms",
    ext_modules=[
        CUDAExtension(
            name="nms_impl",
            sources=[
                str(ROOT / "csrc/nms.cpp"),
                str(ROOT / "csrc/nms_kernel.cu"),
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
