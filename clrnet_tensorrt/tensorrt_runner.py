"""TensorRT helpers for CLRNet ONNX engines."""

from pathlib import Path

import tensorrt as trt
import torch


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def trt_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    """Map TensorRT tensor dtype to a torch dtype for device buffers."""
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int64:
        return torch.int64
    if dtype == trt.bool:
        return torch.bool
    raise TypeError(f"unsupported TensorRT dtype: {dtype}")


def build_engine_from_onnx(
    onnx_path: Path,
    engine_path: Path,
    workspace_gb: float = 1.0,
    fp16: bool = False,
    int8: bool = False,
    calibrator=None,
) -> None:
    """Build a TensorRT engine from an ONNX file."""
    if int8 and calibrator is None:
        raise ValueError("INT8 engine build requires a calibrator")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(onnx_path.read_bytes()):
        messages = [parser.get_error(i).desc() for i in range(parser.num_errors)]
        raise RuntimeError("failed to parse ONNX:\n" + "\n".join(messages))

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_gb * (1 << 30))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))


class TensorRTEngine:
    """Thin TensorRT v10 runner backed by torch CUDA tensors."""

    def __init__(self, engine_path: Path):
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"failed to load TensorRT engine: {engine_path}")
        self.runtime = runtime
        self.engine = engine
        self.context = engine.create_execution_context()
        self.input_names = []
        self.output_names = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        if len(self.input_names) != 1:
            raise RuntimeError(f"expected one input tensor, got {self.input_names}")

    @property
    def input_name(self) -> str:
        """Return the single TensorRT input tensor name."""
        return self.input_names[0]

    @property
    def input_dtype(self) -> torch.dtype:
        """Return the torch dtype required by the TensorRT engine input."""
        return trt_dtype_to_torch(self.engine.get_tensor_dtype(self.input_name))

    def infer(self, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run inference for one CUDA input tensor and return CUDA outputs."""
        if not input_tensor.is_cuda:
            raise ValueError("TensorRT input tensor must be on CUDA")
        if input_tensor.dtype != self.input_dtype:
            raise ValueError(
                "TensorRT input dtype mismatch: "
                f"engine expects {self.input_dtype}, got {input_tensor.dtype}"
            )
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        input_name = self.input_name
        self.context.set_input_shape(input_name, tuple(input_tensor.shape))
        self.context.set_tensor_address(input_name, input_tensor.data_ptr())

        outputs = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            output = torch.empty(shape, dtype=dtype, device=input_tensor.device)
            self.context.set_tensor_address(name, output.data_ptr())
            outputs[name] = output

        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return outputs
