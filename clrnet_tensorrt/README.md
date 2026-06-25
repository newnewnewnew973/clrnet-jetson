# CLRNet TensorRT

This package contains the TensorRT path for the workspace-local CLRNet setup.
TensorRT-specific export, engine build, runtime, evaluation, and latency scripts
live here. Shared preprocessing, CULane helpers, metric code, checkpoint loading,
and lane decode/NMS helpers live in `clrnet_common/`.

Run every command from the workspace root.

## Inputs

Expected local files:

```text
clrnet/
data/CULane/
clrnet_inference/data/CULane -> ../../data/CULane
weights/culane_dla34.pth
clrnet_common/extensions/nms/nms_impl*.so
```

Build the shared CUDA NMS extension before TensorRT inference or evaluation:

```bash
cd clrnet_common/extensions/nms
TORCH_CUDA_ARCH_LIST=8.7 python setup.py build_ext --inplace
cd ../../..
```

## Export ONNX

```bash
python clrnet_tensorrt/scripts/export_onnx.py \
  --model dla34 \
  --device cuda \
  --output clrnet_tensorrt/outputs/onnx/clrnet_dla34.onnx
```

Output:

```text
clrnet_tensorrt/outputs/onnx/clrnet_dla34.onnx
```

The export code patches CLRNet's head forward only in memory. The goal is to
build TensorRT-friendly prediction tensors without modifying upstream `clrnet/`.

## Build FP16 Engine

```bash
python clrnet_tensorrt/scripts/build_engine.py \
  --model dla34 \
  --onnx clrnet_tensorrt/outputs/onnx/clrnet_dla34.onnx \
  --engine clrnet_tensorrt/outputs/engine/clrnet_dla34_fp16.engine \
  --fp16 \
  --workspace-gb 1
```

Output:

```text
clrnet_tensorrt/outputs/engine/clrnet_dla34_fp16.engine
```

Latest build result:

```text
fp16=True
int8=False
engine_generation_sec=211.342
engine_size_mb=32.59
```

## Single Image Inference

```bash
python clrnet_tensorrt/scripts/inference_culane.py \
  --model dla34 \
  --engine clrnet_tensorrt/outputs/engine/clrnet_dla34_fp16.engine \
  --image data/CULane/driver_100_30frame/05251517_0433.MP4/00000.jpg \
  --output-dir clrnet_tensorrt/outputs/single_image/full_run \
  --device cuda
```

Latest result:

```text
lanes=3
visualization=clrnet_tensorrt/outputs/single_image/full_run/00000_clrnet_dla34_trt.jpg
culane_output=clrnet_tensorrt/outputs/single_image/full_run/00000.lines.txt
```

## Full CULane Prediction

Generate prediction files for the full CULane test split, 34,680 samples:

```bash
python clrnet_tensorrt/scripts/evaluate_culane.py \
  --model dla34 \
  --engine clrnet_tensorrt/outputs/engine/clrnet_dla34_fp16.engine \
  --data-root clrnet_inference/data/CULane \
  --output-dir clrnet_tensorrt/outputs/eval/dla34_fp16_full \
  --device cuda
```

Latest result:

```text
dataset_total=34680
evaluated_samples=34680
predictions_written=34680
prediction_files=34680
```

Verify file count:

```bash
find clrnet_tensorrt/outputs/eval/dla34_fp16_full -name '*.lines.txt' | wc -l
```

## CULane Metric

```bash
python clrnet_tensorrt/scripts/measure_culane_metric.py \
  --model dla34 \
  --pred-dir clrnet_tensorrt/outputs/eval/dla34_fp16_full \
  --data-root clrnet_inference/data/CULane \
  --output-json clrnet_tensorrt/outputs/eval/dla34_fp16_full_metric_0_5.json \
  --workers 4 \
  --progress-interval 500
```

Latest full CULane result:

```text
processed 34680
TP        78416
FP        11593
FN        26470
Precision 0.8712017687
Recall    0.7476307610
F1@0.5    0.8046999666
```

Result file:

```text
clrnet_tensorrt/outputs/eval/dla34_fp16_full_metric_0_5.json
```

## Latency

The latest latency run uses 1,000 CULane samples:

```bash
python clrnet_tensorrt/scripts/measure_tensorrt_latency.py \
  --model dla34 \
  --data-root clrnet_inference/data/CULane \
  --precision fp16 \
  --fp16-engine clrnet_tensorrt/outputs/engine/clrnet_dla34_fp16.engine \
  --output-dir clrnet_tensorrt/outputs/latency/tensorrt_dla34_1000 \
  --device cuda \
  --limit 1000 \
  --warmup 100 \
  --block-warmup 10
```

Latest result:

```text
samples=1000
fp16_engine_input_dtype=torch.float32
fp16_fps_pure_engine_cuda_event=128.67
fp16_latency_pure_engine_cuda_event_mean_ms=7.77
fp16_fps_cuda_event_forward=77.03
fp16_latency_cuda_event_forward_mean_ms=12.98
fp16_latency_cuda_event_forward_p95_ms=13.20
fp16_latency_cuda_event_forward_p99_ms=13.23
fp16_fps_continuous_e2e=14.84
```

Output:

```text
clrnet_tensorrt/outputs/latency/tensorrt_dla34_1000/tensorrt_latency_dla34.json
clrnet_tensorrt/outputs/latency/tensorrt_dla34_1000/TENSORRT_LATENCY.md
```

Pure engine timing measures only TensorRT execution with preallocated buffers and
CUDA events. This is the closest number to pure model FPS. `fps_cuda_event_forward`
uses the normal `TensorRTEngine.infer()` wrapper, so it includes runner overhead
around TensorRT execution. Continuous E2E timing includes dataset access,
preprocessing, host-to-device copy, TensorRT forward, and shared CLRNet lane
decode/NMS.

## Comparison With PyTorch

The latest accuracy runs used the full CULane test split. Latency runs used
1,000 samples.

| Runtime | Samples | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| PyTorch DLA34 | 34,680 | 0.871190 | 0.747678 | 0.804722 |
| TensorRT DLA34 FP16 | 34,680 | 0.871202 | 0.747631 | 0.804700 |

| Runtime | Samples | E2E FPS | Pure model FPS |
| --- | ---: | ---: | ---: |
| PyTorch DLA34 | 1,000 | 12.72 | 31.39 |
| TensorRT DLA34 FP16 | 1,000 | 14.84 | 128.67 |

The FP16 TensorRT metric is effectively equal to the PyTorch baseline for this
run. The larger pure model FPS improvement is partially hidden in E2E FPS because
preprocessing and lane decode/NMS are still shared Python-side work. The TensorRT
runner-wrapper forward FPS from the same run was 77.03.

## Verification

Latest verification commands:

```bash
python -m compileall -q clrnet_common clrnet_inference clrnet_tensorrt
python -m pytest clrnet_inference/tests
```

Latest result:

```text
compileall passed
8 passed in clrnet_inference/tests
```
