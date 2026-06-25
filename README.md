# CLRNet Workspace

This workspace keeps the upstream CLRNet checkout read-only and adds local
PyTorch inference, TensorRT, and DeepStream deployment packages around it.

## Structure

- `clrnet/`: upstream CLRNet source. Do not edit this directory.
- `clrnet_common/`: shared runtime, CULane, image, metric, latency, and NMS code.
- `clrnet_inference/`: PyTorch inference, evaluation, metric, latency, and tests.
- `clrnet_tensorrt/`: ONNX export, TensorRT engine build, inference, evaluation,
  metric, and latency.
- `clrnet_deepstream/`: Jetson DeepStream deployment pipeline with hardware
  decode, TensorRT `nvinfer`, lane overlay, and full-frame video output.

The important design decision is that backend-specific scripts stay in their
own package, while duplicated behavior lives in `clrnet_common/`. The upstream
`clrnet/` directory remains unchanged.

## Required Inputs

```text
data/CULane/
weights/culane_dla34.pth
clrnet/
```

The default scripts expect this dataset link:

```bash
mkdir -p clrnet_inference/data
ln -s ../../data/CULane clrnet_inference/data/CULane
```

Build the shared CUDA NMS extension before real inference:

```bash
cd clrnet_common/extensions/nms
TORCH_CUDA_ARCH_LIST=8.7 python setup.py build_ext --inplace
cd ../../..
```

## Main Commands

PyTorch:

```bash
python clrnet_inference/scripts/inference_culane.py --model dla34 --device cuda
python clrnet_inference/scripts/evaluate_culane.py --model dla34 --device cuda --batch-size 4
python clrnet_inference/scripts/measure_culane_metric.py --model dla34 --pred-dir clrnet_inference/outputs/eval/dla34_full --workers 4
python clrnet_inference/scripts/measure_pytorch_latency.py --model dla34 --device cuda --limit 1000 --warmup 100 --block-warmup 10
```

TensorRT:

```bash
python clrnet_tensorrt/scripts/export_onnx.py --model dla34 --device cuda --output clrnet_tensorrt/outputs/onnx/clrnet_dla34.onnx
python clrnet_tensorrt/scripts/build_engine.py --model dla34 --onnx clrnet_tensorrt/outputs/onnx/clrnet_dla34.onnx --engine clrnet_tensorrt/outputs/engine/clrnet_dla34_fp16.engine --fp16 --workspace-gb 1
python clrnet_tensorrt/scripts/inference_culane.py --model dla34 --engine clrnet_tensorrt/outputs/engine/clrnet_dla34_fp16.engine --device cuda
python clrnet_tensorrt/scripts/evaluate_culane.py --model dla34 --engine clrnet_tensorrt/outputs/engine/clrnet_dla34_fp16.engine --data-root clrnet_inference/data/CULane --output-dir clrnet_tensorrt/outputs/eval/dla34_fp16_full --device cuda
python clrnet_tensorrt/scripts/measure_culane_metric.py --model dla34 --pred-dir clrnet_tensorrt/outputs/eval/dla34_fp16_full --data-root clrnet_inference/data/CULane --workers 4
python clrnet_tensorrt/scripts/measure_tensorrt_latency.py --model dla34 --data-root clrnet_inference/data/CULane --precision fp16 --fp16-engine clrnet_tensorrt/outputs/engine/clrnet_dla34_fp16.engine --device cuda --limit 1000 --warmup 100 --block-warmup 10
```

DeepStream:

```bash
sudo docker build -t clrnet-deepstream:7.1 -f clrnet_deepstream/docker/Dockerfile .

cp clrnet_tensorrt/outputs/onnx/clrnet_dla34.onnx \
  clrnet_deepstream/outputs/clrnet_dla34.onnx

python clrnet_deepstream/scripts/prepare_video_example.py \
  --overwrite \
  --codec h264 \
  --output clrnet_deepstream/outputs/video_example_full.h264

sudo docker run --rm --runtime=nvidia --network=host --privileged \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
  -v /home/newnew/workspace:/workspace \
  -w /workspace \
  clrnet-deepstream:7.1 \
  python3 clrnet_deepstream/scripts/run_lane_overlay.py \
    --compose-full-frame \
    --decoder hardware \
    --input clrnet_deepstream/outputs/video_example_full.h264 \
    --output clrnet_deepstream/outputs/deepstream_lanes_full_compositor.mp4 \
    --nvinfer-config clrnet_deepstream/configs/nvinfer_clrnet.txt
```

More details are in `clrnet_inference/README.md`,
`clrnet_tensorrt/README.md`, and `clrnet_deepstream/README.md`.

## Latest Verification

The latest full check was run after deleting previous output artifacts. Accuracy
evaluation used the full CULane test split, 34,680 samples. Latency used 1,000
samples.

| Runtime | Samples | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| PyTorch DLA34 | 34,680 | 0.871190 | 0.747678 | 0.804722 |
| TensorRT DLA34 FP16 | 34,680 | 0.871202 | 0.747631 | 0.804700 |

Latency summary:

| Runtime | Samples | E2E FPS | Pure model FPS |
| --- | ---: | ---: | ---: |
| PyTorch DLA34 | 1,000 | 12.72 | 31.39 |
| TensorRT DLA34 FP16 | 1,000 | 14.84 | 128.67 |

DeepStream deployment verification:

| Runtime | Input | Output | Frames | Notes |
| --- | --- | --- | ---: | --- |
| DeepStream 7.1 + TensorRT FP16 | H.264 file | full-frame MP4 | 5,400 | `nvv4l2decoder`, `nvinfer`, `nvdsosd`, `nvcompositor` |

The DeepStream output keeps the original 1640x590 frame. The model branch uses
the same CULane crop as CLRNet inference, then the overlay branch is composited
back onto the original frame inside DeepStream.

Generated artifacts:

```text
clrnet_inference/outputs/eval/dla34_full/
clrnet_inference/outputs/eval/dla34_full_metric_0_5.json
clrnet_inference/outputs/latency/pytorch_dla34_1000/
clrnet_tensorrt/outputs/onnx/clrnet_dla34.onnx
clrnet_tensorrt/outputs/engine/clrnet_dla34_fp16.engine
clrnet_tensorrt/outputs/eval/dla34_fp16_full/
clrnet_tensorrt/outputs/eval/dla34_fp16_full_metric_0_5.json
clrnet_tensorrt/outputs/latency/tensorrt_dla34_1000/
clrnet_deepstream/outputs/video_example_full.h264
clrnet_deepstream/outputs/deepstream_lanes_full_compositor.mp4
```

## Tests

```bash
python -m compileall -q clrnet_common clrnet_inference clrnet_tensorrt clrnet_deepstream/scripts
python -m pytest clrnet_inference/tests
```

Latest result:

```text
compileall passed
8 passed in clrnet_inference/tests
```

Running `pytest` over the whole package directories is not useful here because
test discovery can traverse runtime packages and stall during import collection.
Use `clrnet_inference/tests` for the current regression tests.
