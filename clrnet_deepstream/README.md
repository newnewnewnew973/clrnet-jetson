# CLRNet DeepStream

DeepStream deployment workspace for running the CLRNet TensorRT model on Jetson
Orin Nano Super.

The target runtime path is GPU based:

```text
H.264 file
  -> nvv4l2decoder
  -> nvvideoconvert / NVMM
  -> tee
  -> crop branch for CLRNet inference
  -> nvstreammux
  -> nvinfer / TensorRT FP16
  -> raw tensor metadata
  -> CLRNet lane postprocess
  -> nvdsosd lane overlay
  -> nvcompositor with original full-frame branch
  -> encoded output video
```

The final user-facing goal is:

```text
input video -> CLRNet lane recognition through DeepStream -> output video
```

CPU decode is not the deployment target. The script still supports it as a
diagnostic fallback, but the verified path uses Jetson hardware decode.

## Current Status

Verified on the current Jetson environment:

- JetPack 6 family / L4T R36.5
- DeepStream SDK 7.1 container
- CUDA 12.6 in the DeepStream container
- TensorRT 10.3 in the DeepStream container
- `nvinfer`, `nvstreammux`, `nvvideoconvert`, and `nvv4l2decoder` are available
- `nvv4l2decoder` reaches EOS inside the container after adding `kmod`
- CLRNet ONNX is copied into `clrnet_deepstream/outputs`
- DeepStream generated a TensorRT FP16 engine
- DeepStream produced a full-frame lane-overlay output video with hardware decode

```text
clrnet_deepstream/outputs/clrnet_dla34.onnx_b1_gpu0_fp16.engine
clrnet_deepstream/outputs/deepstream_lanes_full_compositor.mp4
```

The full `data/video_example.zip` run was verified as:

```text
input_frames=5400
output_frames=5400
fps=30.0
width=1640
height=590
overlay_frames=5400
overlay_lanes=19431
overlay_segments=1075177
```

The model branch still applies the same CULane crop as CLRNet inference
(`top=270`, `height=320`). The displayed output is restored inside the
DeepStream pipeline with `tee` and `nvcompositor`, so the final video keeps the
original 1640x590 frame.

The earlier `nvv4l2decoder` failure was caused by the container missing `lsmod`
from `kmod`. Jetson's NVIDIA GStreamer decoder calls that utility while
initializing V4L2 decode. Without it, decode failed with:

```text
S_EXT_CTRLS for CUDA_GPU_ID failed
```

Adding `kmod` to the image fixed the hardware decode path.

## Files

```text
clrnet_deepstream/
  configs/nvinfer_clrnet.txt
  docker/Dockerfile
  scripts/check_environment.py
  scripts/clrnet_postprocess.py
  scripts/prepare_video_example.py
  scripts/run_deepstream_file.py
  scripts/run_lane_overlay.py
  scripts/install_deepstream_container.md
  outputs/
```

`outputs/` is ignored by git. It is created as a local runtime directory for
copied ONNX files, generated TensorRT engines, encoded input videos, and output
videos.

## Build Container

Use the JetPack 6 compatible DeepStream 7.1 image:

```bash
sudo docker build \
  -t clrnet-deepstream:7.1 \
  -f clrnet_deepstream/docker/Dockerfile \
  .
```

Check DeepStream:

```bash
sudo docker run --rm --runtime=nvidia --network=host --privileged \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
  -v /home/newnew/workspace:/workspace \
  -w /workspace \
  clrnet-deepstream:7.1 \
  deepstream-app --version-all
```

## Prepare Inputs

Copy the ONNX model from the TensorRT workspace:

```bash
cp clrnet_tensorrt/outputs/onnx/clrnet_dla34.onnx \
  clrnet_deepstream/outputs/clrnet_dla34.onnx
```

Create a full raw H.264 input from `data/video_example.zip`:

```bash
python clrnet_deepstream/scripts/prepare_video_example.py \
  --overwrite \
  --codec h264 \
  --output clrnet_deepstream/outputs/video_example_full.h264
```

## Run Inference Smoke Pipeline

Inside the DeepStream container:

```bash
python clrnet_deepstream/scripts/run_deepstream_file.py \
  --input clrnet_deepstream/outputs/video_example_300.h264 \
  --nvinfer-config clrnet_deepstream/configs/nvinfer_clrnet.txt \
  --sink file \
  --output clrnet_deepstream/outputs/deepstream_output.mp4
```

Or from the host:

```bash
sudo docker run --rm --runtime=nvidia --network=host --privileged \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
  -v /home/newnew/workspace:/workspace \
  -w /workspace \
  clrnet-deepstream:7.1 \
  python clrnet_deepstream/scripts/run_deepstream_file.py \
    --input clrnet_deepstream/outputs/video_example_300.h264 \
    --nvinfer-config clrnet_deepstream/configs/nvinfer_clrnet.txt \
    --sink file \
    --output clrnet_deepstream/outputs/deepstream_output.mp4
```

This smoke command verifies that the decode path and `nvinfer` can run. It does
not draw lanes. Use the lane overlay pipeline below for the final demo output.

## Run Lane Overlay Pipeline

This is the main DeepStream result path:

```bash
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

For diagnostic fallback, switch only the decoder option:

```bash
--decoder software
```

The script performs:

```text
H.264 input
  -> hardware decode
  -> tee
  -> branch A: original full frame
  -> branch B: crop top 270px -> nvstreammux -> nvinfer TensorRT FP16
  -> branch B: NvDsInferTensorMeta -> CLRNet lane decode -> nvdsosd overlay
  -> nvcompositor: branch B over branch A at y=270
  -> full-frame MP4 output
```

The model branch intentionally crops the top 270 pixels because that is how the
CLRNet CULane model was trained and exported. The final output is restored to
the original frame size by compositing the lane-overlay branch over the original
decoded frame inside DeepStream.

## Environment Check

```bash
sudo docker run --rm --runtime=nvidia --network=host --privileged \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
  -v /home/newnew/workspace:/workspace \
  -w /workspace \
  clrnet-deepstream:7.1 \
  python clrnet_deepstream/scripts/check_environment.py
```
