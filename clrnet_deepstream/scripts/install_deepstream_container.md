# DeepStream Container Install Notes

This machine is Jetson aarch64 with L4T R36.5. Native `deepstream-app` and the
GStreamer `nvinfer` plugin are not installed yet.

The least invasive install path is a DeepStream Jetson Docker container because
it does not change the host Python environment or the existing CLRNet project.

## Optional Docker Permission Setup

The current user cannot access Docker directly without `sudo`. This is workable,
but adding the user to the `docker` group makes repeated DeepStream runs easier.

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

Then log out and back in, or start a new SSH session.

Check:

```bash
docker info --format '{{json .Runtimes}}'
```

NVIDIA's DeepStream Docker documentation says Jetson containers are hosted on
NGC and must be pulled from `nvcr.io`. It may require NGC login:

```bash
docker login nvcr.io
```

For JetPack 6 / L4T R36.x, use the matching DeepStream 7.x Jetson image. The
exact image tag should be selected from the NGC DeepStream catalog for this
JetPack/L4T version.

This project currently uses:

```text
nvcr.io/nvidia/deepstream:7.1-samples-multiarch
```

Build the local image with extra multimedia decode plugins:

```bash
sudo docker build \
  -t clrnet-deepstream:7.1 \
  -f clrnet_deepstream/docker/Dockerfile \
  .
```

Verified locally:

```text
DeepStreamSDK 7.1.0
CUDA 12.6
TensorRT 10.3
```

After the image is available, mount this workspace:

```bash
sudo docker run -it --rm --runtime=nvidia --network=host --privileged \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
  -v /home/newnew/workspace:/workspace \
  -w /workspace \
  clrnet-deepstream:7.1
```

Inside the container:

```bash
deepstream-app --version-all
gst-inspect-1.0 nvinfer
python clrnet_deepstream/scripts/check_environment.py
```
