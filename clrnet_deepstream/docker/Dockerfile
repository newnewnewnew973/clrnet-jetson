FROM nvcr.io/nvidia/deepstream:7.1-samples-multiarch

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gstreamer1.0-libav \
      gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad \
      gstreamer1.0-plugins-ugly \
      kmod \
      libavcodec58 \
      libavformat58 \
      libavutil56 \
      libflac8 \
      libmjpegutils-2.1-0 \
      libmp3lame0 \
      libmpg123-0 \
      libpython3.10 \
      libvpx7 \
      libx264-163 \
      libx265-199 \
      libxvidcore4 \
      python3-numpy \
      python3-pip \
      python3-gi \
      python3-gst-1.0 && \
    apt-get install -y --no-install-recommends --reinstall \
      libavcodec58 \
      libavformat58 \
      libavutil56 \
      libflac8 \
      libmjpegutils-2.1-0 \
      libmp3lame0 \
      libmpg123-0 \
      libvpx7 \
      libx264-163 \
      libx265-199 \
      libxvidcore4 && \
    ldconfig && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
    https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_aarch64.whl
