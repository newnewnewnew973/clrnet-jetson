# Jetson Orin Nano Environment

CLRNet CULane 추론/평가 환경을 Jetson Orin Nano에서 재현하기 위한 기록입니다.
기본 baseline은 DLA34 checkpoint이며, 경로는 모두 clone한 repository root 기준입니다.

```bash
git clone git@github.com:newnewnewnew973/clrnet-jetson.git
cd clrnet-jetson
```

## Verified System

현재 장비에서 확인한 환경입니다.

```text
Board              NVIDIA Jetson Orin Nano
OS                 Ubuntu 22.04.5 LTS
Kernel/L4T         5.15.185-tegra / nvidia-l4t-core 36.5.0
Python             3.10.12
CUDA toolkit       12.6 / nvcc V12.6.68
TensorRT           10.3.0
PyTorch            2.11.0
PyTorch CUDA       12.6
OpenCV Python      4.8.0
NumPy              1.26.4
jetson-stats       4.3.2
Power mode         MAXN_SUPER
```

GPU 확인값:

```text
torch.cuda.is_available() = True
torch.cuda.device_count() = 1
torch.cuda.get_device_name(0) = Orin
```

## Python Packages

추론 경로:

```text
torch==2.11.0
numpy==1.26.4
cv2==4.8.0  # system OpenCV
tqdm==4.68.3
```

CULane metric:

```text
scipy==1.15.3
shapely==2.1.2
p_tqdm==1.4.2
imgaug==0.4.0
scikit-image==0.25.2
scikit-learn==1.7.2
```

ONNX/TensorRT 작업용:

```text
onnx==1.21.0
onnxruntime==1.23.2
tensorrt==10.3.0
```

전역 패키지로 설치하지 않은 항목:

```text
mmcv
torchvision
pycuda
```

현재 추론/평가 경로에서는 `clrnet_inference_test/` 내부 shim으로 `mmcv`와
`torchvision` import를 처리합니다. 이 shim은 학습용이 아닙니다.

## Required Repository State

clone 직후 다음 구조를 맞춥니다.

```text
.
├── clrnet/                         # official CLRNet source
├── clrnet_common/                  # shared runtime, latency, and CUDA NMS extension
├── clrnet_inference_test/           # this project
├── data/CULane/                     # CULane dataset
└── weights/culane_dla34.pth         # DLA34 checkpoint
```

`clrnet_common/`과 `clrnet_inference_test/`는 이 repository에 포함합니다.
`clrnet/`, `data/`, `weights/`는 git에 포함하지 않습니다.

## Setup

### 1. CUDA / PyTorch library path

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```

필요하면 위 설정을 `~/.bashrc`에 추가합니다.

PyTorch library path가 없으면 외부 CUDA NMS import 시 다음 오류가 날 수 있습니다.

```text
ImportError: libc10.so: cannot open shared object file
```

### 2. Official CLRNet source

repository root에서 공식 CLRNet을 checkout합니다.

```bash
git clone https://github.com/Turoad/CLRNet.git clrnet
```

이 프로젝트의 proxy package는 `./clrnet/clrnet`을 공식 package 위치로 사용합니다.

### 3. Python dependencies

PyTorch는 Jetson / Python 3.10 / CUDA 12.6에 맞는 NVIDIA wheel을 사용합니다.
wheel URL은 JetPack/L4T 버전에 따라 달라질 수 있으므로 장비 환경에 맞춰 설치합니다.

나머지 dependency:

```bash
python -m pip install --user \
  numpy==1.26.4 \
  scipy shapely tqdm p_tqdm imgaug scikit-image scikit-learn \
  onnx onnxruntime
```

Jetson에서는 `opencv-python` wheel 설치를 피합니다. system OpenCV
`/usr/lib/python3.10/dist-packages/cv2`를 사용합니다. pip OpenCV와 JetPack OpenCV가
섞이면 import/ABI 문제가 생길 수 있습니다.

### 4. CULane dataset

dataset은 다음 위치에 둡니다.

```text
data/CULane/
```

필수 구조:

```text
data/CULane/
  driver_100_30frame/
  driver_161_90frame/
  driver_182_30frame/
  driver_193_90frame/
  driver_23_30frame/
  driver_37_30frame/
  laneseg_label_w16/
  laneseg_label_w16_test/
  list/
```

`clrnet_inference_test/data/CULane` symlink를 생성합니다.

```bash
mkdir -p clrnet_inference_test/data
ln -s ../../data/CULane clrnet_inference_test/data/CULane
```

test split 개수 확인:

```bash
wc -l data/CULane/list/test.txt
# 34680 data/CULane/list/test.txt
```

### 5. Checkpoint

checkpoint는 다음 위치에 둡니다.

```text
weights/culane_dla34.pth
```

DLA34 외 모델을 사용할 경우 `--model`을 바꿉니다. 기본 config는
`clrnet/configs/clrnet/clr_<model>_culane.py`, 기본 checkpoint는
`weights/culane_<model>.pth`로 해석합니다. 위치가 다르면 `--config`,
`--checkpoint`로 개별 override합니다.

### 6. CUDA NMS build

공식 CLRNet NMS 소스는 수정하지 않습니다. 이 프로젝트의 외부 NMS extension만
빌드합니다.

```bash
cd clrnet_common/extensions/nms
TORCH_CUDA_ARCH_LIST=8.7 python setup.py build_ext --inplace
cd ../../..
```

생성 파일:

```text
clrnet_common/extensions/nms/nms_impl*.so
```

## Validation Commands

repository root에서 실행합니다.

```bash
python clrnet_inference_test/tests/test_import_sources.py
python clrnet_inference_test/tests/test_nms_import.py
python clrnet_inference_test/tests/test_nms_compute.py
```

단일 이미지 추론:

```bash
python clrnet_inference_test/scripts/inference_culane.py \
  --image /path/to/culane/image.jpg \
  --model dla34 \
  --device cuda
```

전체 prediction 생성:

```bash
python clrnet_inference_test/scripts/evaluate_culane.py \
  --model dla34 \
  --device cuda \
  --batch-size 8
```

`--output-dir`을 생략하면 실행마다
`clrnet_inference_test/outputs/eval/dla34_<YYYYmmdd_HHMMSS>` 경로를 생성합니다.

prediction 개수 확인:

```bash
find clrnet_inference_test/outputs/eval/dla34_<YYYYmmdd_HHMMSS> -name '*.lines.txt' | wc -l
# 34680
```

F1@0.5 계산:

```bash
python clrnet_inference_test/scripts/measure_culane_metric.py \
  --model dla34 \
  --pred-dir clrnet_inference_test/outputs/eval/dla34_<YYYYmmdd_HHMMSS> \
  --data-root clrnet_inference_test/data/CULane \
  --iou-threshold 0.5 \
  --workers 4 \
  --progress-interval 500
```

PyTorch latency/FPS 측정:

```bash
python clrnet_inference_test/scripts/measure_pytorch_latency.py \
  --model dla34 \
  --device cuda \
  --limit 100 \
  --warmup 100 \
  --block-warmup 10
```

## Current Baseline

현재 Jetson Orin Nano에서 확인한 값입니다.

```text
test_import_sources.py OK
test_nms_import.py OK
test_nms_compute.py OK
single-image CUDA inference OK
full CULane prediction files 34680 / 34680
measure_culane_metric.py OK
```

CULane F1@0.5:

```text
TP        78424
FP        11594
FN        26462
Precision 0.8712035371
Recall    0.7477070343
F1        0.8047449001
```

PyTorch latency smoke, 4 samples / 4 warmup / 2 block warmup:

```text
continuous E2E FPS             17.13
CUDA-event forward FPS         28.76
CUDA-event forward mean        34.77 ms
forward + postprocess mean     39.68 ms
```

위 latency smoke는 커밋 전 동작 확인용 짧은 실행입니다. 성능 비교용 baseline은
`--limit 100 --warmup 100 --block-warmup 10`처럼 충분한 sample과 warmup으로 다시
측정합니다. 논문 throughput 수치와 직접 비교하지 않습니다.
