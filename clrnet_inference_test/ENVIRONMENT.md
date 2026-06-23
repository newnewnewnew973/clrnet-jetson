# Jetson Orin Nano Environment

CLRNet DLA34 CULane 추론/평가 환경을 Jetson Orin Nano에서 재현하기 위한 기록입니다.
경로는 모두 clone한 repository root 기준입니다.

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
├── clrnet_inference_test/           # this project
├── data/CULane/                     # CULane dataset
└── weights/culane_dla34.pth         # DLA34 checkpoint
```

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

### 6. CUDA NMS build

공식 CLRNet NMS 소스는 수정하지 않습니다. 이 프로젝트의 외부 NMS extension만
빌드합니다.

```bash
cd clrnet_inference_test/extensions/nms
TORCH_CUDA_ARCH_LIST=8.7 python setup.py build_ext --inplace
cd ../../..
```

생성 파일:

```text
clrnet_inference_test/extensions/nms/nms_impl*.so
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
python clrnet_inference_test/scripts/inference_culane_dla34.py \
  --image /path/to/culane/image.jpg \
  --device cuda
```

전체 prediction 생성:

```bash
rm -rf clrnet_inference_test/outputs/eval/dla34_official

python clrnet_inference_test/scripts/evaluate_culane_dla34.py \
  --device cuda \
  --output-dir clrnet_inference_test/outputs/eval/dla34_official
```

prediction 개수 확인:

```bash
find clrnet_inference_test/outputs/eval/dla34_official -name '*.lines.txt' | wc -l
# 34680
```

F1@0.5 계산:

```bash
python clrnet_inference_test/scripts/measure_culane_metric.py \
  --pred-dir clrnet_inference_test/outputs/eval/dla34_official \
  --data-root clrnet_inference_test/data/CULane \
  --output-json clrnet_inference_test/outputs/eval/dla34_official_metric_0_5.json \
  --iou-threshold 0.5 \
  --workers 4 \
  --progress-interval 500
```

PyTorch latency/FPS 측정:

```bash
python clrnet_inference_test/scripts/measure_pytorch_latency_dla34.py \
  --device cuda \
  --limit 100 \
  --warmup 10
```

## Current Baseline

현재 Jetson Orin Nano에서 확인한 값입니다.

```text
test_import_sources.py OK
test_nms_import.py OK
test_nms_compute.py OK
single-image CUDA inference OK
full CULane prediction files 34680 / 34680
```

CULane F1@0.5:

```text
TP        78419
FP        11593
FN        26467
Precision 0.8712060614
Recall    0.7476593635
F1        0.8047183655
```

PyTorch latency, 100 samples / 10 warmup:

```text
E2E FPS              11.88
E2E mean latency     84.20 ms
forward mean latency 48.70 ms
```

위 latency는 Jetson Orin Nano에서 PyTorch eager mode로 측정한 end-to-end 기준입니다.
논문 throughput 수치와 직접 비교하지 않습니다.
