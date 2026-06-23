# Jetson Orin Nano 환경 및 설치 기록

이 문서는 Jetson Orin Nano를 새로 flash한 뒤, CLRNet DLA34 CULane
추론/평가 환경을 다시 만들기 위한 기록입니다.

목표:

- 공식 CLRNet 소스는 수정하지 않습니다.
- Jetson에서 필요한 외부 CUDA NMS만 별도 빌드합니다.
- TensorRT 최적화 전 PyTorch baseline을 재현할 수 있게 합니다.

## 1. 현재 확인된 환경

현재 장비에서 확인한 값입니다.

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

실제 Jetson GPU 접근 확인:

```text
torch.cuda.is_available() = True
torch.cuda.device_count() = 1
torch.cuda.get_device_name(0) = Orin
```

## 2. Python 패키지

추론에 사용:

```text
torch==2.11.0
numpy==1.26.4
cv2==4.8.0  # system OpenCV
tqdm==4.68.3
```

CULane 평가에 사용:

```text
scipy==1.15.3
shapely==2.1.2
p_tqdm==1.4.2
imgaug==0.4.0
scikit-image==0.25.2
scikit-learn==1.7.2
```

ONNX/TensorRT 작업용으로 현재 환경에 존재:

```text
onnx==1.21.0
onnxruntime==1.23.2
tensorrt==10.3.0
```

전역 패키지로 설치하지 않은 것:

```text
mmcv
torchvision
pycuda
```

현재 추론/평가 경로에서는 `mmcv`와 `torchvision`을
`clrnet_inference_test/` 내부의 최소 shim으로 처리합니다. 이 shim은 학습용이
아니며, 공식 패키지 전체를 대체하지 않습니다.

## 3. Workspace 구조

필요한 기본 구조:

```text
/home/newnew/workspace/
  clrnet/                    # 공식 CLRNet source, 수정 금지
  clrnet_inference_test/      # 이 프로젝트
  data/CULane/                # 압축 해제된 CULane dataset
  weights/culane_dla34.pth    # 학습된 CLRNet DLA34 checkpoint
```

이 프로젝트 내부에서는 CULane을 symlink로 연결합니다.

```text
clrnet_inference_test/data/CULane -> /home/newnew/workspace/data/CULane
```

CULane 폴더 구조:

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

현재 test split 개수:

```text
data/CULane/list/test.txt: 34680
```

## 4. 새 Jetson flash 이후 세팅 순서

JetPack/L4T, CUDA, TensorRT, system OpenCV가 있는 상태에서 시작한다고 가정합니다.

### 4.1 CUDA / PyTorch library path 설정

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/{디렉토리}local/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```

영구 적용하려면 `~/.bashrc`에 추가합니다.

PyTorch library path가 필요한 이유:

- 외부 CUDA NMS `.so`가 PyTorch C++ 라이브러리(`libc10.so` 등)에 link됩니다.
- 경로가 없으면 다음 에러가 날 수 있습니다.

```text
ImportError: libc10.so: cannot open shared object file
```

### 4.2 Python dependency 설치

PyTorch는 Jetson / Python 3.10 / CUDA 12.6에 맞는 NVIDIA 호환 wheel을 사용해야
합니다. 정확한 wheel URL은 JetPack/L4T 버전에 따라 달라질 수 있으므로, 실제
설치에 사용한 wheel은 별도 기록이 필요합니다.

나머지 Python package:

```bash
python -m pip install --user \
  numpy==1.26.4 \
  scipy shapely tqdm p_tqdm imgaug scikit-image scikit-learn \
  onnx onnxruntime
```

주의:

```text
pip install opencv-python
```

는 가급적 피합니다. Jetson에서는 system OpenCV
`/usr/lib/python3.10/dist-packages/cv2`를 쓰는 것이 안전합니다. pip OpenCV와
JetPack OpenCV가 섞이면 import/ABI 문제가 날 수 있습니다.

### 4.3 CULane data와 weight 준비

CULane 압축 해제 위치:

```text
/home/{디렉토리}/workspace/data/CULane
```

symlink 생성:

```bash
cd /home/{디렉토리}/workspace/clrnet_inference_test
mkdir -p data
ln -s /home/{디렉토리}/workspace/data/CULane data/CULane
```

checkpoint 위치:

```text
/home/{디렉토리}/workspace/weights/culane_dla34.pth
```

### 4.4 외부 CUDA NMS 빌드

공식 CLRNet의 NMS를 직접 수정하지 않습니다. 이 프로젝트의 외부 NMS를 빌드합니다.

```bash
cd /home/{디렉토리}/workspace/clrnet_inference_test/extensions/nms
TORCH_CUDA_ARCH_LIST=8.7 python setup.py build_ext --inplace
```

예상 결과:

```text
clrnet_inference_test/extensions/nms/nms_impl*.so
```

## 5. 검증 명령어

```bash
cd /home/{디렉토리}/workspace
```

import source 확인:

```bash
python clrnet_inference_test/tests/test_import_sources.py
```

NMS import 확인:

```bash
python clrnet_inference_test/tests/test_nms_import.py
```

CUDA NMS 계산 확인:

```bash
python clrnet_inference_test/tests/test_nms_compute.py
```

단일 이미지 추론:

```bash
python clrnet_inference_test/scripts/inference_culane_dla34.py \
  --image /home/{디렉토리}/workspace/clrnet_inference_test/assets/images/02015.jpg \
  --device cuda
```

전체 prediction 생성:

```bash
python clrnet_inference_test/scripts/evaluate_culane_dla34.py \
  --device cuda \
  --output-dir /home/{디렉토리}/workspace/clrnet_inference_test/outputs/eval/dla34_official
```

F1@0.5 계산:

```bash
python clrnet_inference_test/scripts/measure_culane_metric.py \
  --pred-dir /home/{디렉토리}/workspace/clrnet_inference_test/outputs/eval/dla34_official \
  --data-root /home/{디렉토리}/workspace/clrnet_inference_test/data/CULane \
  --output-json /home/{디렉토리}/workspace/clrnet_inference_test/outputs/eval/dla34_official_metric_0_5.json \
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

## 6. 현재 검증 결과

현재 workspace에서 확인한 결과:

```text
official CLRNet git tree clean
test_import_sources.py OK
test_nms_import.py OK
test_nms_compute.py OK
single-image inference OK
full CULane prediction files: 34680 / 34680
F1@0.5: 0.8047183655
```

F1@0.5 상세:

```text
TP        78419
FP        11593
FN        26467
Precision 0.8712060614
Recall    0.7476593635
F1        0.8047183655
```