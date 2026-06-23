# CLRNet DLA34 CULane 추론/평가 베이스라인

이 폴더는 Jetson Orin Nano에서 공식 CLRNet DLA34 CULane 모델을 실행하기
위한 **추론/평가용 래퍼 프로젝트**입니다.

원칙:

- 공식 CLRNet 소스(`./clrnet`)는 수정하지 않습니다.
- PyTorch 2.x / Jetson 환경에서 필요한 CUDA NMS만 외부 확장으로 따로 빌드합니다.
- TensorRT 최적화 전 기준이 되는 PyTorch 정확도와 속도를 재현 가능하게 남깁니다.

환경과 설치 절차는 [ENVIRONMENT.md](ENVIRONMENT.md)에 정리되어 있습니다.

아래 명령어는 GitHub에서 이 저장소를 clone한 뒤 repo root에서 실행하는 것을
기준으로 작성했습니다.

```bash
git clone git@github.com:newnewnewnew973/clrnet-jetson.git
cd clrnet-jetson
```

## 폴더 역할

```text
clrnet_inference_test/
  scripts/
    inference_culane_dla34.py         # 단일 이미지 추론
    evaluate_culane_dla34.py          # 전체 CULane 추론 및 prediction 생성
    measure_culane_metric.py          # 생성된 prediction으로 CULane F1 계산
    measure_pytorch_latency_dla34.py  # PyTorch latency/FPS 측정

  tests/
    test_import_sources.py            # local override / official import 경로 확인
    test_nms_import.py                # 외부 CUDA NMS import 확인
    test_nms_compute.py               # 외부 CUDA NMS 계산 결과 확인

  extensions/nms/                     # 외부 CUDA NMS 확장
  clrnet/                             # 공식 clrnet 패키지 proxy/override
  mmcv/                               # 추론/평가용 최소 MMCV shim
  torchvision/                        # import-only torchvision shim
  data/CULane -> ../../data/CULane
  outputs/                            # 실행 결과 저장
```

## 1. 외부 CUDA NMS 빌드

공식 CLRNet의 `clrnet/ops`를 직접 수정하지 않고, 이 프로젝트 안에서 NMS를
따로 빌드합니다.

```bash
cd clrnet_inference_test/extensions/nms
TORCH_CUDA_ARCH_LIST=8.7 python setup.py build_ext --inplace
cd ../../..
```

빌드 결과:

```text
clrnet_inference_test/extensions/nms/nms_impl*.so
```

`ImportError: libc10.so`가 발생하면 PyTorch C++ 라이브러리 경로가
`LD_LIBRARY_PATH`에 들어가 있는지 확인하세요.

```bash
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```

## 2. 기본 테스트

```bash
cd clrnet-jetson
```

import 경로 확인:

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

정상이라면 local `clrnet.ops.nms`는 이 프로젝트에서 import되고,
`clrnet.models`, `clrnet.datasets`, `clrnet.utils`는 공식 CLRNet에서 import됩니다.

## 3. 단일 이미지 추론

```bash
cd clrnet-jetson

python clrnet_inference_test/scripts/inference_culane_dla34.py \
  --image clrnet_inference_test/assets/images/test.jpg \
  --device cuda
```

출력:

```text
clrnet_inference_test/outputs/single_image/*_clrnet_dla34.jpg
clrnet_inference_test/outputs/single_image/*.lines.txt
```

`.lines.txt`는 CULane prediction text format입니다.

## 4. 전체 CULane prediction 생성

전체 CULane test split 34,680장을 추론하고 prediction 파일을 생성합니다.
Jetson Orin Nano에서 오래 걸릴 수 있습니다.

```bash
cd clrnet-jetson

python clrnet_inference_test/scripts/evaluate_culane_dla34.py \
  --device cuda \
  --output-dir clrnet_inference_test/outputs/eval/dla34_official
```

출력:

```text
clrnet_inference_test/outputs/eval/dla34_official/**/*.lines.txt
```

이번 실행 기준으로 전체 34,680개 prediction 파일 생성을 확인했습니다.

## 5. CULane F1 계산

공식 `dataset.evaluate()`는 내부에서 `Pool(cpu_count())`를 사용해 전체 metric을
계산합니다. Jetson에서는 이 방식이 오래 걸리고 진행률을 보기 어렵습니다.

그래서 이 프로젝트에서는 metric 계산을 별도 스크립트로 분리했습니다.
수식은 공식 `clrnet.utils.culane_metric.culane_metric()`을 그대로 사용하고,
worker 수 제한과 진행률 저장만 추가했습니다.

```bash
cd clrnet-jetson

python clrnet_inference_test/scripts/measure_culane_metric.py \
  --pred-dir clrnet_inference_test/outputs/eval/dla34_official \
  --data-root clrnet_inference_test/data/CULane \
  --output-json clrnet_inference_test/outputs/eval/dla34_official_metric_0_5.json \
  --iou-threshold 0.5 \
  --workers 4 \
  --progress-interval 500
```

현재 전체 평가 결과:

```text
TP        78419
FP        11593
FN        26467
Precision 0.8712060614
Recall    0.7476593635
F1@0.5    0.8047183655
```

결과 파일:

```text
clrnet_inference_test/outputs/eval/dla34_official_metric_0_5.json
```

## 6. PyTorch 속도 측정

TensorRT 변환 전 PyTorch baseline latency/FPS를 측정합니다.

```bash
cd clrnet-jetson

python clrnet_inference_test/scripts/measure_pytorch_latency_dla34.py \
  --device cuda \
  --limit 100 \
  --warmup 10
```

출력:

```text
clrnet_inference_test/outputs/latency/pytorch_dla34/pytorch_latency_dla34.json
clrnet_inference_test/outputs/latency/pytorch_dla34/PYTORCH_LATENCY_BASELINE.md
```

## Compatibility Shims

이 프로젝트는 Jetson에서 full `mmcv-full` 빌드를 피하기 위해 최소 shim을 둡니다.

- `clrnet/`: local proxy package입니다. `clrnet.ops.nms`만 이 프로젝트의 외부 NMS로
  override하고, 나머지 `clrnet.models`, `clrnet.datasets`, `clrnet.utils`는 공식
  CLRNet으로 넘깁니다.
- `extensions/nms/`: PyTorch 2.x에 맞춘 외부 C++/CUDA NMS 확장입니다.
- `mmcv/`: CLRNet 추론/평가 import path에 필요한 최소 API만 제공합니다.
- `torchvision/`: 현재 CULane 추론/평가에서 실제 torchvision API를 호출하지 않기
  때문에 import-only shim으로 처리합니다.

이 shim들은 학습용이 아닙니다. 학습을 하려면 공식 dependency를 별도로 맞춰야 합니다.

## Checkpoint Loading

스크립트는 `weights/culane_dla34.pth`를 `strict=False`로 로드하지만, 위험한
불일치는 실패 처리합니다.

허용되는 missing key:

```text
heads.criterion.weight
heads.prior_feat_ys
heads.prior_ys
heads.sample_x_indexs
```

이 4개는 학습된 inference weight가 아닙니다.

- `heads.criterion.weight`: 학습 loss용 weight
- `heads.prior_feat_ys`, `heads.prior_ys`, `heads.sample_x_indexs`: 공식 CLRNet head가
  config로부터 생성하는 deterministic buffer

그 외 missing key나 unexpected key가 나오면 모델 구조와 checkpoint가 맞지 않는
것으로 보고 실패합니다.

## 현재 검증 상태

확인 완료:

- 공식 CLRNet repo 수정 없음
- 외부 CUDA NMS import 성공
- 외부 CUDA NMS 계산 테스트 성공
- 단일 이미지 추론 성공
- 전체 CULane test split 34,680장 prediction 생성 성공
- F1@0.5 계산 완료: `0.8047183655`

## 라이선스와 공개 시 주의사항

이 저장소는 Apache License 2.0 기준으로 공개할 수 있게 정리했습니다.
루트의 `LICENSE`와 `THIRD_PARTY_NOTICES.md`를 함께 포함해야 합니다.

이 프로젝트는 공식 CLRNet을 기반으로 추론/평가 wrapper를 만든 것입니다.

- 공식 CLRNet: https://github.com/Turoad/CLRNet
- 공식 CLRNet 라이선스: Apache License 2.0
- 외부 CUDA NMS(`extensions/nms`)는 공식 CLRNet NMS 구현을 PyTorch 2.x / Jetson
  환경에서 빌드되도록 맞춘 코드입니다.

GitHub에 포함하면 안 되는 항목:

- `clrnet/`: 공식 upstream repo는 submodule 또는 설치 안내로 처리하고, 이 저장소에
  그대로 vendoring하지 않습니다.
- `data/`, `clrnet_inference_test/data/`: CULane 데이터셋과 symlink는 포함하지 않습니다.
- `weights/`: `culane_dla34.pth` 같은 checkpoint는 포함하지 않습니다.
- `outputs/`: 추론/평가 결과는 실행 산출물이므로 포함하지 않습니다.
- `clrnet_inference_test/extensions/nms/*.so`, `build/`: Jetson에서 다시 빌드해야 하는
  native build 산출물입니다.
- `clrnet_inference_test/assets/images/*.jpg`: CULane에서 복사한 테스트 이미지는
  데이터셋 라이선스 확인 전에는 포함하지 않습니다.

현재 `.gitignore`는 위 항목을 제외하도록 설정되어 있습니다.

## 개발 참고

일부 코드 구조 정리와 문서 작성 과정에서 OpenAI Codex를 활용했습니다.
다만 최종 실행 경로, CUDA NMS 빌드, 단일 이미지 추론, 전체 CULane prediction 생성,
F1 계산은 Jetson Orin Nano 환경에서 직접 검증했습니다.

이 프로젝트의 핵심 작업은 공식 CLRNet 코드를 직접 수정하지 않고, Jetson/PyTorch 2.x
환경에서 동작하도록 외부 CUDA NMS, 최소 dependency shim, 추론/평가 runner를 구성한
것입니다.

## 포함하지 않는 파일 준비 방법

이 저장소는 용량, 라이선스, 실행 환경 차이 때문에 데이터셋, checkpoint,
빌드 산출물, 실행 결과를 포함하지 않습니다. 새 환경에서 사용할 때는 아래 순서로
직접 준비해야 합니다.

### 1. 공식 CLRNet 소스

포함하지 않는 경로:

```text
clrnet-jetson/clrnet
```

준비 방법:

```bash
cd clrnet-jetson
git clone https://github.com/Turoad/CLRNet.git clrnet
```

이 프로젝트는 공식 CLRNet 코드를 수정하지 않고,
`clrnet_inference_test/clrnet` proxy package를 통해 공식 CLRNet 모듈을 import합니다.

### 2. CULane 데이터셋

포함하지 않는 경로:

```text
clrnet-jetson/data/CULane
clrnet-jetson/clrnet_inference_test/data/CULane
```

준비 방법:

```bash
mkdir -p data
```

CULane 압축 파일을 풀어서 아래 구조가 되도록 배치합니다.

```text
data/CULane/driver_xx_xxframe
data/CULane/laneseg_label_w16
data/CULane/list
```

그 다음 이 프로젝트 안에 symlink를 만듭니다.

```bash
mkdir -p clrnet_inference_test/data

ln -s ../../data/CULane clrnet_inference_test/data/CULane
```

### 3. 학습된 checkpoint

포함하지 않는 경로:

```text
clrnet-jetson/weights/culane_dla34.pth
```

준비 방법:

```bash
mkdir -p weights
```

`culane_dla34.pth`를 `weights` 폴더에 직접 배치한 뒤 확인합니다.

```bash
ls weights/culane_dla34.pth
```

### 4. CUDA NMS 빌드 산출물

포함하지 않는 파일:

```text
clrnet_inference_test/extensions/nms/*.so
clrnet_inference_test/extensions/nms/build/
```

준비 방법:

```bash
cd clrnet_inference_test/extensions/nms
TORCH_CUDA_ARCH_LIST=8.7 python setup.py build_ext --inplace
cd ../../..
```

빌드 후 확인:

```bash
ls nms_impl*.so
```

### 5. outputs 실행 결과

포함하지 않는 경로:

```text
clrnet_inference_test/outputs/
```

단일 이미지 결과 생성:

```bash
cd clrnet-jetson

python clrnet_inference_test/scripts/inference_culane_dla34.py \
  --image clrnet_inference_test/assets/images/test.jpg \
  --device cuda
```

전체 CULane prediction 생성:

```bash
cd clrnet-jetson

python clrnet_inference_test/scripts/evaluate_culane_dla34.py \
  --device cuda \
  --output-dir clrnet_inference_test/outputs/eval/dla34_official
```

### 6. 테스트 이미지

CULane에서 복사한 샘플 이미지는 데이터셋 라이선스 문제 때문에 저장소에 포함하지
않습니다. 필요하면 CULane 데이터에서 직접 한 장을 복사해서 사용합니다.

```bash
mkdir -p clrnet_inference_test/assets/images

cp data/CULane/.../이미지.jpg clrnet_inference_test/assets/images/test.jpg
```
