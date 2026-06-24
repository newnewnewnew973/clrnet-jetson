# CLRNet CULane Inference Baseline

Jetson Orin Nano에서 CLRNet CULane checkpoint를 실행하기 위한 추론/평가용
래퍼입니다. 기본 baseline은 DLA34 checkpoint이지만, 실행 모델은 config와
checkpoint로 결정합니다. 이 디렉터리는 TensorRT 변환 전 PyTorch 기준 정확도와
속도를 고정하는 용도입니다.

기본 원칙은 다음과 같습니다.

- 공식 CLRNet 소스는 수정하지 않습니다.
- 이 저장소에는 공식 CLRNet 전체 소스를 vendoring하지 않습니다.
- `clrnet_inference_test/` 안의 proxy package로 필요한 import만 가로챕니다.
- PyTorch 2.x / Jetson에서 필요한 CUDA NMS만 별도 extension으로 빌드합니다.
- 전체 metric 계산은 Jetson에서 안정적으로 돌도록 worker 수를 명시합니다.

환경 구성은 [ENVIRONMENT.md](ENVIRONMENT.md)를 먼저 확인하십시오.

## Repository Layout

이 문서의 모든 명령은 clone한 repository root에서 실행하는 것을 기준으로 합니다.

```bash
git clone git@github.com:newnewnewnew973/clrnet-jetson.git
cd clrnet-jetson
```

필요한 작업 디렉터리 구조:

```text
.
├── clrnet/                         # 공식 CLRNet upstream checkout, git에는 포함하지 않음
├── clrnet_inference_test/           # 이 래퍼 프로젝트
├── data/CULane/                     # CULane dataset, git에는 포함하지 않음
└── weights/culane_dla34.pth         # DLA34 checkpoint, git에는 포함하지 않음
```

`clrnet_inference_test/` 내부 구조:

```text
clrnet_inference_test/
  scripts/
    inference_culane.py               # 단일 이미지 추론
    evaluate_culane.py                # CULane test split prediction 생성
    measure_culane_metric.py          # 생성된 prediction으로 F1 계산
    measure_pytorch_latency.py        # PyTorch latency/FPS 측정
    runtime.py                        # clrnet_common.runtime 호환 import

  tests/
    test_import_sources.py            # local proxy / official source import 확인
    test_nms_import.py                # CUDA NMS extension import 확인
    test_nms_compute.py               # CUDA NMS 계산 확인

  clrnet/                             # 공식 clrnet package proxy
  mmcv/                               # 추론/평가용 최소 MMCV shim
  torchvision/                        # import-only torchvision shim
  outputs/                            # 실행 결과 저장
```

공통 runtime, latency helper, CUDA NMS extension은 workspace root의
`clrnet_common/`에 둡니다. 이후 TensorRT/C++ 변환 테스트도 이 공통 모듈을
재사용합니다.

스크립트는 모델명을 `--model`로 받습니다. 기본값은 `dla34`이며,
`--model resnet34`를 주면 기본 config는 `clrnet/configs/clrnet/clr_resnet34_culane.py`,
기본 checkpoint는 `weights/culane_resnet34.pth`로 해석합니다. config나 checkpoint
위치가 다르면 `--config`, `--checkpoint`로 개별 override합니다.

## External Inputs

clone 직후에는 다음 항목을 직접 준비해야 합니다.

공식 CLRNet source:

```bash
git clone https://github.com/Turoad/CLRNet.git clrnet
```

CULane dataset:

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

DLA34 checkpoint:

```text
weights/culane_dla34.pth
```

`clrnet_inference_test/data/CULane`은 `data/CULane`을 가리키는 symlink로 둡니다.

```bash
mkdir -p clrnet_inference_test/data
ln -s ../../data/CULane clrnet_inference_test/data/CULane
```

## CUDA NMS Build

공식 CLRNet의 `clrnet/ops`는 수정하지 않습니다. 이 프로젝트 안의 NMS extension을
따로 빌드합니다.

```bash
cd clrnet_common/extensions/nms
TORCH_CUDA_ARCH_LIST=8.7 python setup.py build_ext --inplace
cd ../../..
```

빌드 산출물:

```text
clrnet_common/extensions/nms/nms_impl*.so
```

`ImportError: libc10.so`가 발생하면 PyTorch C++ library path를 확인합니다.

```bash
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```

## Sanity Checks

```bash
python clrnet_inference_test/tests/test_import_sources.py
python clrnet_inference_test/tests/test_nms_import.py
python clrnet_inference_test/tests/test_nms_compute.py
```

정상 상태:

- `clrnet.ops.nms`는 `clrnet_inference_test/`에서 import됩니다.
- `clrnet.models`, `clrnet.datasets`, `clrnet.utils`는 공식 `./clrnet/clrnet`에서 import됩니다.
- `mmcv`, `torchvision`은 `clrnet_inference_test/`의 최소 shim을 사용합니다.

## Single Image Inference

```bash
python clrnet_inference_test/scripts/inference_culane.py \
  --image /path/to/culane/image.jpg \
  --model dla34 \
  --device cuda
```

예시 출력:

```text
clrnet_inference_test/outputs/single_image/*_clrnet_<model>.jpg
clrnet_inference_test/outputs/single_image/*.lines.txt
```

`.lines.txt`는 CULane prediction text format입니다.

## Full CULane Prediction

전체 CULane test split 34,680장에 대해 prediction 파일을 생성합니다.

```bash
python clrnet_inference_test/scripts/evaluate_culane.py \
  --model dla34 \
  --device cuda \
  --batch-size 8
```

이 스크립트의 기본 동작은 prediction 파일 생성까지입니다. 공식 CLRNet
`dataset.evaluate()`는 내부에서 `Pool(cpu_count())`를 사용하므로 Jetson에서는 기본
경로로 사용하지 않습니다. `--output-dir`을 생략하면 실행마다 timestamp가 붙은
디렉터리를 생성합니다.

출력:

```text
clrnet_inference_test/outputs/eval/dla34_<YYYYmmdd_HHMMSS>/**/*.lines.txt
```

정상 생성 개수:

```bash
find clrnet_inference_test/outputs/eval/dla34_<YYYYmmdd_HHMMSS> -name '*.lines.txt' | wc -l
# 34680
```

재현 비교를 위해 고정 경로가 필요하면 `--output-dir`을 명시합니다.

## CULane F1 Metric

Jetson에서는 worker 수를 제한한 metric script를 사용합니다.

```bash
python clrnet_inference_test/scripts/measure_culane_metric.py \
  --model dla34 \
  --pred-dir clrnet_inference_test/outputs/eval/dla34_<YYYYmmdd_HHMMSS> \
  --data-root clrnet_inference_test/data/CULane \
  --iou-threshold 0.5 \
  --workers 4 \
  --progress-interval 500
```

커밋 전 전체 검증에서 확인한 결과:

```text
TP        78424
FP        11594
FN        26462
Precision 0.8712035371
Recall    0.7477070343
F1@0.5    0.8047449001
```

## PyTorch Latency Baseline

TensorRT 변환 전 PyTorch 기준 latency/FPS를 측정합니다.

```bash
python clrnet_inference_test/scripts/measure_pytorch_latency.py \
  --model dla34 \
  --device cuda \
  --limit 100 \
  --warmup 100 \
  --block-warmup 10
```

출력:

```text
clrnet_inference_test/outputs/latency/pytorch_dla34/pytorch_latency_dla34.json
clrnet_inference_test/outputs/latency/pytorch_dla34/PYTORCH_LATENCY_BASELINE.md
```

`--model resnet34`처럼 모델명을 바꾸면 기본 출력 경로와 JSON 파일명도
`pytorch_resnet34/pytorch_latency_resnet34.json` 형태로 바뀝니다.

다른 config/checkpoint를 사용할 때는 다음 값을 함께 바꿉니다.

```bash
--config clrnet/configs/clrnet/clr_resnet34_culane.py
--checkpoint weights/culane_resnet34.pth
--model resnet34
```

커밋 전 smoke 확인값:

```text
samples                       4
warmup                        4
block_warmup                  2
continuous E2E FPS            17.13
CUDA-event forward FPS        28.76
CUDA-event forward mean       34.77 ms
forward + postprocess mean    39.68 ms
```

위 latency smoke는 커밋 전 동작 확인용 짧은 실행입니다. 성능 비교용 baseline은
`--limit 100 --warmup 100 --block-warmup 10`처럼 충분한 sample과 warmup으로 다시
측정합니다. 논문 FPS와 직접 비교하지 않습니다.

## Compatibility Shims

이 프로젝트는 Jetson에서 full `mmcv-full` 빌드를 피하기 위해 최소 shim을 둡니다.

- `clrnet/`: local proxy package입니다. `clrnet.ops.nms`만 local CUDA NMS로
  override하고, 나머지 `clrnet.models`, `clrnet.datasets`, `clrnet.utils`는 공식
  CLRNet으로 넘깁니다.
- `../clrnet_common/extensions/nms/`: PyTorch 2.x에 맞춘 C++/CUDA NMS extension입니다.
- `mmcv/`: 현재 추론/평가 경로에 필요한 API만 제공합니다.
- `torchvision/`: 공식 CULane dataset import를 통과시키기 위한 import-only shim입니다.

이 shim들은 추론/평가용입니다. 학습에는 사용하지 않습니다.

## Checkpoint Loading

스크립트는 `weights/culane_dla34.pth`를 `strict=False`로 로드하되, 위험한 key
불일치는 실패 처리합니다.

허용되는 missing key:

```text
heads.criterion.weight
heads.prior_feat_ys
heads.prior_ys
heads.sample_x_indexs
```

이 key들은 inference weight가 아닙니다.

- `heads.criterion.weight`: training loss weight
- `heads.prior_feat_ys`, `heads.prior_ys`, `heads.sample_x_indexs`: config에서 생성되는 deterministic buffer

그 외 missing key나 unexpected key가 있으면 checkpoint와 모델 구조가 맞지 않는
것으로 보고 실패합니다.

## Git Policy

git에 포함하지 않는 항목:

- `clrnet/`
- `data/`
- `weights/`
- `clrnet_inference_test/data/`
- `clrnet_inference_test/outputs/`의 실행 결과
- `*.pth`, `*.onnx`, `*.engine`, `*.trt`, `*.plan`

라이선스 고지는 repository root의 `LICENSE`와 `THIRD_PARTY_NOTICES.md`를 함께
유지합니다.
