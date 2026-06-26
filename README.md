# CLRNet Jetson 배포 작업공간

이 workspace는 공식 CLRNet 구현을 직접 수정하지 않고, Jetson Orin Nano Super
환경에서 PyTorch baseline, TensorRT FP16 최적화, DeepStream 배포까지 검증하기
위해 구성한 차선 인식 배포 프로젝트입니다.

프로젝트의 목적은 새로운 차선 인식 모델을 제안하는 것이 아니라, 이미
검증된 CLRNet을 실제 Jetson 배포 환경으로 가져와 정확도와 성능을 수치로
확인하는 것입니다.

이 프로젝트에서 확인한 범위는 다음과 같습니다.

- CULane 전체 테스트 34,680장 기준 PyTorch 정확도 평가
- PyTorch baseline과 TensorRT FP16 결과의 정확도 비교
- PyTorch와 TensorRT의 latency/FPS 비교
- DeepStream 7.1 container에서 hardware decode, `nvinfer`, lane overlay,
  full-frame output video 생성 검증

## 전체 구조

```text
.
├── clrnet/              # 공식 CLRNet upstream checkout, 직접 수정하지 않는 영역
├── clrnet_common/       # 공통 runtime, CULane, image, metric, latency, CUDA NMS
├── clrnet_inference/    # PyTorch 추론, 평가, metric, latency, test
├── clrnet_tensorrt/     # ONNX export, TensorRT engine build, 추론, 평가, latency
├── clrnet_deepstream/   # Jetson DeepStream 배포 pipeline
├── data/CULane/         # CULane dataset, git에는 포함하지 않음
└── weights/             # CLRNet checkpoint, git에는 포함하지 않음
```

중요한 설계 의도는 두 가지입니다.

첫째, 공식 CLRNet source인 `clrnet/`은 직접 수정하지 않습니다. upstream 코드를
고치면 원본과의 비교, 재현성 설명, 업데이트 추적이 어려워집니다. 따라서 이
프로젝트는 필요한 호환 코드와 배포 코드를 workspace 내부 패키지로 분리했습니다.

둘째, PyTorch와 TensorRT가 공통으로 써야 하는 로직은 `clrnet_common/`에
모았습니다. 전처리, lane decode, NMS, CULane metric이 서로 다르면 PyTorch와
TensorRT 결과 비교가 공정하지 않습니다. 그래서 backend별 차이는
`clrnet_inference/`, `clrnet_tensorrt/`, `clrnet_deepstream/`에 두고, 비교 기준이
되는 로직은 공통 모듈로 공유합니다.

## 패키지별 역할

### `clrnet/`

공식 CLRNet upstream source입니다. 이 디렉터리는 원본 구현을 보존하는 영역이며,
프로젝트 코드에서 직접 수정하지 않는 것을 원칙으로 합니다.

### `clrnet_common/`

PyTorch와 TensorRT가 공유하는 핵심 코드입니다.

- CLRNet runtime helper
- CULane dataset helper
- image preprocessing
- lane decode/NMS
- CULane metric 계산
- latency 측정 helper
- Jetson용 CUDA NMS extension

이 디렉터리가 있기 때문에 PyTorch baseline과 TensorRT 결과를 같은 기준으로
비교할 수 있습니다.

### `clrnet_inference/`

TensorRT 변환 전 PyTorch baseline을 잡는 패키지입니다.

- 단일 이미지 추론
- CULane full test prediction 생성
- CULane F1 metric 계산
- PyTorch latency 측정
- proxy import 및 CUDA NMS test

이 패키지의 목적은 "TensorRT로 빨라졌는가"를 말하기 전에 먼저 PyTorch 기준
정확도와 속도를 고정하는 것입니다.

### `clrnet_tensorrt/`

TensorRT 변환과 실행을 담당하는 패키지입니다.

- PyTorch checkpoint에서 ONNX export
- TensorRT FP16 engine build
- TensorRT 단일 이미지 추론
- CULane full test prediction 생성
- PyTorch baseline과 동일 metric 기준으로 정확도 비교
- TensorRT latency 측정

이 패키지의 핵심 검증은 FP16 변환 후에도 CULane 정확도가 PyTorch baseline과
거의 동일하게 유지되는지 확인하는 것입니다.

### `clrnet_deepstream/`

Jetson DeepStream 배포 pipeline입니다.

- DeepStream 7.1 container build
- Jetson hardware decoder `nvv4l2decoder` 사용
- TensorRT model을 `nvinfer`로 실행
- raw tensor metadata에서 CLRNet lane postprocess 수행
- `nvdsosd`로 lane overlay
- `nvcompositor`로 원본 full-frame output 복원

최종 목표는 다음과 같습니다.

```text
input video
-> hardware decode
-> CLRNet TensorRT inference through DeepStream
-> lane overlay
-> full-frame output video
```

## 필요한 입력 파일

clone 직후에는 다음 항목을 직접 준비해야 합니다.

```text
data/CULane/
weights/culane_dla34.pth
clrnet/
```

기본 스크립트는 다음 dataset link를 기대합니다.

```bash
mkdir -p clrnet_inference/data
ln -s ../../data/CULane clrnet_inference/data/CULane
```

실제 추론 전에 공통 CUDA NMS extension을 빌드합니다.

```bash
cd clrnet_common/extensions/nms
TORCH_CUDA_ARCH_LIST=8.7 python setup.py build_ext --inplace
cd ../../..
```

## 주요 실행 명령

### PyTorch baseline

```bash
python clrnet_inference/scripts/inference_culane.py --model dla34 --device cuda
python clrnet_inference/scripts/evaluate_culane.py --model dla34 --device cuda --batch-size 4
python clrnet_inference/scripts/measure_culane_metric.py --model dla34 --pred-dir clrnet_inference/outputs/eval/dla34_full --workers 4
python clrnet_inference/scripts/measure_pytorch_latency.py --model dla34 --device cuda --limit 1000 --warmup 100 --block-warmup 10
```

### TensorRT FP16

```bash
python clrnet_tensorrt/scripts/export_onnx.py --model dla34 --device cuda --output clrnet_tensorrt/outputs/onnx/clrnet_dla34.onnx
python clrnet_tensorrt/scripts/build_engine.py --model dla34 --onnx clrnet_tensorrt/outputs/onnx/clrnet_dla34.onnx --engine clrnet_tensorrt/outputs/engine/clrnet_dla34_fp16.engine --fp16 --workspace-gb 1
python clrnet_tensorrt/scripts/inference_culane.py --model dla34 --engine clrnet_tensorrt/outputs/engine/clrnet_dla34_fp16.engine --device cuda
python clrnet_tensorrt/scripts/evaluate_culane.py --model dla34 --engine clrnet_tensorrt/outputs/engine/clrnet_dla34_fp16.engine --data-root clrnet_inference/data/CULane --output-dir clrnet_tensorrt/outputs/eval/dla34_fp16_full --device cuda
python clrnet_tensorrt/scripts/measure_culane_metric.py --model dla34 --pred-dir clrnet_tensorrt/outputs/eval/dla34_fp16_full --data-root clrnet_inference/data/CULane --workers 4
python clrnet_tensorrt/scripts/measure_tensorrt_latency.py --model dla34 --data-root clrnet_inference/data/CULane --precision fp16 --fp16-engine clrnet_tensorrt/outputs/engine/clrnet_dla34_fp16.engine --device cuda --limit 1000 --warmup 100 --block-warmup 10
```

### DeepStream

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

세부 내용은 다음 문서를 참고합니다.

- `clrnet_inference/README.md`
- `clrnet_tensorrt/README.md`
- `clrnet_deepstream/README.md`

## 검증 결과

최신 full check는 이전 output artifact를 삭제한 뒤 다시 실행했습니다. 정확도
평가는 CULane full test split 34,680장을 사용했고, latency는 1,000장을 기준으로
측정했습니다.

### CULane 정확도

| Runtime | Samples | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| PyTorch DLA34 | 34,680 | 0.871190 | 0.747678 | 0.804722 |
| TensorRT DLA34 FP16 | 34,680 | 0.871202 | 0.747631 | 0.804700 |

TensorRT FP16 변환 후 F1 차이는 약 `0.000022`입니다. 따라서 이 실행에서는
FP16 TensorRT 변환으로 인한 정확도 손실이 사실상 없다고 볼 수 있습니다.

### Latency

| Runtime | Samples | E2E FPS | Pure model FPS |
| --- | ---: | ---: | ---: |
| PyTorch DLA34 | 1,000 | 12.72 | 31.39 |
| TensorRT DLA34 FP16 | 1,000 | 14.84 | 128.67 |

TensorRT pure engine FPS는 PyTorch pure model FPS보다 크게 높습니다. 다만 E2E
FPS에는 dataset access, preprocessing, host-to-device copy, lane decode, NMS,
Python-side postprocess가 포함되기 때문에 pure engine FPS 향상이 그대로 반영되지는
않습니다.

### DeepStream 배포 검증

| Runtime | Input | Output | Frames | Notes |
| --- | --- | --- | ---: | --- |
| DeepStream 7.1 + TensorRT FP16 | H.264 file | full-frame MP4 | 5,400 | `nvv4l2decoder`, `nvinfer`, `nvdsosd`, `nvcompositor` |

DeepStream output은 원본 `1640x590` full-frame을 유지합니다. model branch는
CLRNet CULane inference와 동일한 crop을 사용하고, lane overlay branch를
DeepStream 내부에서 원본 frame에 다시 compositing합니다.

## 생성 산출물

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
