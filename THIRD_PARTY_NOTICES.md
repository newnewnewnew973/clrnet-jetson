# Third Party Notices

이 저장소는 Jetson Orin Nano에서 CLRNet DLA34 CULane 추론/평가를 재현하기
위한 래퍼 프로젝트입니다.

## CLRNet

- Repository: https://github.com/Turoad/CLRNet
- Paper: CLRNet: Cross Layer Refinement Network for Lane Detection, CVPR 2022
- License: Apache License 2.0

사용 방식:

- 공식 CLRNet 소스 트리는 이 저장소에 포함하지 않습니다.
- `clrnet_inference/clrnet`은 공식 `clrnet` 패키지를 수정하지 않고 import하기
  위한 proxy/override 패키지입니다.
- `clrnet_common/extensions/nms`의 CUDA NMS 확장은 CLRNet의 NMS 구현을
  PyTorch 2.x / Jetson 환경에서 빌드되도록 맞춘 코드입니다.
- `clrnet_inference/mmcv`와 `clrnet_inference/torchvision`은 학습용
  대체 구현이 아니라, 이 추론/평가 경로에서 필요한 최소 import shim입니다.

## Data and Weights

이 저장소는 CULane 데이터셋과 학습된 checkpoint 파일을 포함하지 않습니다.

- CULane 데이터는 사용자가 별도로 준비해야 합니다.
- `weights/culane_dla34.pth` 같은 checkpoint는 원 배포처의 라이선스와 사용 조건을
  따라야 합니다.
- `outputs/` 아래 결과 파일은 실행 산출물이므로 저장소에 포함하지 않습니다.
