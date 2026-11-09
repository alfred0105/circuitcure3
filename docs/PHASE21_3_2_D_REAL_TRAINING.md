# Phase 21.3.2-D — 실제 YOLO 학습 및 탐색 실험

## 범위와 실행 계약

이 Phase는 현재 배포된 10-class 모델과 분리된 연구용 실험이다. `integrated_circuit` 단일 클래스와 Kaggle 원본 9개 클래스 모델을 별도로 학습한다. 기존 배포 ONNX 파일, 서비스 UI/API, SPICE 진단 코드는 바꾸지 않는다. 체크포인트와 실제 데이터는 Git에 넣지 않고 로컬 `data/component_training/` 아래 보존한다.

- 기준 저장소 커밋: `07fb5cb368af3d5426bdb93fa5e244fa58d1f78e`
- 실험 설정: [phase21-3-2-d-ic-yolov8s.yaml](../experiments/component_recognition/configs/phase21-3-2-d-ic-yolov8s.yaml)
- 기존 실험 러너: `scripts/component_training.py`
- PCB-DSLR 무결성 검사: `scripts/verify_pcb_dslr_dataset.py`
- 실험 추론 CLI: `scripts/run_component_experiment_inference.py`
- 실제 실행 결과: 로컬 `data/component_training/runs/` (Git ignore)

## 데이터 상태와 Manifest 불일치

`pcb-dslr-part1-ic-manifest.json`은 현재 `TRAINING_READY`, 87장/2,132 IC Box, 비상업적 연구 전용 사용 범위, 출처 제공 보드 ID 20개 및 14/3/3 보드 분할을 기록한다. 반면 이전 `processed/pcb-dslr-part1-ic-v3/candidate_manifest.json`은 후보 생성 시점의 `IMAGE_DOMAIN_REVIEW_PENDING` 상태를 남긴다. 학습 전에 이 차이를 숨기지 않고, 상위 승인 Manifest와 파일 해시·split·box 수를 다시 검사했다. 현재 Manifest에는 이전에 기록된 `HUMAN_REVIEWED_CONTACT_SHEET` 상태가 있어 이를 그대로 참고했으며 이번 Phase에서 새 수동 이미지 판정은 요구하거나 수행하지 않았다. 자동 검사는 그 시각 판정을 새로 증명하지 않는다.

`verify_pcb_dslr_dataset.py` 결과: 87/87 이미지, annotation 및 board mask, 변환 YOLO 라벨의 SHA-256 일치; Train 59장/14보드/1,825 Box, Validation 14장/3보드/153 Box, Sealed Test 14장/3보드/154 Box; split 간 보드 중복 0. 원본 archive part 1만 사용되며 전체 8개 archive part를 취득한 것은 아니다.

## 환경 및 학습

MacBook Air M5/ARM64, macOS 27.0.1, Python 3.12.14, PyTorch 2.15.0, Ultralytics 8.4.203, ONNX Runtime 1.31.1을 사용했다. MPS는 build/availability 확인과 간단한 gradient/update probe를 통과했다. CUDA는 사용하지 않았다. 4장 YOLO Dataset Loader smoke test도 성공했다. 학습 중 PyTorch가 MPS의 일부 연산에 대해 결정적 구현이 없다고 경고했으므로 seed 설정은 재현성을 돕지만 bitwise deterministic을 보장하지 않는다.

실행 설정은 YOLOv8s pretrained, 640px, batch 2, epochs 상한 50, patience 15, optimizer auto, seed 42, workers 2, cache/AMP off다. `integrated_circuit` 1-class transfer head를 만들었다. 실제 학습은 epoch 34에서 조기 종료됐고 최적 validation checkpoint는 epoch 19다. Validation은 학습 과정 중 매 epoch 사용했으며 sealed test는 checkpoint 선택 후 한 번 평가했다. 학습 전 전체 데이터 Manifest fingerprint를 만들면서 test 파일 byte hash도 기록했지만 test 예측이나 metric은 모델 선택 전에 계산하지 않았다.

**PCB-DSLR 점수는 publisher 제공 Annotation과의 agreement다.** source annotation은 독립 판정된 ground truth가 아니다. 실제 PCB 일반화 성능을 확정하거나 서비스 배포를 승인하는 결과로 보지 않는다. Dataset은 비상업적 연구에만 사용 가능하고 상업적 사용 권한은 확인되지 않았다.

## 별도 탐색 9-class 학습

Kaggle C.1 소스 Annotation을 그대로 유지한 `EXPLORATORY_SOURCE_LABELS` 모델을 별도로 시도한다. 빈 Annotation 114장은 Negative로 취급하지 않고 이 실험에서 제외한다. 원본 9개 클래스는 병합하지 않는다. filename/augmentation group, exact image hash와 모든 C.1 ORB/RANSAC 후보 연결을 단일 split component로 묶고 기존 source split은 무시한다. 물리 PCB ID와 완전한 Annotation은 미확정이며 sealed test를 만들지 않는다.

수동 승인 UI를 학습의 선행조건으로 두지 않는다. 이 실험 점수는 source label agreement로만 해석한다. 실행 결과와 클래스별 점수, support가 없는 클래스는 [Kaggle 실험 보고서](PHASE21_3_2_D_KAGGLE_EXPERIMENT.md)에 기록한다.

## 기존 모델과 분리

실험 checkpoint와 ONNX는 `data/component_training/runs/<run-id>/` 안에 있다. `scripts/run_component_experiment_inference.py --model PATH`로 명시적으로 선택하며 결과에는 `EXPERIMENTAL_COMPONENT_DETECTOR`, 1-class contract 및 `production_model: false`가 기록된다. 기존 10-class ONNX 경로·decoder·서비스 설정은 교체하지 않았다.

## CAD 파일 연계

KiCad/Netlist 파일 후보를 읽기 전용으로 확인했지만, 현재 PCB-DSLR 또는 Kaggle 이미지와 묶을 수 있는 보드 식별자·이미지 경로·등록 정보는 찾지 못했다. CAD 라벨을 Annotation으로 사용하지 않았다. 검색 범위와 자세한 결과는 [CAD 연계 보고서](PHASE21_3_2_D_CAD_ASSOCIATION.md)에 있다.

## 재현 명령

```bash
python3 scripts/verify_pcb_dslr_dataset.py
python3 scripts/component_training.py doctor
python3 scripts/component_training.py train --data data/component_training/processed/pcb-dslr-part1-ic-v3/data.yaml --profile baseline --model yolov8s.pt --epochs 50 --seed 42 --patience 15 --workers 2
python3 scripts/evaluate_pcb_dslr_experiment.py --weights PATH_TO_BEST_PT --split validation --output PATH_TO_VALIDATION_REPORT --device mps
python3 scripts/evaluate_pcb_dslr_experiment.py --weights PATH_TO_BEST_PT --split sealed_test --output PATH_TO_SEALED_TEST_REPORT --device mps
```

가중치·데이터는 재배포하지 않는다. Zenodo dataset의 비상업적 연구 조건을 따른다.
