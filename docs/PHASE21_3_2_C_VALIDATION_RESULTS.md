# Phase 21.3.2-C — 검증 결과

## 검사 결과

- Kaggle 원본 이미지 1,410장, annotation 11,119개를 이미지별 manifest로 옮겼습니다.
- 1,410장에 대한 57개 Contact Sheet 페이지를 AI 시각 감사했습니다. 자동 분류 후보는 실장 PCB 1,410장이지만 사람 확인은 0장입니다. board group은 미확정 후보이며 source physical board ID는 0개입니다.
- 상자 검사: 유효 11,119, 무효 0.
- 중복 검사: SHA-256 duplicate group 0, pHash distance ≤6 pair 228; 이 중 source split 경계를 넘는 쌍 71.
- Group 검사: 433 filename stem group, split 경계를 넘는 37 group / 219장.
- Label 상태: `ANNOTATION_STATUS_UNKNOWN` 1,296장, `EMPTY_UNREVIEWED` 114장, `ANNOTATION_COMPLETE` 0장, reviewed empty 0장.
- 신규 학습/validation 후보 0장. board-level 독립 test 0장.

## Apple Silicon 학습 환경

`scripts/component_training.py doctor` 결과 Python 3.12.14, arm64, PyTorch 2.15.0, MPS available/probe successful, 총 메모리 24 GiB, 검사 시 여유 메모리 약 5.89 GiB였습니다.

Ultralytics DataLoader mechanics-only smoke를 로컬 source train split에서 별도로 수행했습니다. 1,099장 train index를 읽고 batch 2, 320×320 RGB tensor와 28개 source box를 만들었으며 batch tensor를 `mps:0`으로 이동했습니다. 이는 loader와 MPS tensor 동작 확인일 뿐입니다. 이 source split은 board leakage가 있고 annotation이 미검토 상태이므로 모델 학습·validation 또는 성능 증거가 아닙니다. Epoch 학습은 하지 않았습니다.

## 코드 및 테스트

Negative 평가 CLI가 존재하지 않는 `image_count` 필드에 접근하던 코드를 실제 metric 필드로 수정했습니다. 성공·부분 실패·전체 실패·reviewed negative 0장·CLI 실행 회귀 테스트를 추가했습니다. 추론 실패 건은 FP 0으로 점수화되지 않으며 분모에서 제외되고 실패 수로 따로 보고됩니다.

로컬 비통합 테스트 결과는 **390 passed, 1 skipped, 48 deselected**입니다. 48개는 integration, stress, crash-injection mark가 붙어 별도 CI job에서 실행되는 테스트입니다. 기존 ZIP 중복 entry fixture에서 경고 1개가 있었지만 테스트 결과에는 영향이 없습니다.

기계 판독 산출물은 [`pcb-component-detection-curation-manifest.json`](../data/component_training/manifests/pcb-component-detection-curation-manifest.json), [`pcb-component-detection-quality-report.json`](../data/component_training/manifests/pcb-component-detection-quality-report.json), [`pcb-component-detection-board-leakage-report.json`](../data/component_training/manifests/pcb-component-detection-board-leakage-report.json), [`pcb-component-detection-annotation-completeness-report.json`](../data/component_training/manifests/pcb-component-detection-annotation-completeness-report.json)에 있습니다.

테스트 원자료와 MPS/DataLoader 환경 기록은 [`pcb-component-detection-validation-results.json`](../data/component_training/manifests/pcb-component-detection-validation-results.json)에 보관했습니다.
