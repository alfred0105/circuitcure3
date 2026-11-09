# Phase 21.3.2-C — 학습 준비 상태

## 결정: 학습 차단

Kaggle 데이터의 물리 도메인은 목표와 일치하지만, multi-class detection 학습을 위해 필요한 라벨 완전성, 클래스 의미, 보드별 split은 승인되지 않았습니다. 따라서 1,410장 중 실제 학습 가능 이미지 수는 0장입니다. 1,296장의 source annotation은 전체 라벨 coverage 확인 전까지 `ANNOTATION_STATUS_UNKNOWN`, 빈 라벨 114장은 소자가 보이므로 `PARTIAL_ANNOTATION`입니다.

| 용도 | 현재 승인 수 | 이유 |
|---|---:|---|
| 별도 9-class source-only 학습 | 0 | 라벨 coverage 및 source class 의미 미검토 |
| 일부 확인 class 학습 | 0 | Cap 하위 유형·`Resestor` 의미 및 미라벨 대상 미확인 |
| 기존 canonical 10-class 학습 | 0 | 승인된 class mapping 없음 |
| 독립 board-level validation/test | 0 | 시각 board group 1개, source physical ID 0개 |
| negative sample | 0 | 빈 라벨은 빈 장면이 아님 |

`pcb-component-detection-source-only-blocked.yaml`은 정책과 클래스 계약을 보존하는 차단용 파일입니다. 빈 train/validation 목록을 갖고 있어 모델 학습이나 정확도 측정에 사용하면 안 됩니다. 변환 파이프라인 테스트에는 합성 fixture를 사용합니다. 다운로드 원본과 annotation은 그대로 둡니다.

## 다음 학습 단계로 가기 위한 조건

1. 다른 물리 PCB를 수집하고 촬영 세션과 물리 보드 ID를 분리 기록합니다.
2. source class 정의를 확인하고 image별 모든 대상 부품의 annotation coverage를 adjudicate합니다.
3. 빈 라벨 이미지의 대상 부재를 확인하거나 라벨을 보완합니다.
4. 물리 보드 기준으로 개발용 train/validation과 독립 test를 구성합니다.

이 단계에서는 Phase 21.3.2-D IC 전용 학습이나 21.3.2-E 모델 비교 학습을 시작하지 않았습니다.
