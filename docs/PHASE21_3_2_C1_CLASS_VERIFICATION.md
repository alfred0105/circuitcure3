# Phase 21.3.2-C.1 — 소스 클래스 검증

각 원본 클래스를 박스 주변 crop으로 모아 크기와 서로 다른 filename stem을 가로질러 최대 40개를 볼 수 있는 gallery로 만들었습니다. 수량이 3개인 `Resestor`는 존재하는 1개 stem의 샘플만 포함합니다. 자료는 자동 표본 추출 및 시각 점검 보조물이며 reviewer 승인으로 간주하지 않습니다.

| 원본 클래스 | 원본 박스 수 | Gallery 표본 | 의미 결정 |
|---|---:|---:|---|
| Cap1 | 1,295 | 40 | `MAPPING_PENDING` |
| Cap2 | 1,264 | 40 | `MAPPING_PENDING` |
| Cap3 | 1,199 | 40 | `MAPPING_PENDING` |
| Cap4 | 1,074 | 40 | `MAPPING_PENDING` |
| MOSFET | 1,295 | 40 | `MAPPING_PENDING` |
| Mov | 1,291 | 40 | `MAPPING_PENDING` |
| Resestor | 3 | 1 | `MAPPING_PENDING` |
| Resistor | 1,151 | 40 | `MAPPING_PENDING` |
| Transformer | 2,547 | 40 | `MAPPING_PENDING` |

자동 gallery의 육안 표본은 여러 클래스가 일관된 부품 모양을 가리키는지 확인하는 시작점입니다. `Cap1`~`Cap4`를 같은 capacitor 종류 또는 배포 클래스에 합치는 결정, `Resestor` 오타 판단, MOSFET/MOV/Transformer의 의미는 사람 검토 및 증거가 없어 확정하지 않았습니다. 서로 다른 클래스의 박스가 다른 영역을 덮는지, 소자 경계와 정합하는지도 ground truth로 검증하지 않았습니다.

갤러리는 `data/component_training/processed/kaggle-c1-review/class-galleries/<class>/`에 있습니다. 클래스별 표본 수, 고유 filename 그룹 수, reviewer/evidence 및 보류 상태는 [`pcb-component-detection-c1-class-review.json`](../data/component_training/manifests/pcb-component-detection-c1-class-review.json)에 저장했습니다. Reviewer, 확인 오분류 수는 비워 두었습니다. 따라서 Canonical 10-Class와 Source-Only 모두 학습 승인 클래스가 없습니다.
