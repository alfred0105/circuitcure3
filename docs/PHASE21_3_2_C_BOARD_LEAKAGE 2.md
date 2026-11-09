# Phase 21.3.2-C — 보드 그룹 및 누수

## 관찰

- SHA-256 완전 중복은 0개였습니다.
- pHash Hamming distance 6 이하 쌍은 228개였고, 이 중 71쌍이 원본 split 경계를 넘습니다. 이는 near-duplicate 검토 후보이지 물리 보드 ID 증거는 아닙니다.
- 433개 filename stem group 가운데 37개(219장)가 원본 train/validation/test 둘 이상의 split에 걸쳐 있습니다.
- 전체 이미지 contact sheet를 비교한 결과 같은 PCB가 반복해서 나타나 시각 board group 1개로 묶었습니다. 원본 source가 물리 보드 식별자를 주지 않아 source-verified board ID 수는 0입니다.
- 이 시각 그룹 하나가 원본 train, validation, test 모두에 걸칩니다. 따라서 보드 기준 누수 없이 train/validation을 나누거나 독립 test를 만들 수 없습니다.

새 train/validation split은 만들지 않았습니다. 기존 split은 출처 추적만을 위해 manifest에 남겼으며 학습·평가 split으로 사용할 수 없습니다. 다음 수집 단계에서 추가 물리 보드를 확보하고 보드별 ID를 확인해야 합니다.

상세 쌍 목록은 [`pcb-component-detection-duplicate-report.json`](../data/component_training/manifests/pcb-component-detection-duplicate-report.json), 그룹 교차 내역은 [`pcb-component-detection-board-leakage-report.json`](../data/component_training/manifests/pcb-component-detection-board-leakage-report.json), filename group은 [`pcb-component-detection-curation-manifest.json`](../data/component_training/manifests/pcb-component-detection-curation-manifest.json)을 참고하세요.
