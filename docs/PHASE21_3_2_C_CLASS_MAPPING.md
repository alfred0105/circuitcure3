# Phase 21.3.2-C — 클래스 매핑

Source class 문자열을 보존하고 배포 모델의 10-class ID에 매핑하지 않았습니다. DatasetNinja의 클래스 설명은 대부분 의미를 `Unknown`으로 표시하고, `Resestor`는 3개 상자뿐입니다. 클래스 이름이나 개수만으로 의미를 추정하지 않았습니다.

| Source class | Annotation 수 | 결정 |
|---|---:|---|
| Cap1 | 1,295 | 커패시터 하위 유형 의미 미확인. `capacitor`로 병합 보류 |
| Cap2 | 1,264 | 커패시터 하위 유형 의미 미확인. `capacitor`로 병합 보류 |
| Cap3 | 1,199 | 커패시터 하위 유형 의미 미확인. `capacitor`로 병합 보류 |
| Cap4 | 1,074 | 커패시터 하위 유형 의미 미확인. `capacitor`로 병합 보류 |
| MOSFET | 1,295 | 기존 `integrated_circuit`에 매핑 금지 |
| Mov | 1,291 | 기존 `diode`에 매핑 금지 |
| Resestor | 3 | 오타인지 별도 의미인지 미확인. `Resistor`로 병합 보류 |
| Resistor | 1,151 | source 클래스 유지, canonical 대응 승인 안 됨 |
| Transformer | 2,547 | source 클래스 유지, canonical 대응 승인 안 됨 |

현재 확인된 source-only 학습 후보는 0장, canonical 10-class 후보도 0장입니다. 모든 source class가 보존된 차단 YAML은 [`pcb-component-detection-source-only-blocked.yaml`](../data/component_training/manifests/pcb-component-detection-source-only-blocked.yaml)입니다. `train: []`, `val: []`로 작성되어 Ultralytics 학습 입력으로 사용할 수 없습니다.

전체 클래스별 기계 판독 표는 [`pcb-component-detection-curation-manifest.json`](../data/component_training/manifests/pcb-component-detection-curation-manifest.json)의 `class_mapping`과 `summary.class_annotation_counts`에 있습니다. 기존 모델 class ID와 로더 계약은 변경하지 않았습니다.
