# Phase 21.3.2-C — Kaggle PCB 데이터 정제

## 결과

Kaggle/DatasetNinja 원본 1,410장을 모두 25장씩 나눈 Contact Sheet 57페이지로 시각 검토했습니다. 1,410장 모두 실장된 PCB 이미지이며, 같은 PCB의 반복 촬영·회전·프레이밍 변형으로 보이는 한 개의 시각 보드 그룹으로 묶었습니다. 이 그룹은 출처가 제공한 물리 보드 ID가 아니므로 `board_id=BOARD_ID_UNRESOLVED`를 유지합니다.

| 항목 | 수량 |
|---|---:|
| 전체 이미지 | 1,410 |
| 도메인 적합 PCB 이미지 | 1,410 |
| 도메인 기준 제외 | 0 |
| 도메인 검토 보류 | 0 |
| 물리 보드 ID 출처 확인 | 0 |
| 시각적으로 묶은 보드 그룹 | 1 |
| 학습 가능한 이미지 | 0 |
| 학습 보류 이미지 | 1,410 |

도메인 적합은 주 학습 후보가 될 수 있다는 뜻입니다. 라벨과 split 품질이 승인되었다는 뜻은 아닙니다. 1,410장 모두 manifest에 포함 여부와 보류 사유를 기록했으며 원본 파일을 변경하지 않았습니다.

## 라벨 검토

전체 source annotation은 11,119개이고 좌표 범위 검사에서 유효한 상자는 11,119개, 잘못된 상자는 0개였습니다. 비어 있는 annotation 114장은 연락처 시트에서 보드와 실장 부품이 보였습니다. 이를 빈 PCB나 negative sample로 세지 않고 `PARTIAL_ANNOTATION`으로 기록했습니다. 나머지 1,296장의 source annotation은 파일에 상자가 있어도 모든 대상 부품을 빠짐없이 라벨링했는지 확인되지 않아 `ANNOTATION_STATUS_UNKNOWN`입니다.

## 파일

- 이미지별 검토 및 상자 Manifest: [`pcb-component-detection-curation-manifest.json`](../data/component_training/manifests/pcb-component-detection-curation-manifest.json)
- 품질 요약: [`pcb-component-detection-quality-report.json`](../data/component_training/manifests/pcb-component-detection-quality-report.json)
- 제외·보류 목록: [`pcb-component-detection-excluded-image-report.json`](../data/component_training/manifests/pcb-component-detection-excluded-image-report.json)
- 실행 검증 결과: [`pcb-component-detection-validation-results.json`](../data/component_training/manifests/pcb-component-detection-validation-results.json)
- Phase 21.3.2-B source registry 연결: [`phase21-3-2-b-source-registry.json`](../data/component_training/manifests/phase21-3-2-b-source-registry.json)
- 원본 취득 기록: [`pcb-component-detection-acquisition-manifest.json`](../data/component_training/manifests/pcb-component-detection-acquisition-manifest.json)
- 재현 스크립트: [`curate_kaggle_component_dataset.py`](../scripts/curate_kaggle_component_dataset.py)
- Contact Sheet 생성기: [`create_component_contact_sheet.py`](../scripts/create_component_contact_sheet.py)
- **현재 Contact Sheet는 Git 무시 대상인 로컬 processed 폴더** `data/component_training/processed/kaggle-review-sheets-annotated/`에 있습니다.

원본 클래스 수와 데이터 설명은 [DatasetNinja의 PCB Component Detection 항목](https://datasetninja.com/pcb-component-detection)에 대조했습니다. 이 외부 설명은 클래스 의미나 이미지별 annotation 완전성을 증명하지 않습니다.
