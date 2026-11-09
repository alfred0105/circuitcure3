# Phase 21.3.2-C — Kaggle PCB 데이터 정제

## 결과

Kaggle/DatasetNinja 원본 1,410장을 25장씩 배치한 Contact Sheet 57페이지로 AI 시각 감사했습니다. 자동 분류 결과는 실장 PCB 후보 1,410장이지만 사람에 의해 확인된 이미지는 0장입니다. 하나의 유사 board group이 의심되지만, 미확정 후보일 뿐이며 `board_id=BOARD_ID_UNRESOLVED`를 유지합니다.

| 항목 | 수량 |
|---|---:|
| 전체 이미지 | 1,410 |
| AI 분류 PCB 도메인 후보 | 1,410 |
| 사람 확인 도메인 이미지 | 0 |
| 도메인 기준 제외 | 0 |
| 사람 검토 대기 | 1,410 |
| 물리 보드 ID 출처 확인 | 0 |
| 시각적으로 묶은 보드 그룹 | 1 |
| 학습 가능한 이미지 | 0 |
| 학습 보류 이미지 | 1,410 |

자동 도메인 분류는 최종 포함을 뜻하지 않습니다. 1,410장 모두 manifest에 자동 분류 상태와 사람 검토 필요 여부, 포함 보류 사유를 기록했으며 원본 파일은 변경하지 않았습니다.

## 라벨 검토

전체 source annotation은 11,119개이고 좌표 범위 검사에서 유효한 상자는 11,119개, 잘못된 상자는 0개였습니다. 비어 있는 annotation 114장은 자동 시각 감사에서 실장 부품이 보이는 것으로 분류됐지만 사람 확인은 없습니다. 따라서 빈 PCB나 negative sample로 세지 않고 `EMPTY_UNREVIEWED`로 남겼습니다. 나머지 1,296장도 라벨 파일만으로 전체 대상의 coverage가 증명되지 않아 `ANNOTATION_STATUS_UNKNOWN`입니다.

## 파일

- 이미지별 검토 및 상자 Manifest: [`pcb-component-detection-curation-manifest.json`](../data/component_training/manifests/pcb-component-detection-curation-manifest.json)
- 품질 요약: [`pcb-component-detection-quality-report.json`](../data/component_training/manifests/pcb-component-detection-quality-report.json)
- 제외·보류 목록: [`pcb-component-detection-excluded-image-report.json`](../data/component_training/manifests/pcb-component-detection-excluded-image-report.json)
- 실행 검증 결과: [`pcb-component-detection-validation-results.json`](../data/component_training/manifests/pcb-component-detection-validation-results.json)
- 사람 검토 입력 대기열: [`review-queue.csv`](../data/component_training/manifests/kaggle-component-review-queue/review-queue.csv)
- Phase 21.3.2-B source registry 연결: [`phase21-3-2-b-source-registry.json`](../data/component_training/manifests/phase21-3-2-b-source-registry.json)
- 원본 취득 기록: [`pcb-component-detection-acquisition-manifest.json`](../data/component_training/manifests/pcb-component-detection-acquisition-manifest.json)
- 재현 스크립트: [`curate_kaggle_component_dataset.py`](../scripts/curate_kaggle_component_dataset.py)
- Contact Sheet 생성기: [`create_component_contact_sheet.py`](../scripts/create_component_contact_sheet.py)
- **현재 Contact Sheet는 Git 무시 대상인 로컬 processed 폴더** `data/component_training/processed/kaggle-review-sheets-annotated/`에 있습니다.

원본 클래스 수와 데이터 설명은 [DatasetNinja의 PCB Component Detection 항목](https://datasetninja.com/pcb-component-detection)에 대조했습니다. 이 외부 설명은 클래스 의미나 이미지별 annotation 완전성을 증명하지 않습니다.
