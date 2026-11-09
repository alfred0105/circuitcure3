# Phase 21.3.2-B — 데이터 수집 감사

후속 Kaggle PCB 1,410장 정제 결과는 [Phase 21.3.2-C Kaggle Curation](PHASE21_3_2_C_KAGGLE_CURATION.md), [클래스 매핑](PHASE21_3_2_C_CLASS_MAPPING.md), [보드 누수](PHASE21_3_2_C_BOARD_LEAKAGE.md), [학습 준비 상태](PHASE21_3_2_C_TRAINING_READINESS.md)에 기록했습니다.

## 결과

실제 원본 이미지 1,463장을 새로 확보했습니다: Kaggle/DatasetNinja 1,410장과 PCB-WACV 53장. PCB-WACV 중 47장만 XML annotation과 짝이 있고, 나머지 6장은 같은 stem의 PNG/JPEG alternate encoding으로 확인됐습니다. 각 파일은 원본 영역에 보존했으며 Git에는 올리지 않습니다.

가장 중요한 추가 확보는 PCB-DSLR 공식 레코드의 **1번 분할 한 개**입니다. 708,903,524-byte ZIP에서 20개 source PCB ID, 87개 고해상도 PCB 이미지, 87개 IC annotation 및 87개 board mask를 확인했습니다. 2,132 IC bounding boxes를 단일 `integrated_circuit` 목표로 변환하고 source board ID로 split했습니다. 공식 이용 조건이 비상업 연구용이므로 **비상업 연구 실험에서만 사용**합니다. 나머지 7개 ZIP은 내려받지 않았습니다. [공식 PCB-DSLR Zenodo 레코드](https://zenodo.org/records/3886553)

| 출처 | 실제 확보 | 권리 / Annotation 상태 | 학습 상태 |
|---|---:|---|---|
| PCB-DSLR part 1 | 87 images, 2,132 IC boxes, 20 source PCB IDs | 공식 레코드가 비상업 연구용으로 사용을 제한. 4928×3280, IC rotated box 및 선택적 칩 문자열 | **`TRAINING_READY`**, 별도 single-class `integrated_circuit` 연구 실험에 한함 |
| Kaggle PCB Component Detection 변환본 | 1,410 images, 11,119 boxes, 114 empty labels | CC0 source metadata. physical board ID 미확인, source split stem leakage, canonical mapping 미완료 | `DOWNLOADED_NOT_TRAINING_READY` |
| PCB-WACV 2019 | 53 image files, 47 XML-paired images, 18,201 source XML objects | 공식 페이지에 라이선스 표시 없음; 여러 non-component labels 포함 | `LICENSE_REVIEW_REQUIRED` |
| FPIC PhysicalDB | SMD/OCR/VTP/metadata annotation archive만 | 페이지에 원본 라이선스 표시를 찾지 못함. SMD에는 class field 없음 | annotation-only, 학습 보류 |
| FICS-PCB | 미확보 | Trust-Hub의 익명 다운로드 및 권리 확인 안 됨 | access review |

세부 경로·해시·용량은 [source registry](../data/component_training/manifests/phase21-3-2-b-source-registry.json)에 있습니다. PCB-DSLR의 이미지·라벨별 도메인, split, SHA-256, annotation은 [PCB-DSLR manifest](../data/component_training/manifests/pcb-dslr-part1-ic-manifest.json)에 기록했습니다.

## PCB-DSLR format 및 이용 조건

출판사 API는 annotation의 앞 다섯 값이 `(cx, cy, width, height, angle)`이고 뒤 문자열은 선택적인 IC text라고 정의합니다. 앞 다섯 값은 OpenCV 회전 사각형으로 읽어 axis-aligned YOLO box로 감쌌습니다. 뒤 문자열은 OCR/설명 metadata로 보존하고 class로 쓰지 않습니다. 목표를 IC 하나로 한정하여 다른 부품 annotation 누락을 해당 실험의 positive/negative label로 오해하지 않습니다. 출처가 “all IC chips”를 box로 제공한다고 설명하지만 annotation은 source-provided, unadjudicated 상태로 계속 표기합니다. [공식 annotation API](https://github.com/cpra/mva15/tree/master/api/python)

Zenodo 설명은 non-commercial research use를 허용한다고 명시하고 license field는 별도 SPDX license를 제공하지 않습니다. 그래서 원본은 로컬 research use로만 제한하고 raw ZIP은 배포하지 않습니다. 상업적 사용, 제품학습, 공개 재배포에는 별도 권한 확인이 필요합니다.

## 다운로드·저장 공간

PCB-DSLR 1번 ZIP의 SHA-256은 `1700e25683fae1280da86052da619e59d9ed65d658e0b3fcfdc7e7196a5c2920`입니다. ZIP에는 281 members / 261 files / 725,793,819 expanded bytes가 있습니다. 다운로드 중 `.part` 파일 크기 708,903,524 bytes 및 SHA-256을 확인했고, ZIP 경로·타입·확장량을 검사한 후 안전하게 추출했습니다. 전체 데이터는 다운로드하지 않았습니다.

다운로드 및 추출물은 `data/component_training/raw/external/` 아래에 보존합니다. 마지막 확인 당시 `raw/external`은 약 2.3 GiB, 사용 가능한 디스크 공간은 약 133 GiB였습니다. 기존 Phase A 130 images / 3,300 boxes는 유지했습니다.

## 추가 검색 결과

[GEC Surat Roboflow](https://universe.roboflow.com/gec-surat/pcb-w4vdj-fxxx5)는 672 images와 Public Domain을 표시하지만 dataset export version이 0개였습니다. [DohaRagab PCB Components ALL](https://universe.roboflow.com/doharagab/pcb-components-all-hsy38)는 MIT라고 표시된 1,126장 후보지만 public export가 HTTP 403을 반환했습니다. [ParsaVictor의 458-board 프로젝트](https://github.com/ParsaVictor/pcb-component-detection-yolo26)는 Roboflow API key로 원본을 재구성한다고 설명해 접근하지 못했습니다. [Hugging Face 21-class 후보](https://huggingface.co/datasets/Arshia82sbn/pcb-object-detection-dataset)는 MIT를 표시하지만 실제 repo에는 README와 시각화 파일만 있고 이미지/라벨이 없습니다. PCB-AOI/PCB-IND/DeepPCB는 결함 탐지 자료라 부품 detector 데이터와 구분했습니다.

## 기존 문서와 연결

[Phase 21.3.2 학습랩](PHASE21_3_2_TRAINING_LAB.md), [기존 데이터 확장 감사](PHASE21_3_2_DATASET_EXPANSION.md)도 함께 확인하세요.
