# Phase 21.3.2-B — 도메인 필터링 및 manifest

기본 10-class 학습에는 `POPULATED_PCB`, `PARTIALLY_POPULATED_PCB`, `PCB_COMPONENT_CLOSEUP`만 허용합니다. `BARE_PCB`, `BREADBOARD`, `ISOLATED_COMPONENT`, `LOOSE_COMPONENT_COLLECTION`, `UNKNOWN`은 별도 검토 전 학습하지 않습니다. 빈 라벨은 negative sample로 추정하지 않습니다.

## PCB-DSLR part 1 — 검토 완료 및 학습 준비

87개 원본 image를 25장씩 4페이지 contact sheet로 검토했습니다. 각 썸네일에는 source PCB ID, image ID, source box count와 변환 bounding box overlay를 표시했습니다. 확인한 87장 모두 populated PCB이며 단독 부품/브레드보드는 없었습니다. 각 PCB 폴더 ID를 source board group으로 유지했습니다.

모든 이미지마다 annotation과 mask가 존재하며 해상도는 4928×3280입니다. annotation 2,132개 모두 숫자/유한값·양의 면적·회전 후 경계·중복 여부 검사를 통과했습니다. 각 image/annotation/mask SHA-256을 manifest에 저장했습니다. Source label의 IC-only 범위는 별도 실험 설정에만 사용합니다.

| 분할 | source board IDs | images | IC boxes |
|---|---:|---:|---:|
| Train | 14 | 59 | 1,825 |
| Development validation | 3 | 14 | 153 |
| Sealed test | 3 | 14 | 154 |

Split은 20개 source PCB ID에 대해 안정적인 hash 기준으로 배정했습니다. 각 PCB의 촬영 각도/recording은 같은 split에 두었고 board-level leakage는 0입니다. Test는 split 검증 및 loader smoke 외 모델 선택/튜닝에 사용하지 않습니다.

## 다른 후보의 학습 제외

- Kaggle 1,410행은 `PCB_COMPONENT_CLOSEUP`, `AUTO_CLASSIFIED`, medium confidence입니다. 114개 empty annotation은 `EMPTY_UNREVIEWED`; YOLO export에서 제외합니다. 37/433 filename stem group이 source split을 넘고 physical board identity가 없습니다.
- WACV 53개 image file 중 47개 XML pair, 6개 label 누락 alternate encoding. 공식 사용권 불명 및 text/pads/pins/unknown 범주 때문에 학습에서 격리했습니다.
- 기존 Phase A 수동 리뷰는 76 populated PCB, 12 partially populated PCB, 13 close-up을 포함하며 10 breadboard / 19 loose-component collection을 제외했습니다.

총 다운로드 이미지 파일은 1,463(Kaggle 1,410 + WACV 53), PCB-DSLR part 1 87장을 추가하여 **1,550장**입니다. WACV의 6개 annotation 미매칭 alternate encodings를 빼면 1,544장의 image/label pair가 있습니다. Kaggle의 114 empty label은 pair 수에는 들지만 학습 export에는 넣지 않습니다.

## 재현 가능한 산출

- `scripts/prepare_pcb_dslr_candidate.py`: 회전 IC box 변환, 좌표·중복 검증, source board ID 기준 split, symlink 기반 YOLO export.
- `configs/component_training/pcb-dslr-part1-ic.yaml`: 배포 10-class 순서와 분리된 1-class 연구 실험.
- `data/component_training/manifests/pcb-dslr-part1-ic-manifest.json`: image별 출처·분류·review·license scope·board group·split·해시·annotation.
- `scripts/create_component_contact_sheet.py`: contact sheet + review index. 로컬 산출물은 gitignore된 `data/component_training/processed/pcb-dslr-part1-ic-v3/contact-sheets/`에 있습니다.

기존 배포 10-class YAML은 변경하지 않았습니다. Kaggle/WACV 후보는 발견 및 점검 대상이지만 주 학습 목록에 포함되지 않습니다.
