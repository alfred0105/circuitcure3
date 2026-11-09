# Phase 21.3.2-B — 학습 준비도 결정

## 결정

새로운 PCB 데이터와 annotation을 실제로 확보했습니다. PCB-DSLR archive part 1은 **비상업 연구용 단일 `integrated_circuit` detection 실험에 대해 `TRAINING_READY`**입니다. canonical 10-class detector용으로 승인된 신규 이미지 수는 0입니다. 부분 라벨을 전체 10개 class의 학습 배경으로 오인하지 않도록 이 dataset은 one-class 실험에만 연결했습니다.

| 지표 | 결과 |
|---|---:|
| 신규 다운로드 image files | Kaggle 1,410 + WACV 53 + PCB-DSLR 87 = **1,550** |
| annotation-paired images | 1,410 + WACV 47 + PCB-DSLR 87 = **1,544**; WACV 6장은 annotation 미매칭 |
| PCB-DSLR part 1 | 87 images, 2,132 IC boxes, 20 source PCB IDs |
| one-class train / validation / sealed test | 59 / 14 / 14 images; 14 / 3 / 3 board IDs |
| PCB-DSLR board-level split leakage | 0 (PASS) |
| PCB-DSLR exact / pHash≤4 duplicates | 0 / 0; Phase A와 cross overlap 0 / 0 |
| canonical 10-class 신규 추가 | 0 images |
| Kaggle empty labels | 114 `EMPTY_UNREVIEWED`, export/negative에서 제외 |
| WACV unpaired image files | 6; 라이선스도 unresolved |
| 기존 Phase A 유지 | 130 images / 3,300 boxes; filtered train 77 / validation 24 |
| ONNX compatibility | Phase A ONNX로 신규 DSLR image CPU inference 성공; accuracy claim 없음 |
| Mac storage | raw/external 약 2.4 GiB, 마지막 free space 약 136 GiB |

## `TRAINING_READY` scope

`configs/component_training/pcb-dslr-part1-ic.yaml`은 deployed 10-class 순서를 바꾸지 않는 isolated 1-class experiment입니다. PCB-DSLR 이용 조건은 non-commercial research only이며, 상업적 이용 및 raw redistribution은 허용된 것으로 간주하지 않습니다. 87장의 image/annotation/mask pair와 integrity를 확인했고, 도메인 contact-sheet review와 source-provided board ID를 기록했습니다. 모든 IC source box를 단일 canonical experiment class로 변환했으며 board-level split은 통과했습니다. 따라서 **1-class 연구 목적에 한해** 학습 가능한 YOLO YAML과 DataLoader가 준비됐습니다.

## 범위 한계

Kaggle의 1,410 images와 WACV의 47 XML-paired images는 확보됐지만 canonical 10-class 학습에 넣지 않습니다. Kaggle은 board identity와 source split leakage가 미해결이며, WACV는 라이선스·미매핑 class가 있습니다. PCB-DSLR에는 IC 외 부품 라벨이 없으므로 10-class dataset의 빈 배경처럼 혼합하지 않습니다. 이 확장은 전체 배포 모델을 비교할 완전한 10-class 데이터 증대를 제공하지 않으며, IC one-class 연구와 향후 IC class 성능 개선에 쓸 수 있습니다.

검토된 bare PCB 및 breadboard hard-negative set은 신규 출처에 없습니다. Phase A 수동 domain review에는 breadboard 10장, loose component collection 19장이 있었고 bare PCB hard negative는 0장입니다. Kaggle의 114 empty labels는 “빈 PCB” 이미지 수가 아닙니다.

## Smoke / 검증 결과

- PCB-DSLR train/validation/test DataLoader: 59/14/14 images, batch `(2, 3, 320, 320)` 성공.
- 기존 Phase A 10-class DataLoader: train 77 / validation 24 성공.
- Phase A ONNX의 신규 PCB image inference 성공; accuracy 지표 산출 안 함.
- 모델 재학습 및 성능 비교는 실행하지 않았습니다.
- 저장소 CI의 기본 Python job과 같은 비통합 pytest 선택 조건: **365 passed, 1 skipped, 48 deselected**. 경고는 기존 zip duplicate-path fixture에서 발생했으며 테스트는 통과했습니다.
- 신규 PCB-DSLR source manifest: [pcb-dslr-part1-ic-manifest.json](../data/component_training/manifests/pcb-dslr-part1-ic-manifest.json).
- 검토용 4페이지 contact sheet 및 review index: 로컬 ignored 경로 `data/component_training/processed/pcb-dslr-part1-ic-v3/contact-sheets/`.

전체 학습랩과 결과는 [Phase 21.3.2 학습랩](PHASE21_3_2_TRAINING_LAB.md)에 연결되어 있습니다.
