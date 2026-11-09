# Kaggle PCB source-only 9-class 탐색 실험

## 데이터 구성

- Source: Kaggle PCB Component Detection / DatasetNinja C.1 Manifest
- 원본: 1,410장, 11,119 source box, 9개 원본 클래스
- 빈 Annotation 114장은 학습에서 제외했다. 빈 라벨을 negative로 쓰지 않았다.
- 사용: 라벨이 1개 이상인 1,296장, 11,119 box
- 1,410장 모두 C.1에서 `POPULATED_PCB` 자동 후보이나 `HUMAN_REVIEW_REQUIRED`이며, 이 Phase에서 수동 판정은 하지 않았다.
- source 9개 class를 그대로 두고 클래스 병합은 하지 않았다.

원래 train/validation/test 이름은 분할에 쓰지 않았다. Filename Stem, 선언된 augmentation family, exact image SHA-256 중복 및 모든 C.1 ORB/RANSAC 후보 쌍을 connected component로 묶었다. 가장 큰 영상 프레임 component를 train에 고정하고 나머지 component 중에서 development split을 구성했다. Train 1,035장/44 component/344 filename group, development 261장/75 component/87 filename group이다. 개발 split은 물리 보드가 아닌 보수적 시각 중복 그룹을 분리한다. 물리 보드 ID는 여전히 미확정이다.

`Resestor`는 source 전체 3개 box가 모두 train 쪽 한 연결 그룹에 있어 development 표본이 없다. 해당 클래스의 P/R/AP는 `null`이고 `NO_VALIDATION_SUPPORT`다.

## 학습과 결과

YOLOv8s pretrained 9-class 모델을 M5/MPS에서 640px, batch 2, seed 42, AMP off, 5 epochs로 탐색 학습했다. 5 epoch가 끝났고 best epoch는 5다. 이 짧은 개발 실험은 확정 모델이나 독립 평가가 아니다.

| Metric | Development, source label |
|---|---:|
| Precision | 0.965 |
| Recall | 0.947 |
| mAP50 | 0.971 |
| mAP50-95 | 0.656 |

위 P/R은 Ultralytics validation의 기본 confidence sweep에서 얻은 operating point이며 고정 confidence .25의 값이 아니다. mAP 평균은 validation 표본이 있는 8개 클래스 기준이고 `Resestor`는 제외된다.

| Class | Boxes | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|---:|
| Cap1 | 260 | 1.000 | 0.972 | 0.995 | 0.729 |
| Cap2 | 255 | 0.946 | 0.969 | 0.986 | 0.613 |
| Cap3 | 241 | 0.890 | 0.913 | 0.940 | 0.651 |
| Cap4 | 237 | 0.927 | 0.917 | 0.949 | 0.637 |
| MOSFET | 261 | 0.996 | 0.990 | 0.995 | 0.705 |
| Mov | 261 | 1.000 | 0.996 | 0.995 | 0.698 |
| Resestor | 0 | — | — | — | — |
| Resistor | 232 | 0.980 | 0.832 | 0.911 | 0.512 |
| Transformer | 516 | 0.980 | 0.986 | 0.994 | 0.705 |

클래스별 값은 [기계 판독 결과](../experiments/component_recognition/results/phase21-3-2-d-kaggle-experiment.json)에 있다. `Resestor`는 평가 표본이 없어서 점수를 내지 않았다. 나머지 클래스도 publisher source labels와의 agreement로만 표시한다.

이 결과를 높게 일반화하지 않는다. Source Annotation의 완전성이 미확정이고, 연결 component는 물리 보드 식별이 아니다. 독립 보드 test는 없으며 결과를 PCB-DSLR IC 점수와 비교하지 않는다. 모델은 `EXPLORATORY_SOURCE_LABELS` 계약만 가지며 deployed 10-class model에 합치지 않는다.

## 로컬 산출물

가중치와 Ultralytics plots는 `data/component_training/runs/20261106-184118-08e6b8/` 아래 로컬에만 있다. Curation Manifest, 원본 이미지·라벨은 수정하지 않았다. 학습/validation 실행 요약은 `experiments/component_recognition/results/phase21-3-2-d-kaggle-experiment.json`에 보존했다.
