# Phase 21.3.2-C.1 — PCB 보드 식별

## 범위와 결과

원본 1,410장을 433개 filename stem 그룹으로 묶고, 그룹별 대표 이미지, 이미지 수, 원본 split, 해상도, SHA-256, pHash, 클래스별 라벨 수를 기록했습니다. 그룹 overview와 split 충돌 그룹 비교용 연락판은 `data/component_training/processed/kaggle-c1-review/`에 생성됩니다.

37개 filename 그룹(219장)이 source split 경계를 넘습니다. 그룹명은 파일명 관계의 근거일 뿐 같은 물리 보드라는 증거로 승격하지 않았습니다. 한 장의 대표 이미지가 모든 소스 그룹을 대표한다고 가정하지 않습니다.

대표 이미지 간 pHash 거리 14 이하인 쌍을 좁힌 뒤 ORB 특징점과 RANSAC homography를 계산했습니다. 660쌍 중 228쌍은 source split 메타데이터가 서로 다르고, 425쌍은 inlier 20개 이상 및 inlier ratio 0.7 이상이었습니다. 상위 100쌍의 feature-match overlay를 생성합니다. Overlay, inlier 수와 비율은 검토 우선순위 자료입니다. 매우 높은 특징 일치도 역시 동일 capture·증강 관계와 물리 보드 관계를 단독으로 증명하지 않으므로 Board ID 결정은 모두 `UNRESOLVED`입니다.

SHA-256 동일성은 파일 byte가 동일한 경우에만 exact capture로 취급할 수 있습니다. 서로 다른 시점·각도·증강이 같은 실제 보드인지 확인한 사람 판정은 아직 0건입니다. 확인된 물리 PCB 수는 0으로 유지합니다.

## 검토 자료

- 그룹 대표 연락판: `data/component_training/processed/kaggle-c1-review/filename-group-overview/`
- split 충돌 37개 그룹: `.../split-collision-groups/`
- 유사도 후보 매칭 overlay: `.../similarity-pairs/`
- 원본 그룹 지표 및 후보: [`pcb-component-detection-c1-identity-annotation-report.json`](../data/component_training/manifests/pcb-component-detection-c1-identity-annotation-report.json)

물리 보드가 1개뿐이라고 가정할 근거도 아직 없습니다. 독립 Board-Level train/validation/test는 계속 차단하며, 추가 보드 확보 후 sealed test 구성이 필요합니다.
