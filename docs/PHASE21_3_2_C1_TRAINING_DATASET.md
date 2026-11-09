# Phase 21.3.2-C.1 — 학습 데이터셋 준비

## 세 dataset의 상태

| Dataset | 승인 이미지 | 상태 및 근거 |
|---|---:|---|
| A — Canonical 10-Class | 0 | 원본 9개 클래스의 배포 클래스 mapping과 annotation 완전성이 검증되지 않음 |
| B — Source-Only Experimental | 0 | 클래스 의미·box 정합성 및 물리 보드 group split 검증 미완료 |
| C — Annotation Recovery | 1,410 검토 대상 | 원본 라벨을 보존. 빈 라벨 114장은 모두 populated PCB로 보여 `PARTIAL_ANNOTATION_CANDIDATE`로 사람 검토 대기; 정답 승격 없음 |

학습 가능한 파일 목록은 아직 만들지 않았습니다. 1,410장을 모두 보류 처리한 채로 감사만 끝낸 것이 아니라, C dataset에서 검토 단위를 filename group으로 정리하고 유사 보드·split 충돌·class별 box·빈 annotation 작업 목록과 시각 자료를 실제 생성했습니다. 그런데도 train 데이터 승인에 필요한 정답성은 추정하지 않았습니다.

37개 stem 충돌 그룹은 기존 split 누수를 확인합니다. 대표 이미지 pHash/ORB 후보를 포함하면 660개 비교 대상이 생겼습니다. 물리 보드 확인이 0개여서 안전한 train, development validation, sealed test 분할은 불가능합니다. 임의의 image-level split이나 source split을 독립 평가로 사용하지 않습니다.

따라서 Ultralytics DataLoader smoke test와 smoke training을 실행하지 않았습니다. 빈 split을 성공으로 위장할 수 있어 YOLO YAML도 생성하지 않았습니다. 기존 ONNX, frontend/backend, 모델 체크포인트는 수정하지 않았습니다.

구체적인 다음 작업: 사람이 114개 빈 라벨 우선 항목과 충돌 그룹을 검토하고, 반복 촬영 이미지가 실제 동일 보드인지 기록해야 합니다. 독립 보드 여러 개가 확인된 뒤에만 board-level split을 만들 수 있습니다. 이후 box 감사에서 통과한 image와 원본 클래스 mapping 증거가 생기면 각각 Dataset B 또는 A의 allowlist에 추가합니다.
