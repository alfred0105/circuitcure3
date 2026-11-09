# Phase 21.3.2-C.1 — 검증 결과

## 자동 분석 결과

- 원본 이미지: 1,410
- filename stem 그룹 및 대표 이미지: 433 / 433
- split 경계를 가로지르는 filename 그룹: 37 (219장)
- pHash 후보 cutoff 14 및 ORB/RANSAC 비교쌍: 660; 그중 split metadata 충돌 228쌍
- ORB inlier ≥20 및 inlier ratio ≥0.7: 425쌍; 우선 overlay 최대 100장
- exact SHA-256 duplicate 그룹: 기존 C 분석에서 0
- source annotation box: 11,119, 기존 좌표 검사 valid 11,119 / invalid 0
- geometry 휴리스틱 검토 후보: 1,430 box 행 (edge-touch 1,427, 매우 작은 area 3); 확정 오류 0
- 빈 annotation: 114, 모두 자동 시각 검토에서 `PARTIAL_ANNOTATION_CANDIDATE`; manifest 상태 `EMPTY_UNREVIEWED` 114, `EMPTY_REVIEWED` 0
- 자동 클래스 gallery: 8개 클래스 각 40개, `Resestor` 1개
- 사람 검토: 이미지 0, 그룹 0, 클래스 결정 0, box 판정 0
- source label 수정 및 pseudo-label 승격: 0
- Dataset A/B 승인: 각각 0, 독립 Board-Level split: 생성 불가

## 실행 및 한계

`scripts/analyze_kaggle_component_c1.py`가 전체 이미지 manifest를 읽어 그룹 대표, 메타데이터, pHash/ORB 후보, inlier overlay, 클래스 gallery, 114장 검토 queue를 생성합니다. 이 작업은 원본 이미지·annotation을 변경하지 않습니다. 자동 매칭은 후보 검색일 뿐 물리 보드 식별 검증이 아닙니다.

로컬 Python 3.12 CI 명령 결과: 기본 suite 392 passed, 1 skipped, 48 deselected; integration 33 passed, 1 skipped; stress 11 passed; crash-injection 3 passed; native OCR 필수 job 1 passed. Integration skip 한 건은 `models/pcb_component_detector.onnx`가 작업 디렉터리에 없어 발생했습니다. GitHub Actions run `42187766019`은 코드 커밋 `902cad5`에서 Python 3.11/3.12, ngspice integration, bounded stress, isolated crash injection, native OCR을 포함해 성공했습니다.

이 환경에 배포 ONNX가 없고 남은 실험 체크포인트가 이 데이터에 대한 승인된 검토 모델이 아니어서, 모델 기반 누락 라벨 후보를 생성하지 않았습니다. CI 테스트는 실행 후 결과를 이 문서와 manifest에 기록합니다. 데이터셋 승인 전에는 DataLoader smoke test와 training을 실행할 수 없습니다.

세부 수치는 [`pcb-component-detection-c1-identity-annotation-report.json`](../data/component_training/manifests/pcb-component-detection-c1-identity-annotation-report.json) 및 [`pcb-component-detection-c1-class-review.json`](../data/component_training/manifests/pcb-component-detection-c1-class-review.json)에 연결됩니다.
