# Phase 21.3.2-B — 데이터 품질 및 평가 안전성

## PCB-DSLR part 1

- ZIP archive: 281 members, 261 files, 725,793,819 uncompressed bytes; safe path/type/expanded-size checks passed.
- 20 source PCB folders; 87 JPEG images, 87 annotation TXT files, 87 masks. All images are 4928×3280.
- 2,132 source IC rotated boxes. Bounds, positive width/height, finite coordinates, image/label/mask pairing and duplicate row checks passed. The box center/size/angle format was confirmed against the publisher API. The export uses axis-aligned bounding boxes enclosing each rotated rectangle.
- Source says 165 different PCBs for the complete dataset; part 1 contains 20 source PCB IDs. The other seven archives remain undownloaded.
- Exact duplicates inside the 87 images: 0. pHash Hamming ≤4 pairs: 0. Phase A cross-dataset exact and pHash ≤4 overlaps: 0.
- Human visual contact-sheet review covered all 87 images and all generated annotation overlays. All were populated PCB images; no domain misclassification was identified. The source boxes remain marked unadjudicated by the publisher.
- Split by board ID: 14/3/3 train/validation/test board groups; leakage 0.
- License scope: local non-commercial research only. No commercial or redistribution right is inferred.

The single-class IC dataset is **`TRAINING_READY` within this restricted experiment scope**. It is not a complete annotation source for the deployed ten-class detector, and must not be mixed with 10-class data as ordinary background.

## Kaggle and WACV checks

Kaggle/DatasetNinja contains 1,410 unique exact image hashes and 11,119 boxes. pHash found 19 candidate pairs at distance ≤2 and 74 at ≤4; these were not auto-deleted. There are 433 stem groups, 37 crossing source splits. Against Phase A there are 0 exact and 0 pHash≤4 matches. Its 114 empty annotation files are not reviewed negatives.

WACV has 53 image files, 47 XML-paired, 6 unmatched alternate encodings, and 18,201 XML objects including text and board artifacts. Rights and class coverage remain unresolved. FPIC's SMD rows lack component classes; OCR/VTP are different annotation tasks.

## Loader/inference and evaluation

Ultralytics 8.4.203 parsed the single-class YAML and loaded a batch from each split: train 59 images, validation 14, test 14; each first batch had shape `(2, 3, 320, 320)`. Existing Phase A 10-class YAML also loaded train 77 / validation 24 images.

A Phase A ONNX model successfully ran CPU inference on a PCB-DSLR 4928×3280 image (~314 ms). This confirms runtime compatibility only; it is not an accuracy comparison. No training or benchmark was run.

`scripts/evaluate_component_domain.py` now counts only completed detector inferences. Failures are separate from the false-positive denominator. Reviewed bare-PCB and breadboard evaluation sets are still empty, so those rates are not reported.


저장소의 기본 비통합 pytest 선택 조건은 365 passed / 1 skipped / 48 deselected로 통과했습니다. 48개 deselected 테스트는 integration, stress, crash-injection mark가 붙은 별도 CI jobs 대상입니다.
