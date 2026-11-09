# PCB-DSLR IC 단일 클래스 벤치마크

## 실험 요약

- Model: pretrained YOLOv8s, detection, 1 class `integrated_circuit`
- Dataset: PCB-DSLR archive part 1, publisher-provided IC boxes
- Train: 59 images / 14 board IDs / 1,825 boxes
- Validation: 14 / 3 / 153
- Sealed Test: 14 / 3 / 154
- Best checkpoint: local `data/component_training/runs/20261106-183126-06352e/weights/best.pt`
- Best epoch 19; stopped at epoch 34 of 50 by patience 15
- Device MPS, 640px, batch 2, seed 42, AMP off
- License: local non-commercial research only

## Metrics

| Split / setting | Precision | Recall | F1 | mAP50 | mAP50-95 | TP / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Validation, conf .25, IoU .5 | 0.470 | 0.856 | 0.606 | 0.789 | 0.502 | 131 / 148 / 22 |
| Sealed Test, conf .25, IoU .5 | 0.510 | 0.649 | 0.571 | 0.576 | 0.264 | 100 / 96 / 54 |

mAP50 and mAP50-95 use confidence-ranked predictions with IoU thresholds 0.50 and 0.50:0.05:0.95. Fixed-threshold Precision/Recall/F1 use confidence 0.25 and IoU 0.50. Ultralytics' best validation row at epoch 19 reported P 0.664, R 0.749, mAP50 0.788, mAP50-95 0.502; its P/R are the validator's selected operating point and are not the same as fixed confidence .25.

Sealed-test small IC recall at confidence .25 / IoU .5 was 0.651 (95/146 small boxes). It is lower than validation's 0.841 (116/138). Test has 3 source board IDs and only 154 source annotations. The observed validation-to-test drop and missed detections limit confidence in generalization.

## Resolution check (validation only)

640px inference: mAP50 0.789, mAP50-95 0.502, fixed-threshold precision 0.470, recall 0.856, small recall 0.841.

960px inference using the same 640-trained checkpoint: mAP50 0.702, mAP50-95 0.445, precision 0.306, recall 0.915, small recall 0.953. It recovered more small boxes but increased false positives and reduced mAP. Keep 640px for this experiment. The 960px run was validation-only; sealed test was not revisited.

## Runtime and artifacts

MPS validation median was approximately 23.6ms/image (14 images); p95 includes first-call effects and was 397.7ms. Process peak RSS was about 1.96 GB for validation. Sealed-test median was 24.5ms/image and p95 347.0ms. These small-split timings are indicative, not production SLOs.

Generated local artifacts include training `results.csv`, loss/metric curves and Ultralytics PR/confusion plots; fixed-threshold PR curve, confusion counts and per-image comparisons are under the run's `evaluation/` folders. The source-level `experiment.json` records config, hashes, versions, selected epoch and checkpoint checksums. These files and weights are ignored by Git.

**Interpretation:** source-label agreement for a non-commercial research experiment only. The labels were supplied by the dataset publisher, are not independently adjudicated in this Phase, and may not establish complete object ground truth. Do not treat these metrics as deployment performance.
