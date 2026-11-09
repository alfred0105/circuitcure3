# Phase 21.3.2-E — Kaggle Source-Only Development Comparison

## Results

YOLOv8s and YOLO26s were trained from their respective pretrained checkpoints for the same five-epoch cap on the same source-only Kaggle development split, with Ultralytics 8.4.203, PyTorch 2.15.0, Python 3.12.14, Apple M5 MPS, 640 px, batch 2, seed 42, workers 2, AMP off, and patience 3. Best checkpoints were selected by validation mAP50–95 (YOLOv8s epoch 5; YOLO26s epoch 4). The evaluated validation set has 261 images and 2,263 source boxes.

| Metric | YOLOv8s | YOLO26s |
|---|---:|---:|
| Precision | 0.965 | 0.977 |
| Recall | 0.947 | 0.925 |
| mAP50 | 0.971 | 0.963 |
| mAP50–95 | 0.656 | 0.633 |

YOLO26s was close to YOLOv8s on this short run, with slightly higher precision and lower recall and AP. These are descriptive development scores, not a model selection basis for production. The nine source class names were preserved without merging. `Resestor` has no validation examples, so its metrics are null and no class performance can be inferred for it.

## Dataset and interpretation limits

This is an exploratory source-label comparison, not a clean physical-board benchmark. The 1,410 candidate images remain `HUMAN_REVIEW_REQUIRED`; 1,296 non-empty-annotation images entered training or validation, while 114 empty annotations were excluded and were not used as negatives. The C.1 visual grouping placed 1,035 images in train (44 connected visual components / 344 filename groups) and 261 in validation (75 components / 87 filename groups). Physical PCB identities are unverified, and no independent sealed test is available. Source annotations may be incomplete or incorrect. Scores measure agreement with these source annotations only, and must not be directly compared to the single-class PCB-DSLR experiment.

No images were manually reviewed or approved for this phase. Original source data and labels remain unchanged.

## Reproduction

- Training runs: YOLOv8s `20261106-184118-08e6b8` (Phase D) and YOLO26s `20261106-193555-37ad49` (Phase E).
- Split and class details: [`phase21-3-2-d-kaggle-experiment.json`](../experiments/component_recognition/results/phase21-3-2-d-kaggle-experiment.json).
- YOLO26s evaluation JSON and run artifacts are under the ignored local `data/component_training/runs/20261106-193555-37ad49/phase-e/kaggle-development/` directory.
- Both runs used the same nine-class source contract and the same prepared `kaggle-source-only-yolov8s-dev-v1` data YAML.
