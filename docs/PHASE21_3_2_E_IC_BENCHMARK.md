# Phase 21.3.2-E — PCB-DSLR IC Benchmark

## Protocol

- Dataset: `pcb-dslr-part1-ic-research-v1`, one class `integrated_circuit`.
- Board-grouped split: train 59 images / 14 boards; validation 14 / 3 boards; previously exposed holdout 14 / 3 boards.
- The holdout was not used for model predictions or selection in Phase E. The automated dataset-integrity audit verified its stored hashes and board split.
- Both models started from their pretrained checkpoint and trained with MPS, 640 px, batch 2, 50 epoch cap, patience 15, seed 42, workers 2, cache off, AMP off, `optimizer=auto`, same Ultralytics default train-only augmentation.
- Best checkpoint selection: validation mAP50–95. Evaluation: confidence .25, raw-head NMS IoU .45, matching IoU .50, AP IoU .50–.95. NMS-free uses the one-to-one detections without an extra NMS.
- Small means box area < `32 × 32` pixels after scaling to a 640 px model input. For AP-small, predictions matched to large ground truths are ignored rather than counted as false positives.

## Training

| Model | Pretrained checkpoint SHA-256 | Epochs / best | Best validation P / R / mAP50 / mAP50–95 (Ultralytics operating point) | Peak train GPU memory shown by trainer |
|---|---|---:|---|---:|
| YOLOv8s | `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a` | 34 / 19 | .664 / .749 / .788 / .502 | 2.27 GiB |
| YOLO26s | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` | 45 / 30 | .597 / .771 / .771 / .531 | 2.36 GiB |

Ultralytics `optimizer=auto` resolved to AdamW for both models (same training-size branch and class count); architecture-specific loss terms differ: YOLOv8 logs box/cls/DFL, YOLO26 logs box/cls/L1. The trainer’s validation P/R operating point is separate from the fixed-confidence P/R below.

## Fixed-confidence validation results

| Metric | YOLOv8s raw | YOLO26s raw | YOLO26s NMS-free |
|---|---:|---:|---:|
| Precision | 0.4695 | 0.4121 | 0.4099 |
| Recall | 0.8562 | 0.9346 | 0.9216 |
| F1 | 0.6065 | 0.5720 | 0.5674 |
| mAP50 | 0.7889 | 0.7699 | 0.7787 |
| mAP50–95 | 0.5022 | **0.5346** | 0.5227 |
| TP / FP / FN | 131 / 148 / 22 | 143 / 204 / 10 | 141 / 203 / 12 |
| Small GT / TP / FN | 138 / 116 / 22 | 138 / 128 / 10 | 138 / 126 / 12 |
| Small recall | 0.8406 | **0.9275** | 0.9130 |
| Small AP50 | **0.7536** | 0.7262 | 0.7430 |
| Small AP50–95 | 0.4618 | **0.4875** | 0.4841 |
| Mean matched box IoU | 0.8074 | **0.8210** | 0.8152 |
| Mean absolute XYXY error | 5.65 px | **4.97 px** | 5.28 px |

YOLO26s raw gained 12 TP and lost 12 FN against YOLOv8s at the chosen threshold, while adding 56 FP. It improves mAP50–95 and recall, including small recall, but not mAP50, precision, F1, or small AP50. Confidence distributions and PR/confusion plots are in the ignored local validation output directories.

## Per-board recall at confidence .25

| Publisher board | GT | YOLOv8s | YOLO26s raw | YOLO26s NMS-free |
|---|---:|---:|---:|---:|
| pcb14 | 50 | 0.820 | 0.840 | 0.840 |
| pcb16 | 8 | 0.875 | 0.875 | 0.875 |
| pcb17 | 95 | 0.874 | 0.989 | 0.979 |

Only three boards are represented. These rows describe this validation sample and do not establish PCB-wide generalization.

## Historical baseline and test-set handling

Phase D’s historical test figures remain in `experiments/component_recognition/results/phase21-3-2-d-ic-benchmark.json` (YOLOv8s P .510, R .649, mAP50 .576, mAP50–95 .264, small recall .651). They are marked previously exposed. Phase E did not re-evaluate that holdout. A new independent PCB-board holdout is needed to confirm generalization.
