# Phase 21.3.2 — benchmark status

## Filtered domain development comparison (2026-11-06)

Both models were evaluated at confidence 0.25 / NMS IoU 0.45 on the same 24-image filtered validation set (705 source-provided annotations, 4 manual visual board groups) with 960px ONNX inference. Evaluation status is `SOURCE_LABEL_COMPARISON`; labels were not independently adjudicated.

| Model | Precision | Recall | mAP50 | mAP50-95 | TP / FP / FN | Breadboard FPR | Mean FP / breadboard image | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Existing `pcb-yolov8s` ONNX | 0.756 | 0.742 | 0.773 | 0.532 | 523 / 169 / 182 | 0.00 (0/10) | 0.00 | Reference baseline |
| Fresh YOLOv8n, 40 epochs, filtered | 0.401 | 0.233 | 0.204 | 0.104 | 164 / 245 / 541 | 0.10 (1/10) | 0.10 | Underperforms; do not promote |

The baseline median CPU latency was 124.9 ms (p95 143.8 ms, peak process RSS 1.55 GB); the candidate median was 60.7 ms (p95 76.1 ms, peak RSS 1.36 GB), but its detection quality is substantially worse. No reviewed bare-PCB images were available, so bare-PCB FPR and mean FP/image are N/A. Per-class support, including only 4 diode annotations, is too small for a stable class-level conclusion. The 10 breadboards support a narrow development check only.

## Historical pre-filter runs

All scores below were measured before the domain filter and included out-of-domain images. They are retained for history but are **ineligible** for the target-domain comparison.

| Model | Precision | Recall | mAP50 | mAP50-95 | status |
|---|---:|---:|---:|---:|---|
| `pcb-yolov8s-2026-07-11` existing ONNX | 0.498 | 0.612 | 0.585 | 0.334 | Mixed 13-image development validation; ineligible |
| YOLOv8n, 5 epochs, 640px, MPS | 0.412 | 0.168 | 0.098 | 0.055 | Pre-filter, ineligible |
| YOLOv8n, 40 epochs, 640px, MPS | 0.521 | 0.388 | 0.398 | 0.206 | Pre-filter, ineligible |

Original summary, retained for audit only:

| Model | Precision | Recall | mAP50 | mAP50-95 | Resistor recall | Capacitor recall | IC recall | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `pcb-yolov8s-2026-07-11` existing ONNX | 0.498 | 0.612 | 0.585 | 0.334 | 0.423 | 0.678 | 0.750 | Existing 13-image / 196-annotation development validation only |
| YOLOv8n, 5 epochs, 640px, MPS | 0.412 | 0.168 | 0.098 | 0.055 | 0.077 | 0.339 | 0.219 | Same source validation; provisional undertrained run, no improvement |
| YOLOv8n, 40 epochs, 640px, MPS | 0.521 | 0.388 | 0.398 | 0.206 | 0.154 | 0.492 | 0.656 | Same source validation; better than 5 epochs, below baseline |
| Expanded YOLOv8n | — | — | — | — | — | — | — | NOT_RUN: additional sources not downloaded/license-cleared |
| Small-object profile | — | — | — | — | — | — | — | NOT_RUN |

The old pre-filter comparison used the original 13-image / 196-annotation source validation at 960px and confidence 0.25. It is included only as a historical record and must not be used to infer filtered target-domain performance.

The old pre-filter 40-epoch model was also tested only on the old mixed validation and is not comparable to the filtered metrics above.

The source labels are unadjudicated. No test set independent of training/development exists. Physical board IDs are not verified by the source, and these results do not establish independent-board generalization.
