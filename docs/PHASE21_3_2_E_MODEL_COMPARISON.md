# Phase 21.3.2-E — YOLOv8s vs YOLO26s

## Outcome

Both pretrained models were fine-tuned on the same 87-image PCB-DSLR IC dataset under Ultralytics 8.4.203, PyTorch 2.15.0, Python 3.12.14, Apple M5 MPS, 640 px, batch 2, seed 42, workers 2, AMP off, and the same default training augmentation. The Validation split contains 14 images from 3 publisher-identified boards. Checkpoint selection used validation mAP50–95. The 14-image, 3-board test split had already been exposed in Phase D; this phase made no test predictions or selection decisions from it. The automated manifest audit did hash-check its files for dataset integrity.

The YOLO26s checkpoint reached the higher validation mAP50–95 (0.535 vs 0.502) and higher small-IC recall (0.928 vs 0.841). At the fixed confidence 0.25 operating point, it also produced 56 more false positives, with lower precision and F1. Its CPU ONNX median was faster and the ONNX file smaller. Strict PyTorch/ONNX parity did not pass for any path across all 14 images, so YOLO26s remains an experimental candidate and neither ONNX is a deployment replacement.

| Validation metric | YOLOv8s raw | YOLO26s raw | YOLO26s NMS-free |
|---|---:|---:|---:|
| Parameters, full checkpoint | 11,135,987 | 9,948,638 | same YOLO26s checkpoint |
| FLOPs at 640 px | 28.64 G | 22.74 G | 22.74 G reported by Ultralytics profiler |
| Precision @ conf .25 | 0.470 | 0.412 | 0.410 |
| Recall @ conf .25 | 0.856 | 0.935 | 0.922 |
| F1 @ conf .25 | 0.606 | 0.572 | 0.567 |
| mAP50 | 0.789 | 0.770 | 0.779 |
| mAP50–95 | 0.502 | **0.535** | 0.523 |
| TP / FP / FN | 131 / 148 / 22 | 143 / 204 / 10 | 141 / 203 / 12 |
| Small IC recall | 0.841 | **0.928** | 0.913 |
| CPU ONNX median / p95 | 56.95 / 65.07 ms | 43.83 / 49.43 ms | 42.74 / 48.52 ms |
| ONNX size | 44.72 MB | 38.11 MB | 38.12 MB |
| Strict parity, complete images | 11 / 14 | 6 / 14 | 6 / 14 |

The NMS-free head is exported and evaluated from the same YOLO26s checkpoint, not treated as a separately trained model. See [IC benchmark](PHASE21_3_2_E_IC_BENCHMARK.md), [ONNX parity](PHASE21_3_2_E_ONNX_PARITY.md), [runtime results](PHASE21_3_2_E_PERFORMANCE.md), [Kaggle exploratory comparison](PHASE21_3_2_E_KAGGLE_BENCHMARK.md), and [selection decision](PHASE21_3_2_E_MODEL_SELECTION.md).

## Reproduction records

- Config: `experiments/component_recognition/configs/phase21-3-2-e-model-comparison.yaml`
- Local run IDs: YOLOv8s `20261106-191800-ccb464`; YOLO26s `20261106-192334-c62b65`
- Baseline comparison: `experiments/component_recognition/results/phase21-3-2-d-ic-benchmark.json`
- Dataset manifest SHA-256: `a17c9e050f781b5b6882999d4488e990f68cebe432d0ac28b7552b4cdc164b05`
- Run checkpoints, ONNX files, per-image outputs, curves, and overlays remain under ignored `data/component_training/runs/` and are not committed.

## Interpretation limits

The validation set contains only three boards, so board-level differences are descriptive and no stable confidence interval is claimed. Labels are publisher-provided and not independently adjudicated. The historical test results remain documented as previously exposed Phase D results; no new test predictions or model selection used them. MPS emitted nondeterministic-operation warnings, so seed 42 does not promise bit-identical reruns. The deployed detector, ONNX model, and decoder were not changed.

YOLO26 export behavior was checked against the [official YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/) and [official end-to-end detection guide](https://docs.ultralytics.com/guides/end2end-detection/).
