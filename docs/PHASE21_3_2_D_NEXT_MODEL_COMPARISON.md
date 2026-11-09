# Phase 21.3.2-E 비교 실험 준비

Phase E should compare YOLOv8s with YOLO26s using the same PCB-DSLR version, manifest, board split, class contract, image size, confidence threshold, IoU metrics and inference protocol. Do not use the sealed test repeatedly to pick a model; choose architecture/settings using train and validation, then run each locked test protocol once under a predeclared comparison plan.

The saved contract is [phase21-3-2-d-ic-yolov8s.yaml](../experiments/component_recognition/configs/phase21-3-2-d-ic-yolov8s.yaml):

- Dataset version: `pcb-dslr-part1-ic-research-v1` (Zenodo record 3886553 archive part 1)
- Board split: train 14 boards / 59 images; validation 3 / 14; sealed test 3 / 14
- class: `integrated_circuit`; seed 42; input 640px; batch 2; 50 epoch cap; patience 15; optimizer auto; MPS; AMP off
- Fixed reporting: confidence .25; NMS IoU .45; metric IoU .50; mAP50 and mAP50-95
- Also report per-board recall, small-object recall, TP/FP/FN, latency, peak memory and checkpoint size

This Phase did not install or test YOLO26s and did not claim a YOLOv8s-vs-YOLO26s result. Before Phase E, separately verify YOLO26 package/checkpoint compatibility, export output layout and ONNX Runtime decoder behavior. Preserve the single-class research contract and non-commercial source license.
