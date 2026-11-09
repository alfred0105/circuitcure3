# Phase 21.3.2-E — Model Selection

## Decision

**Do not replace the current deployed detector with YOLO26s in this phase. Keep YOLOv8s as the existing production family; retain YOLO26s as a research candidate for a larger board-grouped validation.** This is an experimental selection, not a deployment approval.

YOLO26s has a real validation advantage on mAP50–95 (+0.032) and small-IC recall (+0.087), and lower CPU ONNX median latency (-13.1 ms) with a smaller ONNX file (-6.61 MB). At confidence .25 it also has 56 additional false positives, lower precision (.412 vs .470), lower F1 (.572 vs .606), and lower mAP50 (.770 vs .789). Its small AP50 is lower (.726 vs .754), although small AP50–95 is higher (.487 vs .462). The 3-board validation set is too small to establish these differences as PCB-wide generalization.

The YOLO26 NMS-free path uses the same checkpoint. It has mAP50–95 .523, small recall .913, and fewer CPU postprocessing milliseconds than its raw path. It still has 203 false positives, and strict post-NMS parity did not pass across the full validation set. The one-to-many YOLO26 export also does not pass strict parity.

## Decision factors

| Factor | Observation | Effect on decision |
|---|---|---|
| IC mAP50–95 | YOLO26 raw .535 vs YOLOv8 .502 | Supports further YOLO26 research |
| Small-object recall | .928 vs .841 | Supports further YOLO26 research |
| False positives / precision | 204 / .412 vs 148 / .470 at confidence .25 | Material accuracy tradeoff |
| mAP50 / F1 | .770 / .572 vs .789 / .606 | YOLO26 not uniformly better |
| CPU ONNX median | 43.83 ms raw, 42.74 ms NMS-free vs 56.95 ms | YOLO26 is more efficient on this CPU test |
| PyTorch MPS median | 22.13 ms raw, 21.03 ms NMS-free vs 20.88 ms | Similar, slightly slower for YOLO26 raw |
| ONNX size | about 38.1 MB vs 44.7 MB | Favors YOLO26 |
| Strict ONNX parity | Y8 11/14; Y26 raw and one-to-one 6/14 complete matches | Blocks promotion pending investigation |
| Board count | 3 validation boards | Insufficient for a confident generalization claim |
| Kaggle source-only metrics | Exploratory labels and visual groups only | Cannot settle real-board selection |

Board recall improvements are concentrated on `pcb17` (YOLOv8 .874, YOLO26 raw .989); `pcb14` changes .820 to .840 and `pcb16` is .875 for both. That pattern reinforces the need for more distinct boards. No confidence interval is claimed because three board clusters do not support a stable one.

## Next comparison gate

Collect a separate board-grouped validation cohort with a clear source/license record, keep the previously exposed test out of model selection, and repeat the same automated metrics. Investigate confidence-threshold and coordinate differences on raw ONNX outputs before considering a deployment candidate. Keep deployment model, service decoder, and all existing product code unchanged until a later explicitly approved deployment decision.
