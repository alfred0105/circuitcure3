# Phase 21.3.2-E — ONNX Export and Parity

## Export result

All three exports completed with Ultralytics 8.4.203, passed `onnx.checker`, and opened in ONNX Runtime 1.31.1 using `CPUExecutionProvider`.

| Export | Head and postprocessing | Actual ONNX output | File size |
|---|---|---|---:|
| YOLOv8s raw | one-to-many; external NMS | `[1, 5, 8400]` | 44,719,456 bytes |
| YOLO26s raw | one-to-many; external NMS | `[1, 5, 8400]` | 38,109,825 bytes |
| YOLO26s NMS-free | one-to-one; no second NMS | `[1, 300, 6]` | 38,119,531 bytes |

The export shapes are observed from the actual ONNX Runtime sessions, not inferred from the model family. The one-to-one export uses XYXY boxes, confidence and class ID. The raw exports return the usual per-class candidate tensor. Ultralytics’ NMS-free prediction adapter was used for the one-to-one file; the production YOLOv8 decoder was not changed or given that output.

## Comparison protocol and tolerance

All 14 PCB-DSLR validation images were compared at confidence .25. The raw paths used NMS IoU .45; the NMS-free path applied no NMS. Detections were matched by class at IoU ≥ .80. A strict image passed only if detection counts matched, every matched confidence differed by at most .05, and maximum XYXY coordinate error was ≤ 4 px. Comparisons were run both PyTorch MPS vs ONNX CPU and PyTorch CPU vs ONNX CPU.

| Path | Full strict-match images, MPS vs ORT | Full strict-match images, CPU vs ORT | Outcome |
|---|---:|---:|---|
| YOLOv8s raw | 11 / 14 | 11 / 14 | **Not a full parity pass** |
| YOLO26s raw | 6 / 14 | 6 / 14 | **Not a full parity pass** |
| YOLO26s NMS-free | 6 / 14 | 6 / 14 | **Not a full parity pass** |

For YOLOv8s, matched boxes stayed close (minimum IoU .976, maximum confidence delta .023, maximum coordinate delta 2.92 px); the three strict failures came from detections near the .25 threshold being present on only one path. YOLO26s raw had more differences: minimum matched IoU .897, maximum confidence delta .072, maximum coordinate delta 5.34 px, and count differences on eight images. YOLO26s one-to-one matched boxes remained closer (minimum IoU .968, maximum confidence delta .058, maximum coordinate delta 1.51 px), but counts still differed on eight images, mostly near the .25 score threshold.

The same mismatches occurred in CPU PyTorch vs CPU ONNX, so they are not explained by MPS alone. At a diagnostic confidence .50, complete detection counts matched on 12/14 images for each raw model and 11/14 for NMS-free; this diagnostic does not replace the registered .25 comparison. Full per-image matched/unmatched boxes and deltas are stored under the ignored local `data/component_training/runs/20261106-192334-c62b65/phase-e/onnx-comparison/parity/` directory.

**Parity remains incomplete.** The exported graphs are valid and usable for research benchmarking, but this validation is not enough to qualify either YOLO26 ONNX path for deployment. The production artifact and decoder remain unchanged.

References: [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) and [end-to-end detection / export output shapes](https://docs.ultralytics.com/guides/end2end-detection/).
