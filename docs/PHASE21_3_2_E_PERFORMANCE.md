# Phase 21.3.2-E — Model Size and Inference Performance

## Measurement protocol

Each path used batch 1 at 640 px and the same 14 in-memory validation images, cycled in a fixed order. The benchmark made 10 warm-up calls, recorded the first call separately, then timed 100 calls. PyTorch ran on MPS; exported files ran through ONNX Runtime `CPUExecutionProvider`. `model.predict` end-to-end includes preprocessing, forward, postprocessing, and predictor overhead; image file decoding was excluded. Preprocess, inference, and postprocess phases were timed separately and synchronized on MPS. Process RSS was sampled during each path.

## Size and compute

| Model path | Checkpoint parameters (all model heads counted once) | FLOPs at 640 px | Checkpoint size | ONNX size |
|---|---:|---:|---:|---:|
| YOLOv8s raw | 11,135,987 | 28.64 G | 22.52 MB | 44.72 MB |
| YOLO26s raw | 9,948,638 | 22.74 G | 20.33 MB | 38.11 MB |
| YOLO26s NMS-free | same YOLO26s checkpoint | 22.74 G reported | same 20.33 MB | 38.12 MB |

YOLO26 raw and NMS-free are two inference paths from one trained checkpoint, so its parameter count and checkpoint size are reported once. The profiler returned the same FLOPs value for the two head paths; treat that estimate as a model-level approximation, not a separately measured hardware instruction count.

## End-to-end latency and throughput

| Path | Median / p95 | Images/s (mean latency) | Preprocess median | Forward median | Postprocess median | Peak process RSS |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8s PyTorch MPS | 20.88 / 22.10 ms | 47.85 | 1.67 ms | 10.94 ms | 1.85 ms | 2.35 GB |
| YOLO26s raw PyTorch MPS | 22.13 / 23.29 ms | 45.22 | 1.51 ms | 12.09 ms | 1.75 ms | 2.35 GB |
| YOLO26s NMS-free PyTorch MPS | 21.03 / 22.07 ms | 47.43 | 1.56 ms | 12.33 ms | 0.84 ms | 2.51 GB |
| YOLOv8s raw ONNX CPU | 56.95 / 65.07 ms | 17.44 | 1.20 ms | 46.38 ms | 0.34 ms | 2.35 GB |
| YOLO26s raw ONNX CPU | 43.83 / 49.43 ms | 22.10 | 1.22 ms | 34.71 ms | 0.31 ms | 2.46 GB |
| YOLO26s NMS-free ONNX CPU | 42.74 / 48.52 ms | 22.94 | 1.23 ms | 33.94 ms | 0.12 ms | 2.75 GB |

On this machine and runtime, YOLO26 ONNX was about 23–25% faster by median than YOLOv8 ONNX, while both YOLO26 PyTorch MPS paths were about as fast as YOLOv8. NMS-free reduced postprocessing but did not materially change model-forward latency. The CPU ONNX median rates are still only about 17–23 images/s. These short, single-machine measurements are not service SLOs; first-call values are recorded separately in the local benchmark JSON.

Efficiency per million checkpoint parameters (using the same validation values):

| Metric / million parameters | YOLOv8s | YOLO26s raw | YOLO26s NMS-free |
|---|---:|---:|---:|
| mAP50–95 | 0.0451 | **0.0537** | 0.0525 |
| mAP50 | **0.0709** | 0.0774 | 0.0783 |
| Recall @ .25 | 0.0768 | **0.0940** | 0.0926 |

These are descriptive efficiency ratios; YOLO26’s higher recall ratio comes with substantially more false positives at the selected threshold.
