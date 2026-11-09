# Phase 21.3.2 — Mac MPS training notes

Machine observed on 2026-11-06: Apple M5, ARM64, macOS 27.0.1, 24 GiB unified memory, Python 3.12.14 ARM64. The repository's normal Python is 3.9.6 and is not used. A dedicated `.venv-component-train` was created without modifying the application environment.

Installed versions: PyTorch 2.15.0, torchvision 0.29.2, Ultralytics 8.4.203, OpenCV 4.14.0.94, NumPy 2.5.4, ONNX 1.23.2, ONNX Runtime 1.31.1. These wheels are ARM64 native. Doctor result: MPS is built and available; random FP32 matrix forward, backward, SGD, and synchronize succeeded; selected device `mps`. No CUDA APIs are used by the training script. NVIDIA monitoring transitive packages are not used by this workflow.

At probe time 6.83 GiB was available. Keep at least 4 GiB available before a run, begin at batch 2 / 640 / workers 0, and watch system memory and swap. 960 px is batch 1. The script records host memory before/after and records OOM retries; it cannot measure fanless thermal state, so user-visible activity/throughput monitoring is required. AMP is disabled. MPS failure is surfaced, then CPU is selected only when the MPS probe fails. A CPU run is not labeled an MPS run.

The 1-epoch synthetic smoke completed forward/backward/optimizer, checkpoint save/reload, and ONNX export. PyTorch warned that MPS `scatter_reduce` and `index_put_with_accumulate` lack deterministic implementations, despite deterministic mode being requested; repeat runs may differ. This warning is recorded as a reproducibility limit.

A pre-filter real-data 5-epoch training and a contaminated 40-epoch experiment are historical only; they included out-of-domain content and cannot support the target-domain comparison.

The domain-reviewed run `20261106-133618-daa9b3` completed 40 epochs on MPS, `imgsz=640`, `batch=2`, `workers=0`, FP32, seed 21. It used 77 filtered training images and 24 development-validation images. Best validation epoch was 29. Available host memory changed from 6.18 GiB to 5.37 GiB; peak per-step MPS allocation was about 1.24 GiB. The same nondeterministic MPS operator warnings were emitted. See the benchmark report: this model underperformed the existing detector and is not a candidate for replacing it. Source labels and board identity remain unadjudicated/unverified.
