# Phase 21.3.2 — independent component training lab

Status: domain-reviewed dataset filter, MPS probe, synthetic smoke, fresh 40-epoch filtered training, ONNX export and parity check completed. The filtered model underperformed the existing detector on source-provided labels; it is not a replacement candidate. Labels are not independently adjudicated and physical board identity is unresolved.

The lab uses a separate `.venv-component-train` and scripts in `experiments/component_recognition/` and `scripts/component_training.py`. It does not change CircuitCare application dependencies or detector runtime. Raw and generated large artifacts are ignored by Git. Existing model identifier and baseline artifacts remain untouched.

The YOLOv8n starting checkpoint is fetched from Ultralytics' published `yolov8n.pt` asset and is not committed. Ultralytics publishes the framework under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE); review applicable framework and weight terms before distributing training tooling or derivative weights. The dataset and checkpoint licenses are tracked separately.

## Host and device check (2026-11-06 Asia/Seoul)

| Item | Observed |
|---|---|
| Machine | Apple M5 |
| Architecture | ARM64 |
| OS | macOS 27.0.1 |
| Memory | 24 GiB unified memory; 6.83 GiB available during probe |
| Python | 3.12.14 ARM64 |
| PyTorch / torchvision | 2.15.0 / 0.29.2 |
| MPS built / available | yes / yes |
| Tensor forward, backward, optimizer and sync | passed |
| Selected device | MPS |
| Thermal measurement | not available through the probe; monitor Activity Monitor during long runs |

The environment is reproducible from `experiments/component_recognition/requirements-macos-arm64.txt` and the installed-version snapshot in `requirements-macos-arm64.lock.txt`.

## Domain curation and split

`data/component_training/manifests/pinned-source-domain-review.json` classifies all 130 source images with the eight requested types. Only `POPULATED_PCB`, `PARTIALLY_POPULATED_PCB`, and `PCB_COMPONENT_CLOSEUP` enter ordinary training. Filtering retains 101 images (77 train / 24 development validation) across 33 manually reviewed visual capture groups (29 / 4). It excludes 19 `LOOSE_COMPONENT_COLLECTION` photos and 10 `BREADBOARD` images. Those ten reviewed breadboards remain separate for false-positive evaluation. No `BARE_PCB`, `ISOLATED_COMPONENT`, or `UNKNOWN` images were found in this source. A low-confidence image must remain `UNKNOWN`.

An empty label file never qualifies an image as empty. The manifest requires explicit `target_absence_confirmed` human review before a bare PCB or breadboard becomes a hard negative. One-component PCBs remain included, and closeups share the parent visual board group. All reviewed board groups stay within one split; leakage check passes. The source lacks verified physical board IDs, so 33 is a manual visual group count, not a confirmed unique PCB count.

## Implementation

- `scripts/component_training.py doctor`: actual MPS operation probe with CPU fallback reporting.
- `scripts/component_training.py smoke`: one-epoch FP32 smoke path including checkpoint reload and ONNX export.
- `scripts/component_training.py train`: unique experiment directory, metadata, seed, low initial batch, memory guard, OOM record and batch retry.
- `scripts/component_training.py export`: separate ONNX export, ONNX checker, CPU Runtime load, tensor contract metadata and SHA-256.
- `scripts/run_component_inference.py`: single-image `.pt` or ONNX inference, overlay, JSON and latency.
- Profiles start at YOLOv8n/640/batch 2; 960 profile starts at batch 1. MPS is only selected after a successful operator probe; FP16/AMP remains off.

## Verification state

MPS doctor and a synthetic 1-epoch smoke passed. A fresh YOLOv8n run trained 40 epochs on the filtered dataset at 640px, batch 2, workers 0, FP32 on MPS. It underperformed the pinned baseline on the same 24-image board-group development validation and is not a replacement. ONNX export and raw PyTorch/ONNX parity passed (max absolute error 0.00214). Source labels remain unadjudicated, so reported metrics are development source-label comparisons. Refer to `PHASE21_3_2_DATASET_EXPANSION.md`, `PHASE21_3_2_MAC_MPS_TRAINING.md`, `PHASE21_3_2_MODEL_BENCHMARK.md`, and `PHASE21_3_2_ONNX_EXPORT.md` for details.

## Phase 21.3.2-B 확장 상태

후속 수집 및 품질 검증 결과는 [Phase 21.3.2-B acquisition report](PHASE21_3_2_B_DATASET_ACQUISITION.md)에서 확인한다. PCB-DSLR part 1로 비상업 연구 전용 `integrated_circuit` 1-class 데이터셋(87 images / 20 source PCB IDs)을 준비했다. canonical 10-class용 데이터 확장은 아직 승인되지 않았다.
