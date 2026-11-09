# PCB-DSLR 연구 모델 ONNX 검증

- PyTorch checkpoint: local `data/component_training/runs/20261106-183126-06352e/weights/best.pt`
- ONNX: local `data/component_training/runs/20261106-183126-06352e/onnx/pcb-dslr-ic-yolov8s.onnx`
- Ultralytics 8.4.203; opset 17; input 1×3×640×640; output 1×5×8400 (4 box channels + 1 class score)
- ONNX Runtime 1.31.1 CPU provider; session and `onnx.checker` succeeded
- Weights SHA-256: `e56aa45cf3848c2a889a67ba6e62065a3c4d7c7baa0bf622dbe375e13e70d0e8`
- ONNX SHA-256: `d754a1f9d23b9d987d9822c409542c040b60bd1200539e17c706a4037c0eff71`

On validation image `pcb14/rec2.jpg`, PyTorch and ONNX raw outputs had identical shape and passed `allclose(atol=1e-4, rtol=1e-3)`. Maximum absolute difference was 0.00223 and mean absolute difference 0.0000346; relative tolerance accounts for larger-valued entries. Ultralytics' experimental ONNX adapter decoded detections as `integrated_circuit` in source image coordinates and generated a local overlay. On the same validation image at confidence .25 / NMS IoU .45, PyTorch MPS and ONNX CPU both returned 21 detections; all 21 paired at minimum box IoU 0.988, maximum confidence delta 0.0149 and maximum coordinate delta 0.469px. No NMS count mismatch occurred. Cold first-call times varied across fresh invocations (PyTorch MPS ~800ms; ONNX CPU 121–808ms), so these are not steady-state latency claims.

Run `scripts/verify_component_onnx.py` for raw output comparison, `scripts/compare_component_experiment_inference.py --weights PATH.pt --onnx PATH.onnx --image PCB.jpg --output parity.json` for post-NMS parity, and `scripts/run_component_experiment_inference.py --model PATH --image PCB.jpg --output-dir OUT` for a visual overlay. The existing deployed 10-class decoder was not modified or reused; the experimental adapter uses the checkpoint's own class metadata. Existing production ONNX was not replaced.
