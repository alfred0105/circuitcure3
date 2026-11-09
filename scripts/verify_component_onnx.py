#!/usr/bin/env python3
"""Compare a PyTorch checkpoint and raw ONNX outputs on one identical letterboxed image."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, help="defaults to ONNX static input size")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()
    for path in (args.weights, args.onnx, args.image):
        if not path.is_file(): parser.error(f"file not found: {path}")
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from circuitcare.vision.detection_postprocess import letterbox_rgb

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    shape = input_info.shape
    size = args.imgsz or (shape[-1] if isinstance(shape[-1], int) else 640)
    if isinstance(shape[-1], int) and size != shape[-1]:
        parser.error(f"imgsz {size} does not match static ONNX size {shape[-1]}")
    tensor, _, _ = letterbox_rgb(Image.open(args.image), size)
    onnx_raw = session.run(None, {input_info.name: tensor})[0]
    network = YOLO(str(args.weights)).model.float().cpu().eval()
    with torch.inference_mode():
        result = network(torch.from_numpy(tensor))
    torch_raw = result[0] if isinstance(result, (tuple, list)) else result
    if isinstance(torch_raw, (tuple, list)): torch_raw = torch_raw[0]
    torch_raw = torch_raw.detach().cpu().numpy()
    if torch_raw.shape != onnx_raw.shape:
        raise RuntimeError(f"output shape mismatch: torch={torch_raw.shape}, onnx={onnx_raw.shape}")
    difference = np.abs(torch_raw.astype(np.float32) - onnx_raw.astype(np.float32))
    passed = bool(np.allclose(torch_raw, onnx_raw, atol=args.atol, rtol=args.rtol))
    report = {"weights": str(args.weights.resolve()), "onnx": str(args.onnx.resolve()),
              "image": str(args.image.resolve()), "input_size": size, "provider": "CPUExecutionProvider",
              "output_shape": list(onnx_raw.shape), "max_abs_error": float(difference.max()),
              "mean_abs_error": float(difference.mean()), "atol": args.atol, "rtol": args.rtol,
              "passed": passed}
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
