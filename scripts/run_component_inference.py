#!/usr/bin/env python3
"""Run one PCB image through a .pt or ONNX model and save overlay + JSON."""
import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, help="defaults to the ONNX model's static input size or 640 for .pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if not args.model.is_file() or not args.image.is_file():
        parser.error("model and image must exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.model.suffix.lower() == ".onnx":
        if args.device == "mps":
            parser.error("ONNX inference uses the verified CPU provider; select .pt for MPS inference")
        import onnxruntime as ort
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from circuitcare.vision.detection_postprocess import PCB_MODEL_CLASSES, decode_yolov8_output, letterbox_rgb
        session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
        shape = session.get_inputs()[0].shape
        model_size = shape[-1] if isinstance(shape[-1], int) else 640
        imgsz = args.imgsz or model_size
        if shape[-1] != "height" and isinstance(shape[-1], int) and imgsz != shape[-1]:
            parser.error(f"imgsz {imgsz} does not match static ONNX input size {shape[-1]}")
        tensor, ratio, padding = letterbox_rgb(Image.open(args.image), imgsz)
        raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        rows = decode_yolov8_output(raw, image_size=Image.open(args.image).size, ratio=ratio,
                                    padding=padding, class_names=PCB_MODEL_CLASSES,
                                    confidence_threshold=args.confidence)
        rows = [{"class_id": PCB_MODEL_CLASSES.index(row["predicted_class"]),
                 "class_name": row["predicted_class"], "confidence": row["confidence"],
                 "bbox_xyxy": row["bbox_xyxy"]} for row in rows]
        device = "onnxruntime-cpu"
        inference_result = rows
    else:
        import torch
        from ultralytics import YOLO
        device = args.device
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            parser.error("MPS was selected but is not available")
        model = YOLO(str(args.model))
        result = model.predict(source=str(args.image), imgsz=args.imgsz or 640, conf=args.confidence,
                               device=device, verbose=False)[0]
        names = result.names
        inference_result = []
        if result.boxes is not None:
            for box in result.boxes:
                xyxy = [float(v) for v in box.xyxy[0].cpu().tolist()]
                class_id = int(box.cls[0].cpu())
                inference_result.append({"class_id": class_id, "class_name": str(names[class_id]),
                                         "confidence": float(box.conf[0].cpu()), "bbox_xyxy": xyxy})
    elapsed = (time.perf_counter() - started) * 1000
    rows = inference_result
    image = Image.open(args.image).convert("RGB")
    draw = ImageDraw.Draw(image)
    for row in rows:
        x1, y1, x2, y2 = row["bbox_xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 100), width=max(2, image.width // 800))
        draw.text((x1, max(0, y1 - 18)), f"{row['class_name']} {row['confidence']:.2f}", fill=(0, 255, 100))
    stem = args.image.stem
    image_path = args.output_dir / f"{stem}_detections.jpg"
    json_path = args.output_dir / f"{stem}_detections.json"
    image.save(image_path, quality=92)
    report = {"model": str(args.model.resolve()), "image": str(args.image.resolve()), "device": device,
              "inference_latency_ms": elapsed, "image_size_px": list(Image.open(args.image).size),
              "detections": rows, "overlay": str(image_path.resolve())}
    json_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
