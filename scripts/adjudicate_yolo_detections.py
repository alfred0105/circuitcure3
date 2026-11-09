#!/usr/bin/env python3
"""Label YOLO detections against an image- and provenance-matched GT manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuitcare.vision.detection_adjudication import adjudicate_predictions


def load_predictions(path: Path, manifest: dict[str, Any], image_id: str | None) -> dict[str, list[dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value.get("predictions"), dict):
        return {str(key): rows for key, rows in value["predictions"].items()}
    rows = value.get("detections")
    if not isinstance(rows, list):
        raise ValueError("Prediction JSON needs a detections list or predictions object")
    target = image_id or value.get("image_id")
    if not target and value.get("image_sha256"):
        matches = [str(row["image_id"]) for row in manifest.get("images", [])
                   if str(row.get("image_sha256", "")).lower() == str(value["image_sha256"]).lower()]
        if len(matches) == 1:
            target = matches[0]
    if not target:
        raise ValueError("Flat prediction JSON needs --image-id, image_id, or a hash matching one manifest image")
    return {str(target): rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True,
                        help="circuitcare.component-annotation.v1 manifest with reviewed boxes")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="model output JSON from the detector UI or evaluate_pcb_detector.py")
    parser.add_argument("--image-id", help="required for flat single-image detections if no unique image hash is present")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    manifest_bytes = args.annotations.read_bytes()
    manifest = json.loads(manifest_bytes)
    predictions = load_predictions(args.predictions, manifest, args.image_id)
    result = adjudicate_predictions(predictions, manifest, iou_threshold=args.iou)
    result["provenance"] = {
        "annotation_manifest_path": str(args.annotations.resolve()),
        "annotation_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "predictions_path": str(args.predictions.resolve()),
        "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "prediction_image_count": len(predictions),
        "tool": "scripts/adjudicate_yolo_detections.py",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": result["status"], **result["summary"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
