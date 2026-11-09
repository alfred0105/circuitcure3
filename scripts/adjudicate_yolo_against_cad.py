#!/usr/bin/env python3
"""Use an automatic photo/CAD fit to label YOLO component locations."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuitcare.vision.cad_adjudication import adjudicate_pair_manifest


def _load_predictions(path: Path | None, pair: dict[str, Any], image_id: str | None) -> dict[str, list[dict[str, Any]]] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value.get("predictions"), dict):
        aliases = {}
        for row in pair.get("images", []):
            for token in (row.get("image_id"), row.get("photo_path"),
                          Path(str(row.get("photo_path", ""))).name,
                          Path(str(row.get("photo_path", ""))).stem, row.get("photo_sha256")):
                if token:
                    aliases[str(token)] = str(row.get("image_id"))
        mapped = {}
        for key, rows in value["predictions"].items():
            canonical = aliases.get(str(key), aliases.get(Path(str(key)).name, str(key)))
            mapped[canonical] = list(rows)
        known_ids = {str(row.get("image_id")) for row in pair.get("images", [])}
        unknown = sorted(set(mapped) - known_ids)
        if unknown:
            raise ValueError(f"Prediction images do not match pair-manifest IDs/paths: {unknown[:5]}")
        return mapped
    detections = value.get("detections")
    if not isinstance(detections, list):
        raise ValueError("Prediction JSON must contain detections or a predictions object")
    target = image_id or value.get("image_id")
    if not target and value.get("image_sha256"):
        matched = [str(row["image_id"]) for row in pair.get("images", [])
                   if str(row.get("photo_sha256", "")).lower() == str(value["image_sha256"]).lower()]
        if len(matched) == 1:
            target = matched[0]
    if not target:
        raise ValueError("Single-image predictions need --image-id or a hash matching one pair-manifest photo")
    return {str(target): detections}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, required=True,
                        help="curated photo/CAD pair manifest")
    parser.add_argument("--predictions", type=Path,
                        help="YOLO JSON from detector UI or evaluate_pcb_detector.py; omit to inspect registration only")
    parser.add_argument("--image-id", help="identify a single-image prediction bundle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--location-margin-px", type=float, default=4.0,
                        help="allowed center distance outside the approximate CAD footprint polygon")
    args = parser.parse_args()
    if args.location_margin_px < 0:
        parser.error("--location-margin-px must be non-negative")
    pair_bytes = args.pair_manifest.read_bytes()
    pair = json.loads(pair_bytes)
    predictions = _load_predictions(args.predictions, pair, args.image_id)
    result = adjudicate_pair_manifest(args.pair_manifest, predictions_by_image=predictions,
                                      location_margin_px=args.location_margin_px)
    result["provenance"] = {"pair_manifest_sha256": hashlib.sha256(pair_bytes).hexdigest(),
                            "predictions_path": str(args.predictions.resolve()) if args.predictions else None,
                            "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest() if args.predictions else None,
                            "tool": "scripts/adjudicate_yolo_against_cad.py"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = {}
    for image in result.get("images", []):
        for detection in image.get("predictions", []):
            label = detection.get("adjudication_label", "UNLABELED")
            counts[label] = counts.get(label, 0) + 1
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "images": len(result.get("images", [])), "prediction_label_counts": counts}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
