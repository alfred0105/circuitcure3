#!/usr/bin/env python3
"""Project CAD component footprints into photos and export candidate YOLO boxes.

These labels remain explicitly non-final until visible-component completeness,
assembly state, board revision, and image registration are independently known.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuitcare.vision.cad_adjudication import adjudicate_pair_manifest  # noqa: E402


CLASS_NAMES = [
    "resistor", "capacitor", "diode", "led", "integrated_circuit",
    "connector", "switch_button", "header", "crystal", "debug_test_point",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    result = adjudicate_pair_manifest(args.pair_manifest)
    output = args.output_dir.resolve()
    images_dir, labels_dir = output / "images", output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    name_counts: dict[str, int] = {}
    records = []

    for image in result.get("images", []):
        photo = Path(image.get("photo_path", ""))
        registration = image.get("registration", {})
        width, height = registration.get("photo_size_px", [0, 0])
        image_id = image.get("image_id", photo.stem or "unknown")
        base = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(image_id))
        name_counts[base] = name_counts.get(base, 0) + 1
        if name_counts[base] > 1:
            base = f"{base}-{name_counts[base]}"
        label_rows = []
        invalid_boxes = []
        for component in image.get("cad_components", []):
            class_name = component.get("class_name")
            box = component.get("bbox_xyxy")
            if class_name not in CLASS_NAMES or not box or width <= 0 or height <= 0:
                continue
            x1, y1, x2, y2 = map(float, box)
            x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
            y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                invalid_boxes.append(component.get("component_id"))
                continue
            cls_id = CLASS_NAMES.index(class_name)
            xc, yc = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
            bw, bh = (x2 - x1) / width, (y2 - y1) / height
            label_rows.append(f"{cls_id} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")

        copied_photo = None
        if photo.is_file() and registration.get("status") == "AUTO_REGISTERED_CANDIDATE":
            suffix = photo.suffix.lower() if photo.suffix else ".jpg"
            copied_photo = images_dir / f"{base}{suffix}"
            shutil.copy2(photo, copied_photo)
            (labels_dir / f"{base}.txt").write_text("\n".join(label_rows) + ("\n" if label_rows else ""))

        unmatched = image.get("unclassified_cad_components", [])
        records.append({
            "image_id": image_id,
            "image_path": str(copied_photo) if copied_photo else None,
            "source_image_path": str(photo),
            "image_sha256": image.get("photo_sha256"),
            "board_id": image.get("board_id"),
            "split_group": image.get("board_id") or image_id,
            "registration_status": registration.get("status"),
            "selected_side": registration.get("selected_side"),
            "registration_metrics": registration.get("candidates", {}).get(registration.get("selected_side"), {}),
            "cad_component_count": image.get("cad_component_count", 0),
            "supported_box_count": len(label_rows),
            "unclassified_cad_components": unmatched,
            "invalid_box_component_ids": invalid_boxes,
            "label_path": str(labels_dir / f"{base}.txt") if copied_photo else None,
            "label_status": "CAD_DERIVED_CANDIDATE_PARTIAL" if label_rows else "NO_LABELS_REGISTRATION_UNRESOLVED",
            "image_level_completeness": "UNVERIFIED",
            "training_eligible": False,
            "reason_not_training_eligible": (
                "CAD footprint locations/sizes and class mapping are candidates; CAD BOM does not establish per-photo visibility, population, or complete image-level coverage."
            ),
        })

    manifest = {
        "schema_version": "circuitcare.cad-yolo-candidate-labels.v1",
        "status": "CANDIDATE_LABELS_GENERATED_NOT_TRAINING_READY",
        "source_pair_manifest": str(args.pair_manifest.resolve()),
        "class_names": CLASS_NAMES,
        "box_definition": "Projected approximate CAD footprint body bounds; YOLO normalized xywh.",
        "label_semantics": "CAD-derived pseudo-label candidates, not verified visual ground truth.",
        "images": records,
        "summary": {
            "image_count": len(records),
            "registered_image_count": sum(bool(r["image_path"]) for r in records),
            "candidate_box_count": sum(r["supported_box_count"] for r in records),
            "training_eligible_image_count": 0,
        },
        "limitations": [
            "Geometric registration can be plausible while still aligning the wrong side, revision, or partial board area.",
            "Pick-and-place/BOM entries do not prove a part is visible or populated in a specific photograph.",
            "Approximate footprint body dimensions are not guaranteed to match the physical object box.",
            "Unclassified CAD entries are listed and are not silently assigned a class.",
            "No image is marked complete or training-eligible without verified visible-component coverage.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (output / "classes.json").write_text(json.dumps({str(i): n for i, n in enumerate(CLASS_NAMES)}, indent=2) + "\n")
    print(json.dumps(manifest["summary"] | {"status": manifest["status"], "output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
