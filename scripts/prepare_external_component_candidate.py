#!/usr/bin/env python3
"""Validate and convert the DatasetNinja Supervisely PCB candidate to YOLO.

This creates a local loader-smoke candidate only. It deliberately does not
promote source images to TRAINING_READY or infer physical board identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/component_training/raw/external/pcb-component-detection/extracted"
OUTPUT = ROOT / "data/component_training/processed/external-pcb-candidate-loader-smoke-v2"
SOURCE_SPLITS = ("train", "validation", "test")
EXPORT_SPLITS = {"train": "train", "validation": "validation"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def convert(source: Path = SOURCE, output: Path = OUTPUT) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    metadata = json.loads((source / "meta.json").read_text())
    names = [row["title"] for row in metadata["classes"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("source class names must be non-empty and unique")
    output.mkdir(parents=True, exist_ok=True)
    manifest_rows, errors = [], []
    class_counts, split_counts = Counter(), Counter()
    empty_count = 0
    for source_split in SOURCE_SPLITS:
        target_split = EXPORT_SPLITS.get(source_split)
        image_dir = source / source_split / "img"
        ann_dir = source / source_split / "ann"
        if target_split is not None:
            images_out = output / "images" / target_split
            labels_out = output / "labels" / target_split
            images_out.mkdir(parents=True, exist_ok=True)
            labels_out.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(p for p in image_dir.iterdir() if p.is_file()):
            ann_path = ann_dir / (image_path.name + ".json")
            if not ann_path.is_file():
                errors.append(f"missing annotation: {ann_path}")
                continue
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
            data = json.loads(ann_path.read_text())
            if data.get("size") != {"height": height, "width": width}:
                errors.append(f"image/annotation dimensions differ: {image_path}")
                continue
            objects, seen_boxes, lines = data.get("objects"), set(), []
            if not isinstance(objects, list):
                errors.append(f"objects must be an array: {ann_path}")
                continue
            for index, obj in enumerate(objects):
                class_name = obj.get("classTitle")
                exterior = obj.get("points", {}).get("exterior", [])
                if class_name not in names or len(exterior) != 2:
                    errors.append(f"unsupported class or rectangle: {ann_path} object {index}")
                    continue
                (x1, y1), (x2, y2) = exterior
                if not all(isinstance(v, (int, float)) for v in (x1, y1, x2, y2)):
                    errors.append(f"non-numeric box: {ann_path} object {index}")
                    continue
                x1, x2 = sorted((float(x1), float(x2)))
                y1, y2 = sorted((float(y1), float(y2)))
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    errors.append(f"out-of-bounds or zero-area box: {ann_path} object {index}")
                    continue
                key = (class_name, x1, y1, x2, y2)
                if key in seen_boxes:
                    errors.append(f"duplicate box: {ann_path} object {index}")
                    continue
                seen_boxes.add(key)
                class_id = names.index(class_name)
                cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
                bw, bh = (x2 - x1) / width, (y2 - y1) / height
                lines.append(f"{class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
                class_counts[class_name] += 1
            if len(lines) != len(objects):
                continue
            source_stem = image_path.stem.split(".rf.")[0]
            exported_label = None
            smoke_exported = bool(lines and target_split is not None)
            if not lines:
                empty_count += 1
            if smoke_exported:
                image_link = images_out / image_path.name
                image_link.symlink_to(image_path.resolve())
                label_path = labels_out / f"{image_path.stem}.txt"
                label_path.write_text("\n".join(lines) + "\n")
                exported_label = display_path(label_path)
                split_counts[target_split] += 1
            manifest_rows.append({
                "image_id": f"{source_split}:{image_path.name}",
                "image_path": display_path(image_path),
                "label_source_path": display_path(ann_path),
                "exported_label_path": exported_label,
                "image_sha256": sha256(image_path), "annotation_sha256": sha256(ann_path),
                "source_split": source_split, "source_stem_group": source_stem,
                "physical_board_id": None, "board_review_status": "BOARD_ID_UNRESOLVED",
                "image_type": "PCB_COMPONENT_CLOSEUP", "classification_status": "AUTO_CLASSIFIED",
                "classification_confidence": "MEDIUM", "review_status": "DOMAIN_REVIEW_REQUIRED",
                "annotation_count": len(lines),
                "annotation_status": "EMPTY_UNREVIEWED" if not lines else "SOURCE_ANNOTATED_NOT_ADJUDICATED",
                "included_in_loader_smoke_export": smoke_exported,
                "training_ready": False
            })
    if errors:
        raise ValueError("conversion validation failed:\n" + "\n".join(errors[:50]))
    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        f"path: {output.resolve()}\ntrain: images/train\nval: images/validation\nnames:\n" +
        "".join(f"  {index}: {name}\n" for index, name in enumerate(names)))
    manifest = {"schema_version": "circuitcare.external-candidate.v1",
        "source": "Kaggle PCB Component Detection / DatasetNinja Supervisely conversion",
        "source_license": "CC0-1.0 (source metadata)", "classes": names,
        "status": "DOWNLOADED_NOT_TRAINING_READY",
        "training_ready": False,
        "blockers": ["BOARD_ID_UNRESOLVED", "SOURCE_SPLIT_LEAKAGE", "ANNOTATIONS_NOT_ADJUDICATED"],
        "empty_labels_are_negative_samples": False,
        "split_policy": "source split retained for loader smoke only; not an approved evaluation split",
        "split_counts": dict(split_counts), "empty_annotation_count": empty_count,
        "annotation_count_by_class": dict(class_counts), "images": manifest_rows}
    (output / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"status": manifest["status"], "images": len(manifest_rows),
        "annotations": sum(class_counts.values()), "empty_unreviewed": empty_count,
        "splits": dict(split_counts), "classes": names, "output": str(output),
        "training_ready": False, "blockers": manifest["blockers"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
