#!/usr/bin/env python3
"""Validate one PCB-DSLR archive part and export its IC boxes to YOLO.

The output is a non-commercial-research-only, board-grouped single-class
candidate. It never changes the deployed ten-class model contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/component_training/raw/external/pcb-dslr/part1-extracted"
OUTPUT = ROOT / "data/component_training/processed/pcb-dslr-part1-ic-v1"
CLASS_NAMES = ["integrated_circuit"]


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


def split_for_board(board_id: str, board_dirs: list[str]) -> str:
    """Stable, board-level 70/15/15 split independent of source file order."""
    index = board_dirs.index(board_id)
    if index < 14:
        return "train"
    if index < 17:
        return "validation"
    return "test"


def convert(source: Path = SOURCE, output: Path = OUTPUT) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    split_images, split_boxes = Counter(), Counter()
    classes, group_splits = Counter(), defaultdict(set)
    image_dirs = sorted((p for p in source.glob("pcb*") if p.is_dir()),
                        key=lambda p: int(p.name[3:]))
    if not image_dirs:
        raise ValueError(f"no pcb<ID> directories under {source}")
    board_order = sorted((p.name for p in image_dirs),
                         key=lambda name: hashlib.sha256(f"pcb-dslr-part1:{name}".encode()).hexdigest())
    manifest_split = {"train": "training", "validation": "validation", "test": "sealed_test"}
    for board_dir in image_dirs:
        board_id = board_dir.name
        split = split_for_board(board_id, board_order)
        group_splits[board_id].add(split)
        image_out = output / "images" / split
        label_out = output / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(p for p in board_dir.glob("rec*.jpg") if "-mask" not in p.name):
            annotation = image_path.with_name(image_path.stem + "-annot.txt")
            mask = image_path.with_name(image_path.stem + "-mask.png")
            if not annotation.is_file() or not mask.is_file():
                errors.append(f"missing annotation or board mask: {image_path}")
                continue
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
            with Image.open(mask) as mask_image:
                if mask_image.size != (width, height):
                    errors.append(f"image/mask dimensions differ: {image_path}")
                    continue
            yolo_lines, boxes, seen = [], [], set()
            for line_number, line in enumerate(annotation.read_text(errors="replace").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) < 5:
                    errors.append(f"short annotation row: {annotation}:{line_number}")
                    continue
                try:
                    cx, cy, box_w, box_h, angle = map(float, fields[:5])
                except ValueError:
                    errors.append(f"non-numeric annotation: {annotation}:{line_number}")
                    continue
                if not all(math.isfinite(v) for v in (cx, cy, box_w, box_h, angle)) or box_w <= 0 or box_h <= 0:
                    errors.append(f"invalid rotated rectangle: {annotation}:{line_number}")
                    continue
                corners = cv2.boxPoints(((cx, cy), (box_w, box_h), angle))
                x1, y1 = corners.min(axis=0)
                x2, y2 = corners.max(axis=0)
                if x1 < -1 or y1 < -1 or x2 > width + 1 or y2 > height + 1:
                    errors.append(f"out-of-bounds rotated box: {annotation}:{line_number}")
                    continue
                x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
                x2, y2 = min(float(width), float(x2)), min(float(height), float(y2))
                key = tuple(round(v, 3) for v in (cx, cy, box_w, box_h, angle))
                if key in seen:
                    errors.append(f"duplicate rotated box: {annotation}:{line_number}")
                    continue
                seen.add(key)
                x_center, y_center = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
                normalized_w, normalized_h = (x2 - x1) / width, (y2 - y1) / height
                yolo_lines.append(f"0 {x_center:.8f} {y_center:.8f} {normalized_w:.8f} {normalized_h:.8f}")
                text_label = " ".join(fields[5:]) if len(fields) > 5 else ""
                boxes.append({"component_id": f"{board_id}:{image_path.stem}:annotation:{line_number:04d}",
                    "class": "integrated_circuit", "source_text_label": text_label or None,
                    "bbox_xyxy": [x1, y1, x2, y2], "rotation_degrees": angle,
                    "source_line": line_number, "occlusion_state": "UNKNOWN", "small_component": None,
                    "reviewer": None, "annotation_status": "DRAFT_ANNOTATION",
                    "source": display_path(annotation),
                    "provenance": {"origin": "publisher_provided_rotated_rectangle", "line_number": line_number}})
                classes["integrated_circuit"] += 1
            if errors:
                continue
            export_stem = f"{board_id}_{image_path.stem}"
            target_image = image_out / f"{export_stem}{image_path.suffix}"
            target_label = label_out / f"{export_stem}.txt"
            target_image.symlink_to(image_path.resolve())
            target_label.write_text("\n".join(yolo_lines) + "\n")
            split_images[split] += 1
            split_boxes[split] += len(boxes)
            rows.append({"image_id": f"{board_id}:{image_path.stem}",
                "image_path": display_path(image_path),
                "annotation_path": display_path(annotation),
                "board_mask_path": display_path(mask),
                "exported_image_path": display_path(target_image),
                "exported_label_path": display_path(target_label),
                "image_sha256": sha256(image_path), "annotation_sha256": sha256(annotation),
                "mask_sha256": sha256(mask), "image_size_px": [width, height],
                "label_path": display_path(target_label), "label_sha256": sha256(target_label),
                "label_state": "LABELED", "board_id": board_id,
                "board_revision": "UNKNOWN", "board_side": "UNKNOWN",
                "board_group_id": board_id, "board_group_review_status": "SOURCE_PROVIDED_BOARD_ID",
                "split": manifest_split[split], "export_split": split, "image_type": "POPULATED_PCB",
                "classification_status": "AUTO_CLASSIFIED", "classification_confidence": "HIGH",
                "review_status": "DOMAIN_REVIEW_REQUIRED", "annotation_count": len(boxes),
                "annotation_coverage": "SOURCE_CLAIMS_ALL_IC_CHIPS; OTHER_CLASS_LABELS_NOT_IN_SCOPE",
                "annotations": boxes, "training_include": False})
    if errors:
        raise ValueError("candidate conversion failed:\n" + "\n".join(errors[:80]))
    leakage = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1}
    if leakage:
        raise ValueError(f"board-level split leakage: {leakage}")
    data_yaml = output / "data.yaml"
    data_yaml.write_text(f"path: {output.resolve()}\ntrain: images/train\nval: images/validation\ntest: images/test\nnames:\n  0: integrated_circuit\n")
    manifest = {"schema_version": "circuitcare.component-annotation.v1",
        "dataset_id": "pcb-dslr-part1-ic-research-v1", "dataset_name": "PCB DSLR Dataset, archive part 1, IC-only",
        "dataset_source": "TU Wien Computer Vision Lab PCB DSLR Dataset",
        "dataset_source_url": "https://zenodo.org/records/3886553",
        "source_version": "Zenodo record v1, archive part 1",
        "dataset_license": "Free for non-commercial research use only",
        "license_scope": "Local research training only; no commercial use or source redistribution",
        "source_annotation_format": "cx cy width height angle [optional IC text]",
        "annotation_source": "Publisher-supplied IC rotated rectangles; optional trailing chip/OCR text is metadata, not a component class",
        "annotation_review_status": "SOURCE_PROVIDED_UNADJUDICATED",
        "board_grouping_policy": "Publisher source pcb<ID> directory is retained as board_id",
        "split_policy": "All recordings from each source board ID stay in one split",
        "split_name": "board-level train/validation/sealed test",
        "split_method": "DETERMINISTIC_SOURCE_BOARD_ID_HASH_SPLIT_70_15_15",
        "board_grouping_policy": "Publisher source pcb<ID> directory is retained as board_id",
        "split_policy": "All recordings from each source board ID stay in one split",
        "class_names": CLASS_NAMES,
        "board_split_leakage": "PASS", "board_id_status": "SOURCE_PROVIDED_PHYSICAL_PCB_IDS",
        "training_ready": False,
        "status": "DOWNLOADED_NOT_TRAINING_READY",
        "blockers": ["IMAGE_DOMAIN_REVIEW_PENDING"],
        "split_images": dict(split_images), "split_boxes": dict(split_boxes),
        "class_counts": dict(classes), "images": rows}
    (output / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return {"image_count": len(rows), "board_count": len(group_splits),
        "split_images": dict(split_images), "split_boards": {
            split: len({r["board_group_id"] for r in rows if r["export_split"] == split})
            for split in ("train", "validation", "test")},
        "split_boxes": dict(split_boxes), "annotations": sum(split_boxes.values()),
        "board_split_leakage": "PASS", "status": manifest["status"],
        "output": str(output), "training_ready": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
