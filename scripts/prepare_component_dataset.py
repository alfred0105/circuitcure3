#!/usr/bin/env python3
"""Apply human-reviewed PCB-domain and board-group filtering without touching raw data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAIN_TYPES = {"POPULATED_PCB", "PARTIALLY_POPULATED_PCB", "PCB_COMPONENT_CLOSEUP",
                "BARE_PCB", "BREADBOARD", "ISOLATED_COMPONENT", "LOOSE_COMPONENT_COLLECTION", "UNKNOWN"}
TRAIN_TYPES = {"POPULATED_PCB", "PARTIALLY_POPULATED_PCB", "PCB_COMPONENT_CLOSEUP"}
HARD_NEGATIVE_TYPES = {"BARE_PCB", "BREADBOARD"}
CLASSES = ["resistor", "capacitor", "diode", "led", "integrated_circuit", "connector",
           "switch_button", "header", "crystal", "debug_test_point"]


def validate_manifest(manifest: dict) -> dict:
    rows = manifest.get("images")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest images must be a non-empty list")
    ids, grouped_splits = set(), defaultdict(set)
    included, errors, hashes_by_split = [], [], defaultdict(dict)
    type_counts, before_groups = Counter(), set()
    for row in rows:
        image_id = row.get("image_id")
        if not image_id or image_id in ids:
            errors.append(f"missing or duplicate image_id: {image_id}")
            continue
        ids.add(image_id)
        domain = row.get("image_type")
        status = row.get("review_status")
        confidence = row.get("classification_confidence")
        if domain not in DOMAIN_TYPES:
            errors.append(f"{image_id}: invalid image_type {domain!r}")
        type_counts[domain] += 1
        group = row.get("board_group_id")
        if group:
            before_groups.add(group)
        if status not in {"HUMAN_REVIEWED_CONTACT_SHEET", "REVIEW_INHERITED_FROM_PARENT_AUGMENTATION"}:
            errors.append(f"{image_id}: image has not been reviewed")
        if confidence == "LOW" and domain != "UNKNOWN":
            errors.append(f"{image_id}: low-confidence classification must be UNKNOWN")
        should_include = domain in TRAIN_TYPES
        if bool(row.get("main_training_included")) != should_include:
            errors.append(f"{image_id}: main_training_included conflicts with domain policy")
        if should_include:
            split = row.get("assigned_split")
            if split not in {"train", "validation"}:
                errors.append(f"{image_id}: target-domain image has invalid assigned_split")
            if not group:
                errors.append(f"{image_id}: target-domain image has unresolved board group")
            else:
                grouped_splits[group].add(split)
            included.append(row)
            sha = row.get("image_sha256")
            if sha:
                hashes_by_split[split][sha] = image_id
        elif row.get("assigned_split") is not None:
            errors.append(f"{image_id}: excluded image has an assigned training split")
        # Empty labels alone do not prove the scene has no target components.
        eligible = (domain in HARD_NEGATIVE_TYPES and confidence == "HIGH"
                    and row.get("target_absence_confirmed") is True
                    and row.get("annotation_count") == 0)
        if bool(row.get("hard_negative_eligible")) != eligible:
            errors.append(f"{image_id}: hard-negative eligibility conflicts with review/annotation state")
        image = ROOT / "data" / "component_training" / row["image_path"]
        label = ROOT / "data" / "component_training" / row["label_path"]
        if not image.is_file() or not label.is_file():
            errors.append(f"{image_id}: source image or label is missing")
        elif row.get("image_sha256") and hashlib.sha256(image.read_bytes()).hexdigest() != row["image_sha256"]:
            errors.append(f"{image_id}: source image hash changed")
        elif row.get("label_sha256") and hashlib.sha256(label.read_bytes()).hexdigest() != row["label_sha256"]:
            errors.append(f"{image_id}: source label hash changed")
    leaked = {group: sorted(splits) for group, splits in grouped_splits.items() if len(splits) > 1}
    if leaked:
        errors.append(f"board-group split leakage: {leaked}")
    duplicate_hashes = set(hashes_by_split["train"]) & set(hashes_by_split["validation"])
    if duplicate_hashes:
        errors.append(f"exact image hash appears across train/validation: {sorted(duplicate_hashes)}")
    if errors:
        raise ValueError("\n".join(errors))
    return {"source_image_count": len(rows), "source_manual_board_group_count": len(before_groups),
            "included_image_count": len(included),
            "included_images_by_split": dict(Counter(row["assigned_split"] for row in included)),
            "included_manual_board_group_count": len(grouped_splits),
            "included_groups_by_split": {split: len([g for g, splits in grouped_splits.items() if split in splits])
                                          for split in ("train", "validation")},
            "excluded_images_by_type": {name: count for name, count in type_counts.items() if name not in TRAIN_TYPES},
            "included_images_by_type": {name: count for name, count in type_counts.items() if name in TRAIN_TYPES},
            "hard_negative_image_count": sum(bool(row.get("hard_negative_eligible")) for row in rows),
            "board_group_leakage": "PASS_FOR_REVIEWED_GROUP_IDS",
            "physical_board_count_status": "MANUAL_GROUP_COUNT_ONLY; source contains no verified physical Board IDs"}


def prepare(manifest_path: Path, output_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    summary = validate_manifest(manifest)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target_rows = [row for row in manifest["images"] if row["main_training_included"]]
    for row in target_rows:
        split = row["assigned_split"]
        image_src = ROOT / "data" / "component_training" / row["image_path"]
        label_src = ROOT / "data" / "component_training" / row["label_path"]
        for kind, source in (("images", image_src), ("labels", label_src)):
            directory = output_dir / kind / split
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / source.name
            if destination.exists():
                raise FileExistsError(f"duplicate filename within filtered split: {destination}")
            destination.symlink_to(source.resolve())
    (output_dir / "data.yaml").write_text(
        f"path: {output_dir.resolve()}\ntrain: images/train\nval: images/validation\nnames:\n" +
        "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES)))
    negative_rows = [row for row in manifest["images"] if row["hard_negative_eligible"]]
    (output_dir / "hard_negatives.json").write_text(json.dumps({
        "policy": "kept separate; not used as ordinary training backgrounds",
        "images": negative_rows}, indent=2))
    (output_dir / "filtered_manifest.json").write_text(json.dumps({
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "summary": summary, "included_images": target_rows}, indent=2))
    source_names = {index: name for index, name in enumerate(CLASSES)}
    eval_rows = []
    for row in target_rows:
        if row["assigned_split"] != "validation":
            continue
        image_src = ROOT / "data" / "component_training" / row["image_path"]
        label_src = ROOT / "data" / "component_training" / row["label_path"]
        annotations = []
        for line_number, line in enumerate(label_src.read_text().splitlines(), 1):
            if not line.strip():
                continue
            class_id, cx, cy, box_w, box_h = int(line.split()[0]), *map(float, line.split()[1:])
            if class_id not in source_names or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < box_w <= 1 and 0 < box_h <= 1):
                raise ValueError(f"invalid YOLO annotation in {label_src}:{line_number}")
            width, height = row["image_size_px"]
            annotations.append({"component_id": f"{row['image_id']}:annotation:{line_number:04d}",
                "class": source_names[class_id],
                "bbox_xyxy": [(cx-box_w/2)*width, (cy-box_h/2)*height,
                              (cx+box_w/2)*width, (cy+box_h/2)*height],
                "occlusion_state": "UNKNOWN", "small_component": None, "reviewer": None,
                "annotation_status": "DRAFT_ANNOTATION",
                "source": "source-provided YOLO label; not independently adjudicated"})
        eval_rows.append({"image_id": row["image_id"],
            "image_path": os.path.relpath(image_src, output_dir),
            "label_path": os.path.relpath(label_src, output_dir),
            "board_id": row["board_group_id"], "board_revision": "UNKNOWN", "board_side": "UNKNOWN",
            "split": "validation", "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"],
            "image_size_px": row["image_size_px"], "label_state": "LABELED",
            "source": f"pinned source:{row['image_id']}", "image_type": row["image_type"],
            "annotations": annotations})
    eval_manifest = {"schema_version": "circuitcare.component-annotation.v1",
        "dataset_id": "pcb-domain-reviewed-dev-validation-v1",
        "dataset_name": "Human-reviewed PCB-domain development validation",
        "dataset_source": "wangshnegqi/pcb-component-detection-dataset-yolo",
        "dataset_source_url": "https://github.com/wangshnegqi/pcb-component-detection-dataset-yolo",
        "source_version": manifest.get("source_commit"), "dataset_license": "CC BY 4.0",
        "annotation_source": "Source-provided YOLO labels; not independently adjudicated",
        "annotation_review_status": "SOURCE_PROVIDED_UNADJUDICATED",
        "split_name": "validation", "split_method": "MANUAL_VISUAL_BOARD_GROUP_SPLIT",
        "board_grouping_policy": "Manual visual capture groups; source has no verified physical board IDs",
        "split_policy": "Grouped image captures by reviewed board_group_id; development-only, not an independent sealed test",
        "class_names": CLASSES, "images": eval_rows}
    (output_dir / "evaluation_manifest.json").write_text(json.dumps(eval_manifest, indent=2))
    summary["validation_image_count"] = len(eval_rows)
    summary["validation_annotation_count"] = sum(len(row["annotations"]) for row in eval_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = prepare(args.manifest, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
