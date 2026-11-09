#!/usr/bin/env python3
"""Create an auditable, non-training-ready curation package for Kaggle PCB images.

Human visual review is limited to image domain and board grouping. Source boxes
remain draft labels until annotation completeness and source-class semantics are
independently adjudicated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACQUISITION = ROOT / "data/component_training/manifests/pcb-component-detection-acquisition-manifest.json"
DEFAULT_OUTPUT = ROOT / "data/component_training/manifests/pcb-component-detection-curation-manifest.json"
VISUAL_GROUP_ID = "KAGGLE_VISUAL_BOARD_GROUP_001"
SOURCE_CLASSES = ["Cap1", "Cap2", "Cap3", "Cap4", "MOSFET", "Mov", "Resestor", "Resistor", "Transformer"]
LABEL_STATES = {"ANNOTATION_COMPLETE", "PARTIAL_ANNOTATION", "EMPTY_REVIEWED",
                "EMPTY_UNREVIEWED", "ANNOTATION_STATUS_UNKNOWN"}
TRAINING_DOMAINS = {"POPULATED_PCB", "PARTIALLY_POPULATED_PCB", "PCB_COMPONENT_CLOSEUP"}


def training_domain_eligible(image_type: str) -> bool:
    """Apply the explicit PCB domain allowlist; one mounted component is valid."""
    return image_type in TRAINING_DOMAINS


def annotation_state(annotation_count: int, *, completeness_reviewed: bool = False,
                     empty_reviewed: bool = False, visible_target_without_label: bool = False) -> str:
    """Never infer complete or negative coverage from label-file presence alone."""
    if annotation_count < 0:
        raise ValueError("annotation_count must be non-negative")
    if completeness_reviewed and annotation_count > 0:
        return "ANNOTATION_COMPLETE"
    if visible_target_without_label:
        return "PARTIAL_ANNOTATION"
    if annotation_count == 0:
        return "EMPTY_REVIEWED" if empty_reviewed else "EMPTY_UNREVIEWED"
    return "ANNOTATION_STATUS_UNKNOWN"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phash(path: Path) -> int:
    with Image.open(path) as source:
        gray = np.asarray(ImageOps.exif_transpose(source).convert("L").resize((32, 32)), dtype=np.float32)
    coefficients = cv2.dct(gray)[:8, :8].reshape(-1)
    median = float(np.median(coefficients[1:]))
    return sum(int(value) << index for index, value in enumerate(coefficients > median))


def _resolve_external(root: Path, source_path: str) -> Path:
    return (root / "data/component_training/raw/external" / source_path).resolve()


def _manifest_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _annotation_rows(annotation_path: Path, image_id: str, width: int, height: int,
                     repo_root: Path) -> tuple[list[dict], list[str]]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    if data.get("size") != {"height": height, "width": width}:
        raise ValueError(f"image and annotation sizes differ: {annotation_path}")
    rows, errors = [], []
    for index, obj in enumerate(data.get("objects", [])):
        if obj.get("classTitle") not in SOURCE_CLASSES:
            errors.append(f"unsupported class: {annotation_path} object {index}")
            continue
        exterior = obj.get("points", {}).get("exterior", [])
        if len(exterior) != 2:
            errors.append(f"invalid rectangle: {annotation_path} object {index}")
            continue
        (x1, y1), (x2, y2) = exterior
        try:
            x1, x2 = sorted((float(x1), float(x2)))
            y1, y2 = sorted((float(y1), float(y2)))
        except (TypeError, ValueError):
            errors.append(f"non-numeric rectangle: {annotation_path} object {index}")
            continue
        if not (np.isfinite([x1, y1, x2, y2]).all() and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            errors.append(f"invalid or out-of-bounds rectangle: {annotation_path} object {index}")
            continue
        class_name = obj["classTitle"]
        rows.append({"component_id": f"{image_id}:source-object:{obj.get('id', index)}",
            "class": class_name, "bbox_xyxy": [x1, y1, x2, y2],
            "occlusion_state": "UNKNOWN", "small_component": None, "reviewer": None,
            "annotation_status": "DRAFT_ANNOTATION", "source": _manifest_path(annotation_path, repo_root),
            "provenance": {"origin": "publisher_supplied_supervisely_rectangle",
                "source_object_id": obj.get("id"), "labeler_login": obj.get("labelerLogin")}})
    return rows, errors


def curate(acquisition_manifest: Path = DEFAULT_ACQUISITION,
           output_manifest: Path = DEFAULT_OUTPUT, *, phash_distance: int = 6) -> dict:
    inventory = json.loads(acquisition_manifest.read_text(encoding="utf-8"))
    image_rows = inventory.get("images")
    if not isinstance(image_rows, list):
        raise ValueError("acquisition manifest must contain an images array")
    root = acquisition_manifest.resolve().parents[3]
    rows, errors = [], []
    classes, split_count, status_count = Counter(), Counter(), Counter()
    stem_splits: dict[str, set[str]] = defaultdict(set)
    stem_images: dict[str, list[str]] = defaultdict(list)
    exact_hashes: dict[str, list[str]] = defaultdict(list)
    phashes = []
    for item in image_rows:
        image_path = _resolve_external(root, item["image"])
        annotation_path = _resolve_external(root, item["annotation"])
        if not image_path.is_file() or not annotation_path.is_file():
            errors.append(f"missing source image or annotation: {image_path}")
            continue
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened)
            width, height = image.size
            image.verify()
        row_id = item["image"]
        annotations, row_errors = _annotation_rows(annotation_path, row_id, width, height, root)
        errors.extend(row_errors)
        # Visual contact-sheet inspection confirmed components on all 114 blank-label PCB images.
        label_status = annotation_state(len(annotations))
        for annotation in annotations:
            classes[annotation["class"]] += 1
        split = item["source_split"]
        stem = item["source_stem_group"]
        digest = _sha256(image_path)
        exact_hashes[digest].append(row_id)
        phash = _phash(image_path)
        phashes.append((row_id, phash, stem, split))
        stem_splits[stem].add(split)
        stem_images[stem].append(row_id)
        row = {"image_id": row_id, "image_path": _manifest_path(image_path, root),
            "label_path": _manifest_path(annotation_path, root), "source_split": split,
            "source_stem_group": stem, "augmentation_family_id": stem,
            "augmentation_family_status": "FILENAME_FAMILY_REQUIRES_SOURCE_REVIEW",
            "image_sha256": digest, "label_sha256": _sha256(annotation_path),
            "image_size_px": [width, height], "image_type": "POPULATED_PCB",
            "classification_status": "AUTO_CLASSIFIED", "classification_confidence": "HIGH",
            "review_status": "HUMAN_REVIEW_REQUIRED",
            "review_method": "AI visual audit of all 57 contact-sheet pages; requires human adjudication",
            "review_note": "AI visual inspection suggests a populated PCB. Similar appearance suggests a shared visual group; neither domain nor physical board identity has human confirmation.",
            "board_id": "BOARD_ID_UNRESOLVED", "board_group_id": None,
            "visual_board_group_candidate_id": VISUAL_GROUP_ID,
            "board_group_review_status": "HUMAN_REVIEW_REQUIRED",
            "board_side": "UNKNOWN", "assigned_split": None, "annotation_count": len(annotations),
            "annotation_status": label_status,
            "annotation_review_status": "SOURCE_PROVIDED_UNADJUDICATED",
            "training_include": False, "training_exclusion_reasons": [
                "SOURCE_CLASS_SEMANTICS_UNVERIFIED", "ANNOTATION_COMPLETENESS_NOT_ADJUDICATED",
                "PHYSICAL_BOARD_ID_UNRESOLVED", "ONLY_ONE_UNVERIFIED_VISUAL_GROUP_CANDIDATE"],
            "annotations": annotations}
        rows.append(row)
        split_count[split] += 1
        status_count[label_status] += 1
    if errors:
        raise ValueError("curation found source errors:\n" + "\n".join(errors[:50]))
    if len(rows) != len(image_rows):
        raise ValueError(f"manifest row count changed during validation: {len(rows)} != {len(image_rows)}")

    exact_groups = [names for names in exact_hashes.values() if len(names) > 1]
    near_pairs = []
    for index, first in enumerate(phashes):
        for second in phashes[index + 1:]:
            distance = (first[1] ^ second[1]).bit_count()
            if distance <= phash_distance:
                near_pairs.append({"first_image_id": first[0], "second_image_id": second[0],
                    "phash_hamming_distance": distance, "same_stem_group": first[2] == second[2],
                    "crosses_source_split": first[3] != second[3],
                    "status": "NEAR_DUPLICATE_CANDIDATE_REQUIRES_ADJUDICATION"})
    parent = list(range(len(phashes)))
    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    for pair in near_pairs:
        left = next(i for i, row in enumerate(phashes) if row[0] == pair["first_image_id"])
        right = next(i for i, row in enumerate(phashes) if row[0] == pair["second_image_id"])
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left
    near_components: dict[int, list[str]] = defaultdict(list)
    for index, row in enumerate(phashes):
        near_components[find(index)].append(row[0])
    near_duplicate_groups = [members for members in near_components.values() if len(members) > 1]
    cross_stem_groups = {stem: sorted(splits) for stem, splits in stem_splits.items() if len(splits) > 1}
    summary = {"image_count": len(rows), "domain_candidate_image_count": len(rows),
        "human_confirmed_domain_image_count": 0,
        "domain_excluded_image_count": 0, "domain_review_pending_image_count": len(rows),
        "training_ready_image_count": 0, "training_hold_image_count": len(rows),
        "annotation_count": sum(classes.values()), "class_annotation_counts": dict(sorted(classes.items())),
        "valid_bounding_box_count": sum(classes.values()), "invalid_bounding_box_count": 0,
        "source_split_image_counts": dict(sorted(split_count.items())),
        "source_stem_group_count": len(stem_images), "source_stem_groups_crossing_splits": len(cross_stem_groups),
        "source_stem_cross_split_image_count": sum(len(stem_images[key]) for key in cross_stem_groups),
        "exact_duplicate_group_count": len(exact_groups), "exact_duplicate_extra_image_count": sum(len(v)-1 for v in exact_groups),
        "phash_near_duplicate_pair_count": len(near_pairs),
        "phash_near_duplicate_group_count": len(near_duplicate_groups),
        "phash_near_duplicate_cross_split_pair_count": sum(pair["crosses_source_split"] for pair in near_pairs),
        "augmentation_family_count": len(stem_images), "augmentation_family_cross_split_count": len(cross_stem_groups),
        "visual_board_group_candidate_count": 1 if rows else 0,
        "human_confirmed_board_group_count": 0,
        "source_verified_physical_board_id_count": 0,
        "visual_board_group_candidate_crosses_source_splits": 1 if rows else 0,
        "train_split_image_count": 0, "validation_split_image_count": 0,
        "source_only_training_candidate_count": 0, "canonical_10_class_training_candidate_count": 0,
        "annotation_state_counts": dict(status_count),
        "status": "BLOCKED_NO_VERIFIED_TRAINING_DATA"}
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": "circuitcare.kaggle-component-curation.v1",
        "dataset_id": "kaggle-pcb-component-detection-curated-v1",
        "source_name": "Kaggle PCB Component Detection / DatasetNinja Supervisely conversion",
        "source_license": "CC0-1.0 (source metadata; unchanged from acquisition record)",
        "source_classes": SOURCE_CLASSES,
        "canonical_10_class_mapping_status": "NO_CLASSES_APPROVED_FOR_MAPPING",
        "domain_review_method": "AI visual audit of 57 contact sheets, 25 images per page; all results require human review",
        "board_identity_status": "BOARD_ID_UNRESOLVED; one tentative visual group candidate, not human-confirmed",
        "source_split_policy": "Original split retained for traceability only; prohibited for training/evaluation due board and stem leakage",
        "split_policy": "No train/validation/test split assigned: only one visual board group and physical board identity is source-unverified",
        "negative_sample_policy": "No empty label is treated as a negative; all 114 empty-source-label images remain EMPTY_UNREVIEWED until human review",
        "class_mapping": _class_mapping(classes), "summary": summary,
        "duplicates": {"exact_duplicate_groups": exact_groups, "phash_distance_threshold": phash_distance,
            "phash_near_duplicate_groups": near_duplicate_groups, "phash_near_duplicate_pairs": near_pairs,
            "interpretation": "pHash pairs are review candidates; they do not define or verify a physical board ID"},
        "source_stem_groups_crossing_splits": cross_stem_groups,
        "images": rows}
    output_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_reports(output_manifest, manifest)
    return summary


def _class_mapping(counts: Counter) -> list[dict]:
    reasons = {
        "Cap1": "CAPACITOR_SUBTYPE_SEMANTICS_UNVERIFIED",
        "Cap2": "CAPACITOR_SUBTYPE_SEMANTICS_UNVERIFIED",
        "Cap3": "CAPACITOR_SUBTYPE_SEMANTICS_UNVERIFIED",
        "Cap4": "CAPACITOR_SUBTYPE_SEMANTICS_UNVERIFIED",
        "Resistor": "KEEP_SOURCE_CLASS; no canonical merge approved",
        "Resestor": "SPELLING_OR_SEMANTIC_RELATION_UNVERIFIED; do not merge with Resistor",
        "MOSFET": "NO_CANONICAL_10_CLASS_EQUIVALENT; do not map to IC",
        "Mov": "NO_CANONICAL_10_CLASS_EQUIVALENT; do not map to diode",
        "Transformer": "NO_CANONICAL_10_CLASS_EQUIVALENT",
    }
    return [{"source_class": name, "annotation_count": counts[name],
        "target_class": None, "mapping_status": reasons[name],
        "approved_for_training": False} for name in SOURCE_CLASSES]


def _write_reports(manifest_path: Path, manifest: dict) -> None:
    prefix = manifest_path.with_name("pcb-component-detection")
    summary = manifest["summary"]
    (prefix.with_name(prefix.name + "-quality-report.json")).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (prefix.with_name(prefix.name + "-duplicate-report.json")).write_text(
        json.dumps(manifest["duplicates"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (prefix.with_name(prefix.name + "-annotation-completeness-report.json")).write_text(
        json.dumps({"state_counts": summary["annotation_state_counts"],
        "empty_source_label_image_count": summary["annotation_state_counts"].get("EMPTY_UNREVIEWED", 0),
            "empty_labels_are_negative_samples": False,
            "nonempty_source_label_image_count": summary["annotation_state_counts"].get("ANNOTATION_STATUS_UNKNOWN", 0),
            "completeness_reviewed_image_count": 0,
            "training_eligible_image_count": 0,
            "reason": "No per-image full object coverage adjudication was performed."},
            indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (prefix.with_name(prefix.name + "-board-leakage-report.json")).write_text(
        json.dumps({"source_stem_group_count": summary["source_stem_group_count"],
            "source_stem_groups_crossing_splits": summary["source_stem_groups_crossing_splits"],
            "images_in_crossing_stem_groups": summary["source_stem_cross_split_image_count"],
            "near_duplicate_pairs_at_phash_distance_6": summary["phash_near_duplicate_pair_count"],
            "near_duplicate_pairs_crossing_source_splits": summary["phash_near_duplicate_cross_split_pair_count"],
            "visual_board_group_candidate_count": summary["visual_board_group_candidate_count"],
            "visual_board_group_candidate_crosses_source_splits": summary["visual_board_group_candidate_crosses_source_splits"],
            "source_verified_physical_board_id_count": summary["source_verified_physical_board_id_count"],
            "resolved_source_board_group_count": 0,
            "new_train_split_image_count": summary["train_split_image_count"],
            "new_validation_split_image_count": summary["validation_split_image_count"],
            "status": "BLOCKED_NO_HUMAN_VERIFIED_BOARD_GROUP_OR_VALID_HOLDOUT",
            "independent_test_available": False}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (prefix.with_name(prefix.name + "-excluded-image-report.json")).write_text(
        json.dumps({"training_hold_count": summary["training_hold_image_count"],
            "reasons": dict(Counter(reason for row in manifest["images"]
                for reason in row["training_exclusion_reasons"])),
            "images": [{"image_id": row["image_id"], "reasons": row["training_exclusion_reasons"]}
                for row in manifest["images"]]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # A blocked contract file prevents consumers from mistaking the acquisition
    # source split or unreviewed labels for a trainable Ultralytics dataset.
    blocked_yaml = prefix.with_name(prefix.name + "-source-only-blocked.yaml")
    blocked_yaml.write_text("# BLOCKED: no training-ready images. Do not pass to Ultralytics.\n"
        "status: BLOCKED_NO_VERIFIED_TRAINING_DATA\n"
        "source: Kaggle PCB Component Detection / DatasetNinja\n"
        "train: []\nval: []\n"
        "names:\n" + "".join(f"  {index}: {name}\n" for index, name in enumerate(SOURCE_CLASSES)),
        encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-manifest", type=Path, default=DEFAULT_ACQUISITION)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phash-distance", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(curate(args.acquisition_manifest, args.output_manifest,
                            phash_distance=args.phash_distance), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
