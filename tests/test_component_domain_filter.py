import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_component_dataset", ROOT / "scripts" / "prepare_component_dataset.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def row(tmp_path, image_id, domain, split, board, empty=False):
    base = tmp_path / "data" / "component_training"
    image = base / f"raw/{image_id}.jpg"
    label = base / f"raw/{image_id}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image-" + image_id.encode())
    label.write_bytes(b"" if empty else b"0 .5 .5 .1 .1\n")
    return {"image_id": image_id, "image_path": f"raw/{image_id}.jpg", "label_path": f"raw/{image_id}.txt",
            "image_type": domain, "review_status": "HUMAN_REVIEWED_CONTACT_SHEET",
            "classification_confidence": "HIGH", "source_split": split,
            "assigned_split": split if domain in prepare.TRAIN_TYPES else None,
            "board_group_id": board, "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest(), "annotation_count": 0 if empty else 1,
            "main_training_included": domain in prepare.TRAIN_TYPES,
            "target_absence_confirmed": domain in prepare.HARD_NEGATIVE_TYPES and empty,
            "hard_negative_eligible": domain in prepare.HARD_NEGATIVE_TYPES and empty}


def test_domain_policy_separates_training_and_verified_negatives(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    manifest = {"images": [row(tmp_path, "p1", "POPULATED_PCB", "train", "b1"),
                           row(tmp_path, "bread", "BREADBOARD", "train", "b2", empty=True),
                           row(tmp_path, "unknown", "UNKNOWN", "train", None)]}
    summary = prepare.validate_manifest(manifest)
    assert summary["included_image_count"] == 1
    assert summary["hard_negative_image_count"] == 1
    assert summary["excluded_images_by_type"] == {"BREADBOARD": 1, "UNKNOWN": 1}


def test_empty_annotation_without_target_absence_review_is_not_a_hard_negative(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    bread = row(tmp_path, "bread", "BREADBOARD", "train", "b1", empty=True)
    bread["target_absence_confirmed"] = False
    bread["hard_negative_eligible"] = False
    summary = prepare.validate_manifest({"images": [bread]})
    assert summary["hard_negative_image_count"] == 0


def test_group_cannot_leak_between_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    a = row(tmp_path, "a", "POPULATED_PCB", "train", "same-board")
    b = row(tmp_path, "b", "POPULATED_PCB", "validation", "same-board")
    with pytest.raises(ValueError, match="board-group split leakage"):
        prepare.validate_manifest({"images": [a, b]})
