"""Verify Kaggle curation is audit-only, explicit and leakage-aware."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_kaggle_component_candidate", ROOT / "scripts" / "audit_kaggle_component_candidate.py"
)
curation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curation)


def row(image_id, split, stem, count=1, sha=None):
    return {
        "image_id": image_id,
        "image_path": f"data/component_training/raw/external/{image_id}.jpg",
        "source_split": split,
        "source_stem_group": stem,
        "annotation_count": count,
        "image_sha256": sha or image_id,
        "image_type": "PCB_COMPONENT_CLOSEUP",
    }


def reviewed(domain="POPULATED_PCB", annotation="ANNOTATION_COMPLETE", board="board-1"):
    return {
        "reviewer": "reviewer-1", "reviewed_at_utc": "2026-11-06T01:00:00Z",
        "confirmed_image_type": domain, "annotation_completeness": annotation,
        "confirmed_board_id": board, "board_id_basis": "HUMAN_VERIFIED",
        "review_notes": "Visual review of original photo and annotations",
    }


def test_unreviewed_source_stem_leakage_is_not_promoted_to_ready():
    manifest = {"source": "Kaggle", "images": [
        row("a1", "train", "same-board-family", sha="aaa"),
        row("a2", "validation", "same-board-family", sha="bbb"),
        row("empty", "train", "second-family", count=0, sha="ccc"),
    ]}
    report, queue = curation.audit(manifest)
    assert report["image_count"] == 3
    assert report["unreviewed_image_count"] == 3
    assert report["source_stem_cross_split_group_count"] == 1
    assert report["source_stem_cross_split_groups"] == ["same-board-family"]
    assert report["canonical_10_class_training_ready_images"] == 0
    assert report["source_only_training_ready_images"] == 0
    assert report["blocker_image_counts"]["EMPTY_LABEL_NOT_VERIFIED"] == 1
    assert all(not x["reviewer"] and not x["confirmed_board_id"] for x in queue)


def test_reviewed_board_cross_split_is_reported_not_silently_reassigned():
    manifest = {"images": [
        row("a1", "train", "stem-1"),
        row("a2", "validation", "stem-2"),
    ]}
    report, _ = curation.audit(manifest, {"a1": reviewed(), "a2": reviewed()})
    assert report["reviewed_image_count"] == 2
    assert report["confirmed_board_group_count"] == 1
    assert report["confirmed_board_cross_split_count"] == 1
    assert report["confirmed_board_cross_split_ids"] == ["board-1"]
    assert report["canonical_10_class_training_ready_images"] == 0


def test_explicit_review_does_not_assume_complete_labels_or_board_identity():
    manifest = {"images": [row("x", "train", "x")]}
    review = reviewed(domain="ISOLATED_COMPONENT", annotation="PARTIAL_ANNOTATION", board="")
    report, queue = curation.audit(manifest, {"x": review})
    assert report["human_confirmed_domains"] == {"ISOLATED_COMPONENT": 1}
    assert report["blocker_image_counts"]["DOMAIN_NOT_CONFIRMED_FOR_TRAINING"] == 1
    assert report["blocker_image_counts"]["ANNOTATION_COMPLETENESS_NOT_CONFIRMED"] == 1
    assert report["blocker_image_counts"]["BOARD_ID_UNRESOLVED"] == 1
    assert queue[0]["annotation_completeness"] == "PARTIAL_ANNOTATION"


@pytest.mark.parametrize("review_override", [
    {"reviewer": ""},
    {"reviewed_at_utc": ""},
    {"confirmed_image_type": "UNSUPPORTED"},
    {"annotation_completeness": "ASSUMED_COMPLETE"},
    {"board_id_basis": "INFERRED"},
])
def test_invalid_or_unsubstantiated_manual_review_rejected(review_override):
    manifest = {"images": [row("a", "train", "stem")]}
    value = reviewed()
    value.update(review_override)
    with pytest.raises(ValueError):
        curation.audit(manifest, {"a": value})


def test_duplicate_image_ids_and_unknown_review_id_rejected():
    duplicate = row("same", "train", "stem")
    with pytest.raises(ValueError, match="duplicate image_id"):
        curation.audit({"images": [duplicate, duplicate]})
    with pytest.raises(ValueError, match="missing image IDs"):
        curation.audit({"images": [duplicate]}, {"unknown": reviewed()})


def test_run_generates_review_queue_without_modifying_source(tmp_path):
    manifest = tmp_path / "candidate.json"
    original = {"images": [row("a", "train", "same"), row("b", "validation", "same")]}
    manifest.write_text(json.dumps(original))
    before = manifest.read_bytes()
    result = curation.run(manifest, tmp_path / "audit")
    assert result["status"] == "CURATION_AUDIT_ONLY_NOT_TRAINING_READY"
    assert manifest.read_bytes() == before
    with (tmp_path / "audit" / "review-queue.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["reviewer"] == ""
    assert rows[0]["confirmed_image_type"] == ""
    assert (tmp_path / "audit" / "audit-report.json").is_file()
