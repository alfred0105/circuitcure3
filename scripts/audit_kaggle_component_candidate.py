#!/usr/bin/env python3
"""Audit Kaggle component labels and generate a human review queue.

No photo, component class, board identity or annotation is certified by this script.
It operates on the existing converted candidate manifest, leaves source data intact,
and NEVER creates a training-ready dataset or silently changes source splits.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ALLOWED_SPLITS = {"train", "validation", "test"}
TARGET_DOMAINS = {"POPULATED_PCB", "PARTIALLY_POPULATED_PCB", "PCB_COMPONENT_CLOSEUP"}
DOMAIN_TYPES = TARGET_DOMAINS | {
    "BARE_PCB", "BREADBOARD", "ISOLATED_COMPONENT",
    "LOOSE_COMPONENT_COLLECTION", "UNKNOWN",
}
ANNOTATION_STATES = {
    "ANNOTATION_COMPLETE", "PARTIAL_ANNOTATION", "EMPTY_REVIEWED",
    "EMPTY_UNREVIEWED", "ANNOTATION_STATUS_UNKNOWN",
}
REVIEW_FIELDS = (
    "image_id", "source_split", "source_stem_group", "image_path",
    "annotation_count", "suggested_image_type", "reviewer", "reviewed_at_utc",
    "confirmed_image_type", "annotation_completeness", "confirmed_board_id",
    "board_id_basis", "review_notes",
)


def _review_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(REVIEW_FIELDS[6:]).issubset(reader.fieldnames):
            raise ValueError("completed review CSV is missing required review columns")
        result: dict[str, dict[str, str]] = {}
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if not image_id or image_id in result:
                raise ValueError(f"empty or duplicate reviewed image_id: {image_id}")
            result[image_id] = {key: (value or "").strip() for key, value in row.items()}
        return result


def audit(manifest: dict, reviewed: dict[str, dict[str, str]] | None = None) -> tuple[dict, list[dict]]:
    reviewed = reviewed or {}
    images = manifest.get("images")
    if not isinstance(images, list):
        raise ValueError("candidate manifest has no images list")
    ids: set[str] = set()
    by_stem: dict[str, set[str]] = defaultdict(set)
    by_sha: dict[str, set[str]] = defaultdict(set)
    by_reviewed_board: dict[str, set[str]] = defaultdict(set)
    review_queue: list[dict] = []
    source_splits, domains, annotation_states = Counter(), Counter(), Counter()
    confirmed_count = 0
    reviewed_board_groups: set[str] = set()
    blocked = Counter()

    for row in images:
        image_id = row.get("image_id")
        split = row.get("source_split")
        stem = row.get("source_stem_group")
        if not isinstance(image_id, str) or not image_id or image_id in ids:
            raise ValueError(f"missing or duplicate image_id: {image_id!r}")
        if split not in ALLOWED_SPLITS or not isinstance(stem, str) or not stem:
            raise ValueError(f"{image_id}: missing source split or augmentation stem")
        ids.add(image_id)
        source_splits[split] += 1
        by_stem[stem].add(split)
        sha = row.get("image_sha256")
        if isinstance(sha, str) and sha:
            by_sha[sha].add(split)

        proposed = row.get("image_type", "UNKNOWN")
        human = reviewed.get(image_id)
        confirmed_type = "UNKNOWN"
        annotation_state = "ANNOTATION_STATUS_UNKNOWN"
        board_id = ""
        if human is not None:
            if not human.get("reviewer") or not human.get("reviewed_at_utc"):
                raise ValueError(f"{image_id}: reviewed image lacks reviewer/time")
            confirmed_type = human.get("confirmed_image_type", "")
            annotation_state = human.get("annotation_completeness", "")
            if confirmed_type not in DOMAIN_TYPES or annotation_state not in ANNOTATION_STATES:
                raise ValueError(f"{image_id}: invalid reviewed domain/annotation status")
            board_id = human.get("confirmed_board_id", "")
            basis = human.get("board_id_basis", "")
            if board_id and basis not in {"SOURCE_VERIFIED", "HUMAN_VERIFIED"}:
                raise ValueError(f"{image_id}: confirmed board ID has no verified basis")
            confirmed_count += 1
            if board_id:
                reviewed_board_groups.add(board_id)
                by_reviewed_board[board_id].add(split)
        domains[confirmed_type] += 1
        annotation_states[annotation_state] += 1

        label_count = row.get("annotation_count", 0)
        if not isinstance(label_count, int) or label_count < 0:
            raise ValueError(f"{image_id}: invalid annotation count")
        if confirmed_type not in TARGET_DOMAINS:
            blocked["DOMAIN_NOT_CONFIRMED_FOR_TRAINING"] += 1
        if annotation_state != "ANNOTATION_COMPLETE":
            blocked["ANNOTATION_COMPLETENESS_NOT_CONFIRMED"] += 1
        if not board_id:
            blocked["BOARD_ID_UNRESOLVED"] += 1
        if label_count == 0 and annotation_state != "EMPTY_REVIEWED":
            blocked["EMPTY_LABEL_NOT_VERIFIED"] += 1

        review_queue.append({
            "image_id": image_id, "source_split": split, "source_stem_group": stem,
            "image_path": row.get("image_path", ""), "annotation_count": label_count,
            "suggested_image_type": proposed,
            "reviewer": human.get("reviewer", "") if human else "",
            "reviewed_at_utc": human.get("reviewed_at_utc", "") if human else "",
            "confirmed_image_type": confirmed_type if human else "",
            "annotation_completeness": annotation_state if human else "",
            "confirmed_board_id": board_id,
            "board_id_basis": human.get("board_id_basis", "") if human else "",
            "review_notes": human.get("review_notes", "") if human else "",
        })
    unknown_reviews = set(reviewed) - ids
    if unknown_reviews:
        raise ValueError(f"review refers to missing image IDs: {sorted(unknown_reviews)[:5]}")

    source_conflicts = sorted(k for k, v in by_stem.items() if len(v) > 1)
    sha_conflicts = sorted(k for k, v in by_sha.items() if len(v) > 1)
    board_conflicts = sorted(k for k, v in by_reviewed_board.items() if len(v) > 1)
    report = {
        "status": "CURATION_AUDIT_ONLY_NOT_TRAINING_READY",
        "source": manifest.get("source"), "source_license": manifest.get("source_license"),
        "image_count": len(images), "reviewed_image_count": confirmed_count,
        "unreviewed_image_count": len(images) - confirmed_count,
        "source_split_counts": dict(source_splits),
        "source_stem_group_count": len(by_stem),
        "source_stem_cross_split_group_count": len(source_conflicts),
        "source_stem_cross_split_groups": source_conflicts,
        "exact_sha_cross_split_count": len(sha_conflicts),
        "exact_sha_cross_split_hashes": sha_conflicts,
        "confirmed_board_group_count": len(reviewed_board_groups),
        "confirmed_board_cross_split_count": len(board_conflicts),
        "confirmed_board_cross_split_ids": board_conflicts,
        "human_confirmed_domains": dict(domains),
        "human_annotation_completeness": dict(annotation_states),
        "blocker_image_counts": dict(blocked),
        "canonical_10_class_training_ready_images": 0,
        "source_only_training_ready_images": 0,
        "reason": [
            "Source class meanings and annotation completeness need human adjudication.",
            "The source splits overlap by augmentation stem; reviewed board groups require a newly generated split.",
            "This metadata-only audit does not promote images to training or count physical PCB IDs.",
        ],
    }
    return report, review_queue


def run(manifest_path: Path, output_dir: Path, review_csv: Path | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report, queue = audit(manifest, _review_rows(review_csv))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "review-queue.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(queue)
    (output_dir / "audit-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="local candidate_manifest.json created by prepare_external_component_candidate.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, help="optional explicit human-review CSV")
    args = parser.parse_args()
    result = run(args.manifest, args.output_dir, args.review_csv)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
