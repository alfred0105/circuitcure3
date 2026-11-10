"""Conservative reconciliation helpers for detector, OCR, and review records."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


REFERENCE_CLASS_HINTS = {
    "R": "resistor", "C": "capacitor", "U": "IC", "D": "diode",
    "Q": "transistor", "L": "inductor", "F": "fuse", "J": "connector",
    "P": "connector", "CN": "connector", "T": "transformer", "Y": "oscillator",
}


def box_iou(left: list[float] | tuple[float, ...], right: list[float] | tuple[float, ...]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    ix = max(0.0, min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0])))
    iy = max(0.0, min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1])))
    intersection = ix * iy
    area_left = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    area_right = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def box_intersection_over_smaller(
    left: list[float] | tuple[float, ...], right: list[float] | tuple[float, ...],
) -> float:
    """Return intersection divided by the smaller box area (containment-aware)."""
    if len(left) != 4 or len(right) != 4:
        return 0.0
    intersection = max(0.0, min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0]))) * max(
        0.0, min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1]))
    )
    areas = [max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
             for box in (left, right)]
    smaller = min(areas)
    return intersection / smaller if smaller > 0 else 0.0


def boxes_overlap_candidate(
    candidate: list[float] | tuple[float, ...], existing: list[float] | tuple[float, ...],
    *, iou_threshold: float = 0.30, containment_threshold: float = 0.72,
) -> bool:
    """Treat substantial containment as overlap so nested duplicates are filtered."""
    return (box_iou(candidate, existing) >= iou_threshold
            or box_intersection_over_smaller(candidate, existing) >= containment_threshold)


def duplicate_box_metrics(
    left: list[float] | tuple[float, ...], right: list[float] | tuple[float, ...],
) -> dict[str, float]:
    """Measure overlap, relative size, and normalized center distance for a box pair."""
    if len(left) != 4 or len(right) != 4:
        return {"iou": 0.0, "smaller_coverage": 0.0, "area_ratio": 0.0,
                "center_distance_norm": float("inf")}
    boxes = [[float(value) for value in left], [float(value) for value in right]]
    areas = [max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) for box in boxes]
    intersection = max(0.0, min(boxes[0][2], boxes[1][2]) - max(boxes[0][0], boxes[1][0])) * max(
        0.0, min(boxes[0][3], boxes[1][3]) - max(boxes[0][1], boxes[1][1])
    )
    union = areas[0] + areas[1] - intersection
    centers = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes]
    distance = ((centers[0][0] - centers[1][0]) ** 2 + (centers[0][1] - centers[1][1]) ** 2) ** .5
    largest_scale = max(1.0, areas[0], areas[1]) ** .5
    return {"iou": intersection / union if union else 0.0,
            "smaller_coverage": intersection / min(areas) if min(areas) else 0.0,
            "area_ratio": min(areas) / max(areas) if max(areas) else 0.0,
            "center_distance_norm": distance / largest_scale}


def same_component_box_pair(
    left: list[float] | tuple[float, ...], right: list[float] | tuple[float, ...],
) -> bool:
    """Conservatively identify two detections of the same physical component.

    A small box fully inside a much larger box is deliberately retained: PCB body
    proposals can enclose a neighboring real part, so containment alone is not
    enough to suppress it.
    """
    metrics = duplicate_box_metrics(left, right)
    if metrics["area_ratio"] < 0.45:
        return False
    if metrics["center_distance_norm"] > 0.22:
        return False
    return (metrics["iou"] >= 0.50
            or (metrics["smaller_coverage"] >= 0.90 and metrics["area_ratio"] >= 0.55))


def find_group_box_candidates(
    detections: list[dict[str, Any]], *, min_children: int = 2,
) -> list[dict[str, Any]]:
    """Find a large container box around multiple spatially independent boxes.

    This is separate from duplicate detection: a parent is only flagged for Qwen
    review, and no box is removed here. The relative size, containment, and center
    spacing filters reduce duplicate and single-package false positives.
    """
    normalized = []
    for row in detections:
        box = row.get("bbox_xyxy")
        if not isinstance(box, (list, tuple)):
            box = [row.get(key) for key in ("x1", "y1", "x2", "y2")]
        component_id = row.get("detection_id", row.get("component_id"))
        if component_id and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            normalized.append({**row, "detection_id": str(component_id), "bbox_xyxy": list(map(float, box))})

    groups = []
    for parent in normalized:
        child_metrics = []
        parent_box = parent["bbox_xyxy"]
        for child in normalized:
            if child["detection_id"] == parent["detection_id"]:
                continue
            metrics = duplicate_box_metrics(parent_box, child["bbox_xyxy"])
            if (metrics["area_ratio"] <= 0.45 and metrics["smaller_coverage"] >= 0.72
                    and metrics["center_distance_norm"] <= 0.48):
                child_metrics.append((child, metrics))
        independent = []
        for child, metrics in sorted(child_metrics, key=lambda item: item[1]["area_ratio"]):
            child_box = child["bbox_xyxy"]
            if any(same_component_box_pair(child_box, other["bbox_xyxy"])
                   or box_iou(child_box, other["bbox_xyxy"]) >= 0.35 for other in independent):
                continue
            independent.append(child)
        if len(independent) < min_children:
            continue
        child_pair_metrics = [duplicate_box_metrics(left["bbox_xyxy"], right["bbox_xyxy"])
                              for index, left in enumerate(independent)
                              for right in independent[index + 1:]]
        if not child_pair_metrics or any(
                pair["center_distance_norm"] < 0.28 and pair["iou"] > 0.15
                for pair in child_pair_metrics):
            continue
        groups.append({
            "parent": parent,
            "children": independent,
            "metrics": {
                "child_count": len(independent),
                "containment_min": min(duplicate_box_metrics(parent_box, row["bbox_xyxy"])["smaller_coverage"]
                                        for row in independent),
                "child_parent_area_ratio_max": max(duplicate_box_metrics(parent_box, row["bbox_xyxy"])["area_ratio"]
                                                   for row in independent),
                "child_center_distance_min": min(pair["center_distance_norm"] for pair in child_pair_metrics),
                "child_iou_max": max(pair["iou"] for pair in child_pair_metrics),
            },
        })
    return sorted(groups, key=lambda item: (item["metrics"]["child_count"],
                                             item["metrics"]["containment_min"]), reverse=True)


def remove_unreviewed_legacy_secondary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop only unreviewed boxes from the removed YOLO second-pass feature."""
    return [row for row in rows if not (
        row.get("source") == "yolo_second_pass"
        and row.get("상태") not in {"확인", "제외"}
        and not row.get("user_edited")
    )]


def exclude_review_row(row: dict[str, Any], *, reason: str, review: dict[str, Any] | None = None) -> None:
    """Record a user-confirmed exclusion without deleting the component history."""
    row.update({"상태": "제외", "user_edited": True, "exclusion_reason": reason})
    if review is not None:
        row["ai_review"] = review


def board_visible_review_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hide excluded records from the default PCB overlay while retaining their history."""
    return [row for row in rows if isinstance(row, dict) and row.get("상태") != "제외"]


def deduplicate_component_boxes(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate the complete cross-model location set using score and geometry."""
    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    ordered = sorted(candidates, key=lambda item: float(item.get("score", item.get("confidence", 0.0)) or 0.0),
                     reverse=True)
    for candidate in ordered:
        box = candidate.get("box", candidate.get("bbox_xyxy"))
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        matching = next((row for row in kept
                         if same_component_box_pair(box, row.get("box", row.get("bbox_xyxy", [])))), None)
        if matching is None:
            kept.append(candidate)
        else:
            suppressed.append({"kept": matching, "suppressed": candidate,
                               "metrics": duplicate_box_metrics(box, matching.get("box", matching.get("bbox_xyxy", [])))})
    return kept, suppressed


def _edge_distance(left: list[float], right: list[float]) -> float:
    dx = max(float(left[0]) - float(right[2]), float(right[0]) - float(left[2]), 0.0)
    dy = max(float(left[1]) - float(right[3]), float(right[1]) - float(left[3]), 0.0)
    return (dx * dx + dy * dy) ** 0.5


def _class_hint(reference: str) -> str | None:
    import re

    match = re.match(r"\s*([A-Z]+)", reference.upper())
    return REFERENCE_CLASS_HINTS.get(match.group(1)) if match else None


def associate_ocr_with_components(
    detections: list[dict[str, Any]], ocr_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach nearby OCR as evidence, leaving class and review status unchanged."""
    rows = [deepcopy(row) for row in detections]
    for row in rows:
        row["ocr_records"] = []
        row["ocr_candidates"] = []
        row["reference_designators"] = []
        row["ocr_class_hints"] = []
        row["ocr_association_status"] = "NO_TEXT"
    for record in ocr_records:
        text_box = record.get("bbox_xyxy")
        text = str(record.get("raw_text", record.get("text", ""))).strip()
        if not text or not isinstance(text_box, (list, tuple)) or len(text_box) != 4:
            continue
        ranked = []
        for index, row in enumerate(rows):
            box = row.get("bbox_xyxy")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            width = max(1.0, float(box[2]) - float(box[0]))
            height = max(1.0, float(box[3]) - float(box[1]))
            distance = _edge_distance([float(value) for value in text_box], [float(value) for value in box])
            threshold = max(24.0, ((width * width + height * height) ** 0.5) * 1.5)
            if distance <= threshold:
                ranked.append((distance / threshold, index))
        if not ranked:
            continue
        ranked.sort()
        ambiguous = len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 0.2
        target_indices = [ranked[0][1]] if not ambiguous else [ranked[0][1], ranked[1][1]]
        ref_candidates = record.get("reference_candidates") or []
        for index in target_indices:
            row = rows[index]
            row["ocr_records"].append({**record, "association_status": "AMBIGUOUS" if ambiguous else "NEAR_COMPONENT"})
            row["ocr_candidates"] = list(dict.fromkeys([*row["ocr_candidates"], text]))
            row["reference_designators"] = list(dict.fromkeys([
                *row["reference_designators"], *[str(ref).upper() for ref in ref_candidates],
            ]))
            row["ocr_class_hints"] = list(dict.fromkeys([
                *row["ocr_class_hints"], *[hint for ref in ref_candidates if (hint := _class_hint(str(ref)))],
            ]))
            row["ocr_association_status"] = "AMBIGUOUS" if ambiguous else "ASSOCIATED"
    return rows


def preserve_review_rows(
    old_rows: list[dict[str, Any]], detections: list[dict[str, Any]], *, match_threshold: float = 0.30,
) -> list[dict[str, Any]]:
    """Reuse detector IDs and keep user-authored/reviewed rows across re-inference."""
    proposals = [row for row in detections if isinstance(row, dict) and row.get("bbox_xyxy")]
    previous = [deepcopy(row) for row in old_rows if isinstance(row, dict)]
    matches = []
    for new_index, proposal in enumerate(proposals):
        for old_index, old in enumerate(previous):
            old_box = [old.get(key) for key in ("x1", "y1", "x2", "y2")]
            if all(isinstance(value, (int, float)) for value in old_box):
                score = box_iou(proposal["bbox_xyxy"], old_box)
                if old.get("상태") == "제외":
                    matched = score >= 0.75 or same_component_box_pair(proposal["bbox_xyxy"], old_box)
                else:
                    matched = score >= match_threshold or same_component_box_pair(proposal["bbox_xyxy"], old_box)
                if matched:
                    # Prefer an active record over an excluded duplicate when a new
                    # detection geometrically matches both.
                    active_priority = int(old.get("상태") != "제외")
                    geometry_score = max(score, duplicate_box_metrics(proposal["bbox_xyxy"], old_box)["smaller_coverage"])
                    matches.append((geometry_score, active_priority, new_index, old_index))
    matched_new: set[int] = set()
    matched_old: set[int] = set()
    assignments: dict[int, dict[str, Any]] = {}
    for _score, _active_priority, new_index, old_index in sorted(matches, reverse=True):
        if new_index in matched_new or old_index in matched_old:
            continue
        matched_new.add(new_index)
        matched_old.add(old_index)
        assignments[new_index] = previous[old_index]

    result = []
    for index, proposal in enumerate(proposals):
        old = assignments.get(index)
        bbox = [float(value) for value in proposal["bbox_xyxy"]]
        row = {
            "component_id": (old or {}).get("component_id"),
            "소자 종류": proposal.get("class_name") or "unknown",
            "상태": "검수 필요",
            "신뢰도": float(proposal.get("confidence") or 0.0),
            "location_confidence": proposal.get("location_confidence", proposal.get("body_confidence", proposal.get("confidence"))),
            "type_confidence": proposal.get("type_confidence", proposal.get("confidence")),
            "match_score": proposal.get("match_score"),
            "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
            "source": (old or {}).get("source") if (old or {}).get("source") in {"manual", "qwen_suggestion", "yolo_second_pass"} else "yolo",
        }
        if old:
            row["component_id"] = old.get("component_id")
            # A reviewed or explicitly edited record owns its class and geometry.
            if (old.get("source") in {"manual", "qwen_suggestion"}
                    or old.get("상태") in {"확인", "제외"} or old.get("user_edited")):
                for key in ("소자 종류", "상태", "x1", "y1", "x2", "y2", "user_edited"):
                    if key in old:
                        row[key] = old[key]
            for key in ("ocr_records", "ocr_candidates", "reference_designators", "ocr_class_hints", "ocr_association_status",
                        "ai_review", "ai_proposed_class", "duplicate_of", "exclusion_reason",
                        "group_split_member_ids", "group_parent_id", "group_review_confirmed",
                        "group_review_decision"):
                if key in old:
                    row[key] = old[key]
        result.append(row)

    # Keep manual additions, Qwen additions, and explicitly reviewed boxes even
    # when a later detector pass does not return the same geometry.
    for old_index, old in enumerate(previous):
        keep = old.get("source") in {"manual", "qwen_suggestion", "yolo_second_pass"} or old.get("상태") in {"확인", "제외"} or old.get("user_edited")
        if keep and old_index not in matched_old:
            result.append(old)
    return result
