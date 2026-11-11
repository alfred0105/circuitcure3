"""Bounded, coordinate-safe PCB context for local diagnostic assistance."""

from __future__ import annotations

import json
import math
from typing import Any


DEFAULT_TOKEN_BUDGET = 6000


def _box(row: dict[str, Any]) -> list[float] | None:
    value = row.get("bbox_xyxy")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _token_estimate(value: Any) -> int:
    # A conservative, dependency-free estimate for mixed Korean/JSON content.
    return max(1, math.ceil(len(json.dumps(value, ensure_ascii=False, default=str)) / 3.2))


def build_pcb_context(
    *, components: list[dict[str, Any]], photo_records: tuple[Any, ...] | list[Any],
    diagnosis: dict[str, Any], suspected_components: list[dict[str, Any]] | None = None,
    board_registry: dict[str, Any] | None = None,
    token_budget: int = DEFAULT_TOKEN_BUDGET, max_focus_components: int = 24,
) -> dict[str, Any]:
    """Build Stage A summary and bounded Stage B detail without joining photo frames.

    Component IDs in this application are the IDs accepted by the active diagnosis
    snapshot. Photo-local coordinates are included only when the source frame and
    dimensions are known; spatial neighbors are scoped to a single photo/frame.
    """
    photos: dict[str, dict[str, Any]] = {}
    for photo in photo_records:
        photo_id = str(getattr(photo, "image_id", "") or "")
        if photo_id:
            dimensions = getattr(photo, "image_dimensions", None)
            photos[photo_id] = {
                "photo_id": photo_id, "side": getattr(photo, "side", "unknown"),
                "capture_type": getattr(photo, "capture_type", "unknown"),
                "width": dimensions[0] if dimensions and len(dimensions) == 2 else None,
                "height": dimensions[1] if dimensions and len(dimensions) == 2 else None,
            }

    blocks = {str(row.get("component_id")): row for row in diagnosis.get("functional_blocks", [])
              if isinstance(row, dict) and row.get("component_id")}
    registry_sources = {}
    if isinstance(board_registry, dict):
        registry_sources = {str(row.get("board_component_id")): list(row.get("source_components", []))
                            for row in board_registry.get("components", [])
                            if isinstance(row, dict) and row.get("board_component_id")}
    measures: dict[str, list[dict[str, Any]]] = {}
    for row in diagnosis.get("measurement_evidence", []):
        if isinstance(row, dict) and row.get("component_id"):
            measures.setdefault(str(row["component_id"]), []).append({
                key: row.get(key) for key in ("measurement_type", "value", "unit", "verification_status", "fault_id",
                             "measurement_target", "operating_conditions", "assessment", "verdict", "reason",
                             "reference_verification_status")
                if key in row
            })
    suspects = [row for row in (suspected_components or []) if isinstance(row, dict)]
    suspect_by_id: dict[str, dict[str, Any]] = {}
    for suspect in suspects:
        suspect_ids = {str(suspect.get("component_id") or ""),
                       str(suspect.get("board_component_id") or "")}
        for component in components:
            component_ids = {str(component.get("component_id") or ""),
                             str(component.get("board_component_id") or "")}
            if suspect_ids & component_ids:
                suspect_by_id[str(component.get("component_id"))] = suspect
    details: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    photo_counts: dict[str, int] = {}
    for row in components:
        component_id = str(row.get("component_id") or "")
        if not component_id:
            continue
        kind = str(row.get("component_type", row.get("type")) or "unknown")
        type_counts[kind] = type_counts.get(kind, 0) + 1
        source_ids = row.get("source_backend_photo_ids", row.get("source_photo_ids", row.get("photo_ids", []))) or []
        photo_id = str(row.get("photo_id") or (source_ids[0] if len(source_ids) == 1 else ""))
        photo = photos.get(photo_id, {})
        if photo_id:
            photo_counts[photo_id] = photo_counts.get(photo_id, 0) + 1
        bbox = _box(row)
        width, height = photo.get("width"), photo.get("height")
        normalized = None
        coordinate_frame = row.get("coordinate_frame", "photo_local_px" if photo_id else "UNKNOWN")
        if (bbox and photo_id and coordinate_frame in {"source_image_px", "photo_local_px"}
                and width and height and width > 0 and height > 0):
            normalized = {
                "center_x": round(((bbox[0] + bbox[2]) / 2) / width, 5),
                "center_y": round(((bbox[1] + bbox[3]) / 2) / height, 5),
                "width": round((bbox[2] - bbox[0]) / width, 5),
                "height": round((bbox[3] - bbox[1]) / height, 5),
            }
        block = blocks.get(component_id, {})
        suspect = suspect_by_id.get(component_id)
        board_id = str(row.get("board_component_id") or component_id)
        raw_sources = registry_sources.get(board_id, [])
        source_observations = []
        for source in raw_sources:
            source_photo_id = str(source.get("photo_id") or "")
            source_photo = photos.get(source_photo_id, {})
            source_box = _box(source)
            frame = str(source.get("coordinate_frame") or "UNKNOWN")
            source_normalized = None
            source_width, source_height = source_photo.get("width"), source_photo.get("height")
            if (source_box and frame.casefold() in {"source_photo_pixel", "source_image_px", "photo_local_px"}
                    and source_width and source_height and source_width > 0 and source_height > 0):
                source_normalized = {
                    "center_x": round(((source_box[0] + source_box[2]) / 2) / source_width, 5),
                    "center_y": round(((source_box[1] + source_box[3]) / 2) / source_height, 5),
                    "width": round((source_box[2] - source_box[0]) / source_width, 5),
                    "height": round((source_box[3] - source_box[1]) / source_height, 5),
                }
            source_observations.append({
                "photo_id": source_photo_id or None,
                "photo_local_component_id": source.get("component_id"),
                "detection_id": source.get("detection_id"),
                "coordinate_frame": frame,
                "bbox_xyxy": source_box,
                "normalized_geometry": source_normalized,
                "type": source.get("component_type", "unknown"),
                "detection_confidence": source.get("detection_confidence"),
                "review_status": source.get("review_status", "UNCONFIRMED"),
                "presence_status": source.get("presence_status", "UNCONFIRMED"),
                "ocr_marking": source.get("ocr_marking"),
                "ocr_candidates": source.get("ocr_candidates", []),
                "possible_part_number": source.get("possible_part_number"),
                "source_observation_fingerprint": source.get("source_observation_fingerprint"),
            })
        details.append({
            "component_id": component_id,
            "board_component_id": row.get("board_component_id"),
            "photo_id": photo_id or None,
            "photo_local_component_id": row.get("photo_local_component_id", component_id),
            "photo": photo or None,
            "source_observations": source_observations,
            "coordinate_frame": coordinate_frame,
            "bbox_xyxy": bbox,
            "normalized_geometry": normalized,
            "type": kind, "detection_confidence": row.get("detection_confidence"),
            "identification_status": row.get("identification_status", row.get("identity_status", row.get("presence_status", "UNKNOWN"))),
            "review_status": row.get("presence_status", "UNCONFIRMED"),
            "ocr_marking": row.get("ocr_marking"), "possible_part_number": row.get("possible_part_number"),
            "silkscreen_hints": [item for item in diagnosis.get("silkscreen_labels", [])
                                 if isinstance(item, dict) and item.get("component_id") == component_id][:4],
            "functional_roles": block.get("candidate_blocks", []),
            "functional_role_status": "INFERRED" if block else "UNKNOWN",
            "functional_role_evidence": block.get("role_evidence", {}),
            "nearby_component_ids": block.get("nearby_component_ids", []),
            "spatial_relation_scope": "same photo/frame only; not electrical connectivity",
            "spatial_connectivity_warning": "proximity does not establish electrical connection",
            "connectivity_status": "UNKNOWN",
            "measurements": measures.get(component_id, []),
            "user_suspicion": ({"fault_id": suspect.get("suspected_fault_mode"),
                                "source": "USER", "evidence_status": "UNVERIFIED"} if suspect else None),
        })

    by_id = {row["component_id"]: row for row in details}
    confirmed = [row for row in details if row["review_status"] == "CONFIRMED"]
    summaries = {
        "component_count": len(details), "confirmed_component_count": len(confirmed),
        "types": dict(sorted(type_counts.items())), "photos": dict(sorted(photo_counts.items())),
        "ocr_hints": [{"component_id": row["component_id"], "marking": row["ocr_marking"],
                        "part_number": row["possible_part_number"]} for row in details
                       if row.get("ocr_marking") or row.get("possible_part_number")][:12],
        "functional_role_hints": [{"component_id": row["component_id"], "roles": row["functional_roles"],
                                    "status": row["functional_role_status"]} for row in details
                                   if row["functional_roles"]][:12],
    }
    focus_ids: list[str] = []
    for row in suspects:
        cid = str(row.get("board_component_id") or row.get("component_id") or "")
        if cid in by_id and cid not in focus_ids:
            focus_ids.append(cid)
    for candidate in diagnosis.get("component_candidates", [])[:8]:
        cid = str(candidate.get("component_id") or "")
        if cid in by_id and cid not in focus_ids:
            focus_ids.append(cid)
    for row in details:
        if row["functional_roles"] and row["component_id"] not in focus_ids:
            focus_ids.append(row["component_id"])
    focus_ids.extend(row["component_id"] for row in confirmed if row["component_id"] not in focus_ids)
    focus_ids.extend(row["component_id"] for row in details if row["component_id"] not in focus_ids)

    base = {
        "schema_version": "pcb_context_v1",
        "connectivity_status": "UNKNOWN",
        "symptoms": diagnosis.get("symptoms", []),
        "symptom_text": str(diagnosis.get("symptom_text", ""))[:1200],
        "observations": diagnosis.get("observations", {}),
        "user_suspicions": [{"component_id": cid, "fault_id": row.get("suspected_fault_mode"),
                             "evidence_status": "UNVERIFIED"} for cid, row in suspect_by_id.items()
                            if cid in by_id],
        "hypotheses": [{"fault_id": row.get("fault_id"), "title": row.get("title"),
                        "supporting_evidence": row.get("supporting_evidence", [])[:4],
                        "opposing_evidence": row.get("opposing_evidence", [])[:4]}
                       for row in diagnosis.get("hypotheses", [])[:8]],
        "measurement_evidence": diagnosis.get("measurement_evidence", [])[:24],
        "summary": summaries,
        "focus_components": [],
    }
    budget = max(600, int(token_budget))
    for cid in focus_ids:
        if len(base["focus_components"]) >= max(1, max_focus_components):
            break
        base["focus_components"].append(by_id[cid])
        if _token_estimate(base) > budget:
            base["focus_components"].pop()
            break
    included = {row["component_id"] for row in base["focus_components"]}
    omitted = [row["component_id"] for row in details if row["component_id"] not in included]
    base["omitted_component_ids"] = omitted[:80]
    base["omitted_component_count"] = len(omitted)
    base["omitted_ids_truncated"] = len(omitted) > 80
    base["context_token_estimate"] = _token_estimate(base)
    base["token_budget"] = budget
    return base
