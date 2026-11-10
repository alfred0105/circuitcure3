"""Human-in-the-loop visual review of YOLO component proposals using Ollama."""

from __future__ import annotations

import io
import json
import math
from typing import Any

from PIL import Image, ImageDraw

from circuitcare.local_llm import OllamaClient
from circuitcare.component_taxonomy import COMPONENT_LABELS, UNKNOWN_CLASS, normalize_component_class
from circuitcare.vision.box_reconciliation import (
    boxes_overlap_candidate, duplicate_box_metrics, find_group_box_candidates, same_component_box_pair,
)

DEFAULT_REVIEW_MODEL = "qwen3.5:9b"
REVIEW_CLASSES = COMPONENT_LABELS


class AIReviewStageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "class_reviews": {"type": "array", "items": {"type": "object", "properties": {
            "detection_id": {"type": "string"}, "decision": {"type": "string", "enum": ["suggest", "uncertain"]},
            "proposed_class": {"type": "string", "enum": list(REVIEW_CLASSES)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "string"},
        }, "required": ["detection_id", "decision", "proposed_class", "confidence", "evidence"], "additionalProperties": False}},
        "duplicate_reviews": {"type": "array", "items": {"type": "object", "properties": {
            "detection_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
            "decision": {"type": "string", "enum": ["same_component", "different_components", "uncertain"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "string"},
        }, "required": ["detection_ids", "decision", "confidence", "evidence"], "additionalProperties": False}},
        "group_reviews": {"type": "array", "items": {"type": "object", "properties": {
            "parent_id": {"type": "string"},
            "decision": {"type": "string", "enum": ["single_component", "multiple_components", "uncertain"]},
            "child_ids": {"type": "array", "items": {"type": "string"}},
            "new_components": {"type": "array", "items": {"type": "object", "properties": {
                "bbox_norm_999": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 999}, "minItems": 4, "maxItems": 4},
                "proposed_class": {"type": "string", "enum": list(REVIEW_CLASSES)},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {"type": "string"},
            }, "required": ["bbox_norm_999", "proposed_class", "confidence", "evidence"], "additionalProperties": False}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "string"},
        }, "required": ["parent_id", "decision", "child_ids", "new_components", "confidence", "evidence"], "additionalProperties": False}},
        "missed_component_candidates": {"type": "array", "items": {"type": "object", "properties": {
            "image_index": {"type": "integer", "minimum": 0, "maximum": 48},
            "bbox_norm_999": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 999}, "minItems": 4, "maxItems": 4},
            "proposed_class": {"type": "string", "enum": list(REVIEW_CLASSES)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "string"},
        }, "required": ["image_index", "bbox_norm_999", "proposed_class", "confidence", "evidence"], "additionalProperties": False}},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["class_reviews", "duplicate_reviews", "group_reviews", "missed_component_candidates", "notes"],
    "additionalProperties": False,
}

_SYSTEM = """You review a PCB photo and its YOLO component proposals. Use only the supplied shared component labels. Return unknown whenever appearance is ambiguous; return not_component only when the box clearly contains no physical component. Be conservative: only suggest a class when the physical component appearance is clear. Small SMD parts are often visually ambiguous. Use body markings and nearby OCR/silkscreen as cross-check evidence; a reference such as C12 may not label the closest body. Conflicting evidence means decision=uncertain and proposed_class=unknown. Do not claim certainty. Do not invent unseen parts. Existing boxes are identified by IDs; assess duplicate pairs only when their IDs are listed in duplicate_pair_crops, using the associated paired crop and surrounding board context. For group_candidate_crops, decide whether the parent box encloses multiple independent components or one physical component/package (including a connector); never split pins or internal regions of one package. Reuse listed child detection IDs first. If clearly visible separate components have no listed box, provide their tight body boxes in new_components, normalized 0..999 within that group's crop image. Do not duplicate listed child boxes. Return multiple_components only when the crop clearly shows separate physical component bodies. Search for at most a few clearly visible missed components, including when there are no detector proposals. Every proposed box is only a rough review candidate and must be manually checked. Each supplied image has an image_index from image_sources; missed boxes must be normalized relative to the image they cite. Return only the requested JSON object."""


def _review_images(
    image_bytes: bytes, detections: list[dict[str, Any]], *,
    group_candidates: list[dict[str, Any]] | None = None,
    review_diagnostics: list[dict[str, Any]] | None = None, max_crop_count: int = 10,
) -> tuple[list[bytes], tuple[int, int], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    with Image.open(io.BytesIO(image_bytes)) as source_image:
        image = source_image.convert("RGB")
    width, height = image.size
    guide = image.copy()
    guide.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    sx, sy = guide.width / width, guide.height / height
    painter = ImageDraw.Draw(guide)
    ordered = sorted(detections, key=lambda row: float(row.get("confidence") or 0.0))[:max_crop_count]
    pair_candidates = _duplicate_pair_candidates(detections, diagnostics=review_diagnostics)
    ids_to_crop = {str(row["detection_id"]) for row in ordered}
    for row in detections:
        box = row.get("bbox_xyxy")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value) * scale)) for value, scale in zip(box, (sx, sy, sx, sy))]
        detection_id = str(row.get("detection_id", "?"))
        color = "#ff9d2e" if detection_id in ids_to_crop else "#77d8ff"
        painter.rectangle((x1, y1, x2, y2), outline=color, width=3)
        painter.text((x1 + 2, max(0, y1 - 14)), detection_id, fill="white", stroke_width=2, stroke_fill="black")
    images = [guide]
    manifest = [{"image_index": 0, "kind": "annotated_full_image", "bbox_xyxy": [0, 0, width, height]}]
    for row in ordered:
        box = row.get("bbox_xyxy")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = map(float, box)
        pad_x, pad_y = max(3, (x2 - x1) * .25), max(3, (y2 - y1) * .25)
        crop = image.crop((max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
                           min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))))
        crop.thumbnail((720, 720), Image.Resampling.LANCZOS)
        images.append(crop)
        manifest.append({"image_index": len(images) - 1, "kind": "detection_crop",
                         "detection_id": str(row["detection_id"]),
                         "bbox_xyxy": [max(0.0, x1 - pad_x), max(0.0, y1 - pad_y),
                                       min(float(width), x2 + pad_x), min(float(height), y2 + pad_y)]})

    # Add overlapping native-resolution views so small parts remain visible to Qwen.
    tile_size, overlap, max_tiles = 1024, 0.25, 16
    if width > tile_size or height > tile_size:
        step = int(tile_size * (1 - overlap))
        def positions(length: int) -> list[int]:
            if length <= tile_size:
                return [0]
            starts = list(range(0, length - tile_size + 1, step))
            if starts[-1] != length - tile_size:
                edge = length - tile_size
                if edge - starts[-1] < step * 0.5:
                    starts[-1] = edge
                else:
                    starts.append(edge)
            return starts
        tile_boxes = [(x, y, min(width, x + tile_size), min(height, y + tile_size))
                      for y in positions(height) for x in positions(width)]
        if len(tile_boxes) > max_tiles:
            selected = sorted({round(index * (len(tile_boxes) - 1) / (max_tiles - 1))
                               for index in range(max_tiles)})
            tile_boxes = [tile_boxes[index] for index in selected]
        for x1, y1, x2, y2 in tile_boxes:
            images.append(image.crop((x1, y1, x2, y2)))
            manifest.append({"image_index": len(images) - 1, "kind": "native_resolution_tile",
                             "bbox_xyxy": [x1, y1, x2, y2]})
    pair_manifest = []
    for pair in pair_candidates:
        left, right = pair["detections"]
        a, b = left["bbox_xyxy"], right["bbox_xyxy"]
        pad_x = max(16.0, max(float(a[2]) - float(a[0]), float(b[2]) - float(b[0])) * .40)
        pad_y = max(16.0, max(float(a[3]) - float(a[1]), float(b[3]) - float(b[1])) * .40)
        crop_box = [max(0, int(min(a[0], b[0]) - pad_x)), max(0, int(min(a[1], b[1]) - pad_y)),
                    min(width, int(max(a[2], b[2]) + pad_x)), min(height, int(max(a[3], b[3]) + pad_y))]
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            continue
        crop = image.crop(tuple(crop_box))
        painter = ImageDraw.Draw(crop)
        for detection, color in ((left, "#ff9d2e"), (right, "#77d8ff")):
            box = detection["bbox_xyxy"]
            local_box = (int(box[0] - crop_box[0]), int(box[1] - crop_box[1]),
                         int(box[2] - crop_box[0]), int(box[3] - crop_box[1]))
            painter.rectangle(local_box, outline=color, width=4)
            painter.text((local_box[0] + 2, max(0, local_box[1] - 16)), detection["detection_id"],
                         fill="white", stroke_width=2, stroke_fill="black")
        crop.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        image_index = len(images)
        images.append(crop)
        pair_manifest.append({"image_index": image_index, "detection_ids": [left["detection_id"], right["detection_id"]],
                              "crop_bbox_xyxy": crop_box, "metrics": pair["metrics"]})
        manifest.append({"image_index": image_index, "kind": "duplicate_pair_crop",
                         "detection_ids": [left["detection_id"], right["detection_id"]],
                         "bbox_xyxy": crop_box})
    group_manifest = []
    for group in (group_candidates or [])[:8]:
        parent, children = group["parent"], group["children"]
        parent_box = parent["bbox_xyxy"]
        group_boxes = [parent_box, *[child["bbox_xyxy"] for child in children]]
        pad_x = max(10.0, (float(parent_box[2]) - float(parent_box[0])) * .12)
        pad_y = max(10.0, (float(parent_box[3]) - float(parent_box[1])) * .12)
        crop_box = [max(0, int(min(box[0] for box in group_boxes) - pad_x)),
                    max(0, int(min(box[1] for box in group_boxes) - pad_y)),
                    min(width, int(max(box[2] for box in group_boxes) + pad_x)),
                    min(height, int(max(box[3] for box in group_boxes) + pad_y))]
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            continue
        crop = image.crop(tuple(crop_box))
        painter = ImageDraw.Draw(crop)
        for detection, color in [(parent, "#ff9d2e"), *[(child, "#77d8ff") for child in children]]:
            box = detection["bbox_xyxy"]
            local_box = (int(box[0] - crop_box[0]), int(box[1] - crop_box[1]),
                         int(box[2] - crop_box[0]), int(box[3] - crop_box[1]))
            painter.rectangle(local_box, outline=color, width=4)
            painter.text((local_box[0] + 2, max(0, local_box[1] - 16)), detection["detection_id"],
                         fill="white", stroke_width=2, stroke_fill="black")
        crop.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        image_index = len(images)
        images.append(crop)
        child_ids = [child["detection_id"] for child in children]
        group_manifest.append({"image_index": image_index, "parent_id": parent["detection_id"],
                               "child_ids": child_ids, "crop_bbox_xyxy": crop_box,
                               "metrics": group["metrics"]})
        manifest.append({"image_index": image_index, "kind": "group_candidate_crop",
                         "parent_id": parent["detection_id"], "child_ids": child_ids,
                         "bbox_xyxy": crop_box})
    encoded = []
    for candidate in images:
        buffer = io.BytesIO()
        candidate.save(buffer, format="JPEG", quality=88)
        encoded.append(buffer.getvalue())
    return encoded, (width, height), manifest, pair_manifest, group_manifest


def _duplicate_pair_candidates(
    detections: list[dict[str, Any]], *, limit: int = 8,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Select only nearby, similar-scale pairs for focused visual re-review."""
    pairs = []
    for index, left in enumerate(detections):
        left_box = left.get("bbox_xyxy")
        if not isinstance(left_box, (list, tuple)) or len(left_box) != 4:
            continue
        for right in detections[index + 1:]:
            right_box = right.get("bbox_xyxy")
            if not isinstance(right_box, (list, tuple)) or len(right_box) != 4:
                continue
            metrics = duplicate_box_metrics(left_box, right_box)
            pair_ids = [str(left.get("detection_id", "")), str(right.get("detection_id", ""))]
            if metrics["area_ratio"] < .12:
                if diagnostics is not None:
                    diagnostics.append({"stage": "duplicate_pair_search", "detection_ids": pair_ids,
                                        "status": "rejected", "reason": "area_ratio_too_small", "metrics": metrics})
                continue
            if metrics["center_distance_norm"] > .80:
                if diagnostics is not None:
                    diagnostics.append({"stage": "duplicate_pair_search", "detection_ids": pair_ids,
                                        "status": "rejected", "reason": "centers_too_far", "metrics": metrics})
                continue
            if metrics["iou"] < .02 and metrics["smaller_coverage"] < .10:
                if diagnostics is not None:
                    diagnostics.append({"stage": "duplicate_pair_search", "detection_ids": pair_ids,
                                        "status": "rejected", "reason": "insufficient_overlap", "metrics": metrics})
                continue
            pairs.append({"detections": (left, right), "metrics": metrics,
                          "rank": metrics["iou"] + .35 * metrics["smaller_coverage"]
                          + max(0.0, .80 - metrics["center_distance_norm"]) * .15})
            if diagnostics is not None:
                diagnostics.append({"stage": "duplicate_pair_search", "detection_ids": pair_ids,
                                    "status": "candidate", "reason": "overlap_or_nearby_scale_match",
                                    "metrics": metrics})
    ranked = sorted(pairs, key=lambda item: item["rank"], reverse=True)
    if diagnostics is not None:
        for pair in ranked[limit:]:
            diagnostics.append({"stage": "duplicate_pair_search",
                                "detection_ids": [row.get("detection_id") for row in pair["detections"]],
                                "status": "deferred", "reason": "duplicate_crop_limit",
                                "metrics": pair["metrics"]})
    return ranked[:limit]


def _map_normalized_box(
    box: Any, image_index: Any, image_manifest: list[dict[str, Any]], *, min_size_px: float = 4.0,
) -> tuple[list[float] | None, str | None]:
    """Validate a Qwen box and map its cited image/crop coordinates to source pixels."""
    if type(image_index) is not int or not 0 <= image_index < len(image_manifest):
        return None, "invalid_image_index"
    if (not isinstance(box, list) or len(box) != 4
            or any(type(value) is not int or not 0 <= value <= 999 for value in box)):
        return None, "invalid_normalized_coordinates"
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None, "degenerate_normalized_box"
    region = image_manifest[image_index].get("bbox_xyxy")
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        return None, "invalid_source_region"
    left, top, right, bottom = map(float, region)
    if right <= left or bottom <= top:
        return None, "invalid_source_region"
    mapped = [left + x1 * (right - left) / 999, top + y1 * (bottom - top) / 999,
              left + x2 * (right - left) / 999, top + y2 * (bottom - top) / 999]
    if mapped[2] - mapped[0] < min_size_px or mapped[3] - mapped[1] < min_size_px:
        return None, "box_below_minimum_source_size"
    return mapped, None


def review_components_with_qwen(
    client: OllamaClient,
    model: str,
    image_bytes: bytes,
    detections: list[dict[str, Any]],
    ocr_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Request suggestions and validate every ID and normalized box before display."""
    identified = []
    for index, row in enumerate(detections, start=1):
        box = row.get("bbox_xyxy", [row.get("x1"), row.get("y1"), row.get("x2"), row.get("y2")])
        identified.append({"detection_id": str(row.get("component_id") or f"det-{index}"),
                           "class_name": normalize_component_class(row.get("class_name", row.get("소자 종류", "unknown"))),
                           "confidence": row.get("confidence", row.get("신뢰도")),
                           "bbox_xyxy": box,
                           "ocr_text": row.get("ocr_candidates", []),
                           "ocr_class_hints": row.get("ocr_class_hints", []),
                           "reference_designators": row.get("reference_designators", []),
                           "body_marking": row.get("body_marking")})
    candidate_diagnostics: list[dict[str, Any]] = []
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image_size = source.size
        group_candidates = find_group_box_candidates(identified, image_size=image_size,
                                                     diagnostics=candidate_diagnostics)
        for group in group_candidates[8:]:
            candidate_diagnostics.append({"stage": "candidate_search", "component_id": group["parent"]["detection_id"],
                                          "status": "deferred", "reason": "group_crop_limit"})
        images, (width, height), image_manifest, pair_manifest, group_manifest = _review_images(
            image_bytes, identified, group_candidates=group_candidates,
            review_diagnostics=candidate_diagnostics)
    except Exception as exc:
        raise AIReviewStageError("candidate_search", str(exc)) from exc
    payload = {
        "image_coordinates_px": {"width": width, "height": height},
        "annotated_image_box_ids": [row["detection_id"] for row in identified],
        "image_sources": image_manifest,
        "duplicate_pair_crops": pair_manifest,
        "group_candidate_crops": group_manifest,
        "detector_proposals": [{key: value for key, value in row.items() if key != "bbox_xyxy"}
                               | {"bbox_xyxy": [round(float(value), 1) for value in row["bbox_xyxy"]]}
                               for row in identified],
        "nearby_ocr_text_candidates": [{"text": row.get("raw_text", row.get("text", "")),
                                        "bbox_xyxy": row.get("bbox_xyxy"),
                                        "confidence": row.get("confidence"),
                                        "reference_candidates": row.get("reference_candidates", [])}
                                       for row in ocr_records[:120]],
        "coordinate_rule": "For each missed candidate, provide image_index and bbox_norm_999=[x1,y1,x2,y2] normalized 0..999 within that exact image.",
    }
    try:
        response = client.chat_json(
            model, [{"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": "Review the PCB image, YOLO proposals, and OCR evidence. Return conservative candidate findings.\n" + json.dumps(payload, ensure_ascii=False)}],
            output_schema=REVIEW_SCHEMA, images=images,
        )
        content = response["content"]
    except Exception as exc:
        raise AIReviewStageError("qwen", str(exc)) from exc
    known_ids = {row["detection_id"] for row in identified}
    allowed_pairs = {tuple(sorted(pair["detection_ids"])) for pair in pair_manifest}
    class_reviews = []
    for row in content.get("class_reviews", []):
        confidence = row.get("confidence")
        proposed = normalize_component_class(row.get("proposed_class"))
        if (row.get("detection_id") not in known_ids
                or row.get("decision") not in {"suggest", "uncertain"}
                or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence))
                or not 0 <= float(confidence) <= 1):
            continue
        if row.get("decision") == "uncertain":
            proposed = UNKNOWN_CLASS
        target = next(item for item in identified if item["detection_id"] == row["detection_id"])
        hints = {normalize_component_class(item, allow_not_component=False)
                 for item in target.get("ocr_class_hints", [])}
        hints.discard(UNKNOWN_CLASS)
        if hints and proposed not in hints and proposed not in {UNKNOWN_CLASS, "not_component"}:
            proposed = UNKNOWN_CLASS
            decision = "uncertain"
            evidence = (str(row.get("evidence", "")) + " OCR/silkscreen type hint conflicts with visual class.").strip()
        else:
            decision = row["decision"]
            evidence = str(row.get("evidence", ""))
        class_reviews.append({**row, "decision": decision, "proposed_class": proposed,
                              "confidence": float(confidence), "evidence": evidence[:500]})
    duplicate_reviews = []
    seen_pairs = set()
    for row in content.get("duplicate_reviews", []):
        ids = row.get("detection_ids", [])
        confidence = row.get("confidence")
        if (len(ids) != 2 or ids[0] == ids[1] or not set(ids) <= known_ids
                or tuple(sorted(ids)) not in allowed_pairs
                or row.get("decision") not in {"same_component", "different_components", "uncertain"}
                or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence))
                or not 0 <= float(confidence) <= 1):
            continue
        pair = tuple(sorted(ids))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        duplicate_reviews.append({**row, "detection_ids": list(pair), "confidence": float(confidence),
                                  "evidence": str(row.get("evidence", ""))[:500]})
    allowed_groups = {group["parent_id"]: set(group["child_ids"])
                      for group in group_manifest}
    group_reviews = []
    seen_group_ids = set()
    for row in content.get("group_reviews", []):
        parent_id = row.get("parent_id")
        child_ids = row.get("child_ids", [])
        group_confidence = row.get("confidence")
        if (parent_id not in allowed_groups or parent_id in seen_group_ids
                or row.get("decision") not in {"single_component", "multiple_components", "uncertain"}
                or not isinstance(child_ids, list) or len(child_ids) != len(set(child_ids))
                or not set(child_ids) <= allowed_groups[parent_id]
                or not isinstance(group_confidence, (int, float)) or not math.isfinite(float(group_confidence))
                or not 0 <= float(group_confidence) <= 1):
            continue
        raw_new_components = row.get("new_components", [])
        if not isinstance(raw_new_components, list):
            raw_new_components = []
        mapped_new_components = []
        group_image = next((item for item in group_manifest if item["parent_id"] == parent_id), None)
        parent_box = next((item["bbox_xyxy"] for item in identified
                           if item["detection_id"] == parent_id), None)
        for new_index, candidate in enumerate(raw_new_components[:6] if row["decision"] == "multiple_components" else []):
            mapped, rejection_reason = _map_normalized_box(
                candidate.get("bbox_norm_999"), group_image["image_index"] if group_image else None,
                image_manifest,
            )
            class_name = normalize_component_class(candidate.get("proposed_class"))
            candidate_confidence = candidate.get("confidence")
            if class_name == "not_component":
                rejection_reason = "not_component_cannot_be_added_as_box"
            elif (not isinstance(candidate_confidence, (int, float)) or not math.isfinite(float(candidate_confidence))
                  or not 0 <= float(candidate_confidence) <= 1):
                rejection_reason = "invalid_confidence"
            elif mapped and parent_box:
                candidate_area = max(1.0, (mapped[2] - mapped[0]) * (mapped[3] - mapped[1]))
                intersection_width = max(0.0, min(mapped[2], parent_box[2]) - max(mapped[0], parent_box[0]))
                intersection_height = max(0.0, min(mapped[3], parent_box[3]) - max(mapped[1], parent_box[1]))
                if intersection_width * intersection_height / candidate_area < 0.80:
                    rejection_reason = "group_candidate_outside_parent_box"
            if not rejection_reason and mapped and any(
                    same_component_box_pair(mapped, existing["bbox_xyxy"])
                    for existing in identified
                    if isinstance(existing.get("bbox_xyxy"), (list, tuple)) and len(existing["bbox_xyxy"]) == 4):
                rejection_reason = "duplicates_existing_detection"
            if rejection_reason:
                candidate_diagnostics.append({"stage": "qwen_candidate_validation", "parent_id": parent_id,
                                              "candidate_index": new_index, "status": "rejected",
                                              "reason": rejection_reason})
                continue
            mapped_new_components.append({**candidate, "proposed_class": class_name,
                                          "confidence": float(candidate_confidence), "bbox_xyxy": mapped,
                                          "review_status": "검수 필요", "source": "qwen_suggestion"})
            candidate_diagnostics.append({"stage": "qwen_candidate_validation", "parent_id": parent_id,
                                          "candidate_index": new_index, "status": "candidate",
                                          "reason": "validated_group_crop_box"})
        if row["decision"] == "multiple_components" and len(child_ids) + len(mapped_new_components) < 2:
            candidate_diagnostics.append({"stage": "qwen_group_validation", "parent_id": parent_id,
                                          "status": "rejected", "reason": "fewer_than_two_component_boxes"})
            continue
        if row["decision"] != "multiple_components":
            child_ids = []
            mapped_new_components = []
        seen_group_ids.add(parent_id)
        group_reviews.append({**row, "child_ids": child_ids, "confidence": float(group_confidence),
                              "new_components": mapped_new_components,
                              "evidence": str(row.get("evidence", ""))[:500],
                              "review_status": "AI_SUGGESTIONS_ONLY"})
    missed = []
    for row in content.get("missed_component_candidates", [])[:8]:
        proposed = normalize_component_class(row.get("proposed_class"))
        confidence = row.get("confidence")
        image_index = row.get("image_index", 0)
        candidate, rejection_reason = _map_normalized_box(row.get("bbox_norm_999"), image_index, image_manifest)
        if proposed == "not_component":
            rejection_reason = "not_component_cannot_be_added_as_box"
        elif (not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence))
              or not 0 <= float(confidence) <= 1):
            rejection_reason = "invalid_confidence"
        if rejection_reason:
            candidate_diagnostics.append({"stage": "qwen_candidate_validation", "image_index": image_index,
                                          "status": "rejected", "reason": rejection_reason})
            continue
        if any(same_component_box_pair(candidate, row["bbox_xyxy"])
               for row in identified if isinstance(row.get("bbox_xyxy"), (list, tuple)) and len(row["bbox_xyxy"]) == 4):
            candidate_diagnostics.append({"stage": "qwen_candidate_validation", "image_index": image_index,
                                          "status": "rejected", "reason": "duplicates_existing_detection"})
            continue
        possible_duplicate_ids = [det["detection_id"] for det in identified
            if isinstance(det.get("bbox_xyxy"), (list, tuple)) and len(det["bbox_xyxy"]) == 4
            and boxes_overlap_candidate(candidate, det["bbox_xyxy"])
            and not same_component_box_pair(candidate, det["bbox_xyxy"])]
        missed.append({**row, "proposed_class": proposed,
                       "confidence": float(confidence), "bbox_xyxy": candidate, "evidence": str(row.get("evidence", ""))[:500],
                       "possible_duplicate_ids": possible_duplicate_ids,
                       "geometry_review_required": bool(possible_duplicate_ids),
                       "review_status": "검수 필요", "source": "qwen_suggestion"})
        candidate_diagnostics.append({"stage": "qwen_candidate_validation", "image_index": image_index,
                                      "status": "candidate", "reason": "validated_image_box"})
    qwen_diagnostic_items = [item for item in candidate_diagnostics
                             if item.get("stage") in {"qwen_candidate_validation", "qwen_group_validation"}]
    return {"class_reviews": class_reviews, "duplicate_reviews": duplicate_reviews,
            "group_reviews": group_reviews,
            "missed_component_candidates": missed,
            "notes": [str(note)[:300] for note in content.get("notes", [])[:10]],
            "model": response.get("model", model), "image_size_px": [width, height],
            "review_image_count": len(images), "review_tiles": sum(item["kind"] == "native_resolution_tile" for item in image_manifest),
            "duplicate_pair_crop_count": len(pair_manifest),
            "review_diagnostics": {
                "candidate_search": {"status": "complete", "candidate_count": len(group_candidates),
                                      "items": [item for item in candidate_diagnostics
                                                if item.get("stage") == "candidate_search"][:160]},
                "duplicate_pair_search": {"status": "complete", "pair_count": len(pair_manifest),
                                          "items": [item for item in candidate_diagnostics
                                                    if item.get("stage") == "duplicate_pair_search"][:160]},
                "qwen": {"status": "complete", "group_response_count": len(content.get("group_reviews", [])),
                         "accepted_group_reviews": len(group_reviews), "items": qwen_diagnostic_items[:160]},
                "ui": {"status": "pending", "events": []},
            },
            "verification_status": "AI_SUGGESTIONS_ONLY"}
