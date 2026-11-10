"""Human-in-the-loop visual review of YOLO component proposals using Ollama."""

from __future__ import annotations

import io
import json
import math
from typing import Any

from PIL import Image, ImageDraw

from circuitcare.local_llm import OllamaClient
from circuitcare.vision.box_reconciliation import (
    boxes_overlap_candidate, duplicate_box_metrics, same_component_box_pair,
)

DEFAULT_REVIEW_MODEL = "qwen3.5:9b"
REVIEW_CLASSES = (
    "unknown", "resistor", "inductor", "capacitor", "connector", "jumper", "IC",
    "switch", "relay", "transistor", "potentiometer", "battery", "buzzer", "diode",
    "display", "emi_filter", "ferrite_bead", "fuse", "LED", "clock", "oscillator",
    "voltage_regulator", "MOV", "transformer", "heatsink",
)

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
        "missed_component_candidates": {"type": "array", "items": {"type": "object", "properties": {
            "image_index": {"type": "integer", "minimum": 0, "maximum": 48},
            "bbox_norm_999": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 999}, "minItems": 4, "maxItems": 4},
            "proposed_class": {"type": "string", "enum": list(REVIEW_CLASSES)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "string"},
        }, "required": ["image_index", "bbox_norm_999", "proposed_class", "confidence", "evidence"], "additionalProperties": False}},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["class_reviews", "duplicate_reviews", "missed_component_candidates", "notes"],
    "additionalProperties": False,
}

_SYSTEM = """You review a PCB photo and its YOLO component proposals. Be conservative: only suggest a class when the physical component appearance is clear. Small SMD parts are often visually ambiguous. Use OCR strings only as nearby text evidence; a reference such as C12 may not label the closest body. Do not claim certainty. Do not invent unseen parts. Existing boxes are identified by IDs; assess duplicate pairs only when their IDs are listed in duplicate_pair_crops, using the associated paired crop and surrounding board context. Search for at most a few clearly visible missed components, including when there are no detector proposals. Every proposed box is only a rough review candidate and must be manually checked. Each supplied image has an image_index from image_sources; missed boxes must be normalized relative to the image they cite. Return only the requested JSON object."""


def _review_images(
    image_bytes: bytes, detections: list[dict[str, Any]], *, max_crop_count: int = 10,
) -> tuple[list[bytes], tuple[int, int], list[dict[str, Any]], list[dict[str, Any]]]:
    with Image.open(io.BytesIO(image_bytes)) as source_image:
        image = source_image.convert("RGB")
    width, height = image.size
    guide = image.copy()
    guide.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    sx, sy = guide.width / width, guide.height / height
    painter = ImageDraw.Draw(guide)
    ordered = sorted(detections, key=lambda row: float(row.get("confidence") or 0.0))[:max_crop_count]
    pair_candidates = _duplicate_pair_candidates(detections)
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
    encoded = []
    for candidate in images:
        buffer = io.BytesIO()
        candidate.save(buffer, format="JPEG", quality=88)
        encoded.append(buffer.getvalue())
    return encoded, (width, height), manifest, pair_manifest


def _duplicate_pair_candidates(detections: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
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
            if metrics["area_ratio"] < .30 or metrics["center_distance_norm"] > .60:
                continue
            if metrics["iou"] < .05 and metrics["smaller_coverage"] < .20:
                continue
            pairs.append({"detections": (left, right), "metrics": metrics,
                          "rank": metrics["iou"] + .35 * metrics["smaller_coverage"]
                          + max(0.0, .60 - metrics["center_distance_norm"]) * .15})
    return sorted(pairs, key=lambda item: item["rank"], reverse=True)[:limit]


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
                           "class_name": row.get("class_name", row.get("소자 종류", "unknown")),
                           "confidence": row.get("confidence", row.get("신뢰도")),
                           "bbox_xyxy": box,
                           "ocr_text": row.get("ocr_candidates", [])})
    images, (width, height), image_manifest, pair_manifest = _review_images(image_bytes, identified)
    payload = {
        "image_coordinates_px": {"width": width, "height": height},
        "annotated_image_box_ids": [row["detection_id"] for row in identified],
        "image_sources": image_manifest,
        "duplicate_pair_crops": pair_manifest,
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
    response = client.chat_json(
        model, [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": "Review the PCB image, YOLO proposals, and OCR evidence. Return conservative candidate findings.\n" + json.dumps(payload, ensure_ascii=False)}],
        output_schema=REVIEW_SCHEMA, images=images,
    )
    content = response["content"]
    known_ids = {row["detection_id"] for row in identified}
    allowed_pairs = {tuple(sorted(pair["detection_ids"])) for pair in pair_manifest}
    class_reviews = []
    for row in content.get("class_reviews", []):
        confidence = row.get("confidence")
        if (row.get("detection_id") not in known_ids or row.get("proposed_class") not in REVIEW_CLASSES
                or row.get("decision") not in {"suggest", "uncertain"}
                or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence))
                or not 0 <= float(confidence) <= 1):
            continue
        class_reviews.append({**row, "confidence": float(confidence), "evidence": str(row.get("evidence", ""))[:500]})
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
    missed = []
    for row in content.get("missed_component_candidates", [])[:8]:
        box = row.get("bbox_norm_999", [])
        confidence = row.get("confidence")
        image_index = row.get("image_index", 0)
        if (type(image_index) is not int or not 0 <= image_index < len(image_manifest)
                or len(box) != 4 or any(type(value) is not int or not 0 <= value <= 999 for value in box)
                or row.get("proposed_class") not in REVIEW_CLASSES
                or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence))
                or not 0 <= float(confidence) <= 1):
            continue
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        region = image_manifest[image_index]["bbox_xyxy"]
        region_width, region_height = region[2] - region[0], region[3] - region[1]
        candidate = [region[0] + x1 * region_width / 999, region[1] + y1 * region_height / 999,
                     region[0] + x2 * region_width / 999, region[1] + y2 * region_height / 999]
        if candidate[2] - candidate[0] < 4 or candidate[3] - candidate[1] < 4:
            continue
        if any(same_component_box_pair(candidate, row["bbox_xyxy"])
               for row in identified if isinstance(row.get("bbox_xyxy"), (list, tuple)) and len(row["bbox_xyxy"]) == 4):
            continue
        possible_duplicate_ids = [det["detection_id"] for det in identified
            if isinstance(det.get("bbox_xyxy"), (list, tuple)) and len(det["bbox_xyxy"]) == 4
            and boxes_overlap_candidate(candidate, det["bbox_xyxy"])
            and not same_component_box_pair(candidate, det["bbox_xyxy"])]
        missed.append({**row, "confidence": float(confidence), "bbox_xyxy": candidate, "evidence": str(row.get("evidence", ""))[:500],
                       "possible_duplicate_ids": possible_duplicate_ids,
                       "geometry_review_required": bool(possible_duplicate_ids),
                       "review_status": "검수 필요", "source": "qwen_suggestion"})
    return {"class_reviews": class_reviews, "duplicate_reviews": duplicate_reviews,
            "missed_component_candidates": missed,
            "notes": [str(note)[:300] for note in content.get("notes", [])[:10]],
            "model": response.get("model", model), "image_size_px": [width, height],
            "review_image_count": len(images), "review_tiles": sum(item["kind"] == "native_resolution_tile" for item in image_manifest),
            "duplicate_pair_crop_count": len(pair_manifest),
            "verification_status": "AI_SUGGESTIONS_ONLY"}
