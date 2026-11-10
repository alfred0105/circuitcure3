"""Human-in-the-loop visual review of YOLO component proposals using Ollama."""

from __future__ import annotations

import io
import json
import math
from typing import Any

from PIL import Image, ImageDraw

from circuitcare.local_llm import OllamaClient
from circuitcare.vision.box_reconciliation import box_iou

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
            "bbox_norm_999": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 999}, "minItems": 4, "maxItems": 4},
            "proposed_class": {"type": "string", "enum": list(REVIEW_CLASSES)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "string"},
        }, "required": ["bbox_norm_999", "proposed_class", "confidence", "evidence"], "additionalProperties": False}},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["class_reviews", "duplicate_reviews", "missed_component_candidates", "notes"],
    "additionalProperties": False,
}

_SYSTEM = """You review a PCB photo and its YOLO component proposals. Be conservative: only suggest a class when the physical component appearance is clear. Small SMD parts are often visually ambiguous. Use OCR strings only as nearby text evidence; a reference such as C12 may not label the closest body. Do not claim certainty. Do not invent unseen parts. Existing boxes are identified by IDs; propose duplicates only when they cover the same physical body. Search for at most a few clearly visible missed components. Every proposed box is only a rough review candidate and must be manually checked. Return only the requested JSON object."""


def _review_images(
    image_bytes: bytes, detections: list[dict[str, Any]], *, max_crop_count: int = 10,
) -> tuple[list[bytes], tuple[int, int]]:
    with Image.open(io.BytesIO(image_bytes)) as source_image:
        image = source_image.convert("RGB")
    width, height = image.size
    guide = image.copy()
    guide.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    sx, sy = guide.width / width, guide.height / height
    painter = ImageDraw.Draw(guide)
    ordered = sorted(detections, key=lambda row: float(row.get("confidence") or 0.0))[:max_crop_count]
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
    encoded = []
    for candidate in images:
        buffer = io.BytesIO()
        candidate.save(buffer, format="JPEG", quality=88)
        encoded.append(buffer.getvalue())
    return encoded, (width, height)


def review_components_with_qwen(
    client: OllamaClient,
    model: str,
    image_bytes: bytes,
    detections: list[dict[str, Any]],
    ocr_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Request suggestions and validate every ID and normalized box before display."""
    if not detections:
        return {"class_reviews": [], "duplicate_reviews": [], "missed_component_candidates": [], "notes": [], "model": model}
    identified = []
    for index, row in enumerate(detections, start=1):
        box = row.get("bbox_xyxy", [row.get("x1"), row.get("y1"), row.get("x2"), row.get("y2")])
        identified.append({"detection_id": str(row.get("component_id") or f"det-{index}"),
                           "class_name": row.get("class_name", row.get("소자 종류", "unknown")),
                           "confidence": row.get("confidence", row.get("신뢰도")),
                           "bbox_xyxy": box,
                           "ocr_text": row.get("ocr_candidates", [])})
    images, (width, height) = _review_images(image_bytes, identified)
    image_legend = [identified[0]["detection_id"]] if len(images) <= 1 else [
        identified[0]["detection_id"], *[row["detection_id"] for row in sorted(
            identified, key=lambda item: float(item.get("confidence") or 0.0))[:len(images) - 1]]
    ]
    payload = {
        "image_coordinates_px": {"width": width, "height": height},
        "annotated_image_box_ids": [row["detection_id"] for row in identified],
        "crop_image_order_after_annotated_image": image_legend[1:],
        "detector_proposals": [{key: value for key, value in row.items() if key != "bbox_xyxy"}
                               | {"bbox_xyxy": [round(float(value), 1) for value in row["bbox_xyxy"]]}
                               for row in identified],
        "nearby_ocr_text_candidates": [{"text": row.get("raw_text", row.get("text", "")),
                                        "bbox_xyxy": row.get("bbox_xyxy"),
                                        "confidence": row.get("confidence"),
                                        "reference_candidates": row.get("reference_candidates", [])}
                                       for row in ocr_records[:120]],
        "coordinate_rule": "Return missed bbox_norm_999 as [x1,y1,x2,y2] in the full annotated image, each integer from 0 to 999.",
    }
    response = client.chat_json(
        model, [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": "Review the PCB image, YOLO proposals, and OCR evidence. Return conservative candidate findings.\n" + json.dumps(payload, ensure_ascii=False)}],
        output_schema=REVIEW_SCHEMA, images=images,
    )
    content = response["content"]
    known_ids = {row["detection_id"] for row in identified}
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
        if (len(box) != 4 or any(type(value) is not int or not 0 <= value <= 999 for value in box)
                or row.get("proposed_class") not in REVIEW_CLASSES
                or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence))
                or not 0 <= float(confidence) <= 1):
            continue
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        candidate = [x1 * width / 999, y1 * height / 999, x2 * width / 999, y2 * height / 999]
        if candidate[2] - candidate[0] < 4 or candidate[3] - candidate[1] < 4:
            continue
        if any(box_iou(candidate, row["bbox_xyxy"]) >= .30
               for row in identified if isinstance(row.get("bbox_xyxy"), (list, tuple)) and len(row["bbox_xyxy"]) == 4):
            continue
        missed.append({**row, "confidence": float(confidence), "bbox_xyxy": candidate, "evidence": str(row.get("evidence", ""))[:500],
                       "review_status": "검수 필요", "source": "qwen_suggestion"})
    return {"class_reviews": class_reviews, "duplicate_reviews": duplicate_reviews,
            "missed_component_candidates": missed,
            "notes": [str(note)[:300] for note in content.get("notes", [])[:10]],
            "model": response.get("model", model), "image_size_px": [width, height],
            "verification_status": "AI_SUGGESTIONS_ONLY"}
