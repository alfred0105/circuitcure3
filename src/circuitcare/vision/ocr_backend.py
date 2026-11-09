"""Optional local PP-OCR text recognition and conservative KiCad reference extraction."""

from __future__ import annotations

import re
from functools import lru_cache
from decimal import Decimal, InvalidOperation
from typing import Any


class OCRUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_engine():
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise OCRUnavailable("RapidOCR가 없습니다. `python -m pip install -e '.[vision-ocr]'`를 실행하세요.") from exc
    return RapidOCR()


def extract_reference_candidates(text: str) -> list[str]:
    """Extract literal reference-shaped tokens without OCR character correction."""
    return list(dict.fromkeys(match.group(0).upper() for match in re.finditer(
        r"(?<![A-Z0-9])(?:R|C|U|D|Q|J|P|L|F|T|Y|CN|TP)\s*[-_]?\s*\d{1,4}[A-Z]?(?![A-Z0-9])",
        str(text), flags=re.IGNORECASE,
    )))


def run_local_ocr(image: Any) -> dict[str, Any]:
    """Recognize text locally with RapidOCR and retain raw text plus candidate refs."""
    try:
        import numpy as np
        engine = _load_engine()
        result = engine(np.asarray(image.convert("RGB")))
    except OCRUnavailable:
        raise
    except Exception as exc:
        raise OCRUnavailable(f"로컬 OCR 추론이 실패했습니다: {exc}") from exc
    records = []
    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if texts is None:
        texts = ()
    try:
        text_count = len(texts)
    except TypeError as exc:
        raise OCRUnavailable("OCR 엔진의 텍스트 결과 형식이 올바르지 않습니다.") from exc
    score_count = len(scores) if scores is not None else 0
    for index in range(text_count):
        raw_text = texts[index]
        text = str(raw_text).strip()
        if not text:
            continue
        polygon = np.asarray(boxes[index], dtype=float).reshape(-1, 2).tolist() if boxes is not None else []
        if polygon:
            x_values = [point[0] for point in polygon]
            y_values = [point[1] for point in polygon]
            bbox = [float(min(x_values)), float(min(y_values)), float(max(x_values)), float(max(y_values))]
        else:
            bbox = None
        score = float(scores[index]) if scores is not None and index < score_count else 0.0
        records.append({"text": text, "raw_text": text, "confidence": score,
                        "bbox_xyxy": bbox, "polygon_xy": polygon, "source": "local_ocr",
                        "verification_status": "unverified", "reference_candidates": extract_reference_candidates(text)})
    return {"status": "OCR_COMPLETE", "engine": "RapidOCR / PP-OCR ONNX", "runtime": "local-onnxruntime",
            "line_count": len(records), "records": records}


def _format_resistance(ohms: Decimal) -> str:
    if ohms == 0:
        return "0 Ω"
    if ohms >= Decimal("1000000"):
        return f"{_decimal_text(ohms / Decimal('1000000'))} MΩ"
    if ohms >= Decimal("1000"):
        return f"{_decimal_text(ohms / Decimal('1000'))} kΩ"
    return f"{_decimal_text(ohms)} Ω"


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def infer_passive_value(component_type: str, marking: str) -> dict[str, str] | None:
    """Interpret only common, unambiguous resistor/capacitor top-mark codes."""
    kind = str(component_type or "").strip().casefold()
    text = str(marking or "").upper().replace("µ", "U").strip()
    if kind in {"resistor", "저항"}:
        r_code = re.search(r"(?<![A-Z0-9])(\d{0,3})R(\d{0,3})(?![A-Z0-9])", text)
        if r_code and (r_code.group(1) or r_code.group(2)):
            whole = r_code.group(1) or "0"
            fraction = r_code.group(2)
            try:
                value = Decimal(f"{whole}.{fraction}") if fraction else Decimal(whole)
            except InvalidOperation:
                value = None
            if value is not None:
                return {"value": _format_resistance(value), "method": "R-notation", "status": "OCR_VALUE_CANDIDATE"}
        code = re.search(r"(?<![A-Z0-9])(\d{3,4})(?![A-Z0-9])", text)
        if code:
            digits = code.group(1)
            significand = int(digits[:-1])
            exponent = int(digits[-1])
            if exponent <= 9:
                value = Decimal(significand) * (Decimal(10) ** exponent)
                return {"value": _format_resistance(value), "method": f"3/4-digit SMD code {digits}", "status": "OCR_VALUE_CANDIDATE"}
    if kind in {"capacitor", "variable_capacitor", "커패시터", "콘덴서"}:
        explicit = re.search(r"(?<![A-Z0-9])(\d+(?:\.\d+)?)\s*(UF|NF|PF)(?![A-Z])", text)
        if explicit:
            value = explicit.group(1)
            unit = {"UF": "µF", "NF": "nF", "PF": "pF"}[explicit.group(2)]
            voltage = re.search(r"(?<![A-Z0-9])(\d+(?:\.\d+)?)\s*V(?![A-Z])", text)
            result = {"value": f"{value} {unit}", "method": "printed capacitance", "status": "OCR_VALUE_CANDIDATE"}
            if voltage:
                result["value"] += f" · {voltage.group(1)} V"
            return result
        code = re.search(r"(?<![A-Z0-9])(\d{3})(?![A-Z0-9])", text)
        if code:
            digits = code.group(1)
            significand, exponent = int(digits[:2]), int(digits[2])
            if significand > 0 and exponent <= 8:
                picofarads = Decimal(significand) * (Decimal(10) ** exponent)
                if picofarads >= Decimal("1000000"):
                    value, unit = picofarads / Decimal("1000000"), "µF"
                elif picofarads >= Decimal("1000"):
                    value, unit = picofarads / Decimal("1000"), "nF"
                else:
                    value, unit = picofarads, "pF"
                return {"value": f"{_decimal_text(value)} {unit}", "method": f"3-digit EIA code {digits}", "status": "OCR_VALUE_CANDIDATE"}
    return None


def run_component_marking_ocr(image: Any, bbox_xyxy: list[float] | tuple[float, ...],
                              component_type: str) -> dict[str, Any]:
    """Upscale and rotate a selected component crop, then OCR its top marking.

    Results are candidates only. Passive values are inferred from common printed
    codes; short semiconductor top-marks are retained as text without guessing a
    part number.
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image

        source = image.convert("RGB") if hasattr(image, "convert") else Image.fromarray(np.asarray(image)).convert("RGB")
        if len(bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain x1, y1, x2, y2.")
        width, height = source.size
        x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
        left, right = max(0, int(min(x1, x2))), min(width, int(max(x1, x2)))
        top, bottom = max(0, int(min(y1, y2))), min(height, int(max(y1, y2)))
        if right <= left or bottom <= top:
            raise ValueError("Selected component box has no image area.")
        pad_x = max(2, int((right - left) * 0.10))
        pad_y = max(2, int((bottom - top) * 0.10))
        crop = source.crop((max(0, left - pad_x), max(0, top - pad_y),
                            min(width, right + pad_x), min(height, bottom + pad_y)))
        rgb = np.asarray(crop)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
        scale = max(2, min(6, 1200 // max(gray.shape)))
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        engine = _load_engine()
        candidates: dict[str, dict[str, Any]] = {}
        for rotation in (0, 180, 90, 270):
            oriented = np.rot90(enlarged, k=rotation // 90).copy() if rotation else enlarged
            rgb_input = cv2.cvtColor(oriented, cv2.COLOR_GRAY2RGB)
            result = engine(rgb_input)
            texts = getattr(result, "txts", None)
            scores = getattr(result, "scores", None)
            if texts is None:
                continue
            for index, raw in enumerate(texts):
                text = str(raw).strip()
                if not text:
                    continue
                score = float(scores[index]) if scores is not None and index < len(scores) else 0.0
                key = re.sub(r"\s+", "", text).upper()
                prior = candidates.get(key)
                if prior is None or score > prior["ocr_score"]:
                    candidates[key] = {"text": text, "ocr_score": score, "rotation_degrees": rotation}
            if rotation == 180 and candidates:
                break
    except OCRUnavailable:
        raise
    except ImportError as exc:
        raise OCRUnavailable("각인 OCR에는 OpenCV, Pillow, NumPy가 필요합니다.") from exc
    except Exception as exc:
        raise OCRUnavailable(f"선택 부품 각인 OCR에 실패했습니다: {exc}") from exc

    ordered = sorted(candidates.values(), key=lambda row: row["ocr_score"], reverse=True)
    marking = ordered[0]["text"] if ordered else None
    inferred = infer_passive_value(component_type, marking or "") if marking else None
    possible_part = marking if str(component_type).casefold() in {"ic", "integrated_circuit"} and marking else None
    return {
        "status": "OCR_COMPLETE" if ordered else "OCR_NO_TEXT",
        "engine": "RapidOCR · CLAHE · 2–6× upscale · 4 rotations",
        "marking": marking,
        "candidates": ordered[:5],
        "inferred_value": inferred["value"] if inferred else None,
        "inference_method": inferred["method"] if inferred else None,
        "identification_status": inferred["status"] if inferred else ("OCR_TEXT_CANDIDATE" if marking else "NO_MARKING_READ"),
        "possible_part_number": possible_part,
        "verification_status": "unverified",
    }
