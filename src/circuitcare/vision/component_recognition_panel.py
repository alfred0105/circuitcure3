"""Component detection and manual box/class review for the PCB Workbench."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
import importlib.util
import io
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from circuitcare.vision.weight_paths import resolve_pcb_weights_root
from circuitcare.vision.box_reconciliation import (
    associate_ocr_with_components, boxes_overlap_candidate, preserve_review_rows,
    exclude_review_row, remove_unreviewed_legacy_secondary_rows, same_component_box_pair,
)
from circuitcare.component_taxonomy import (
    COMPONENT_LABELS, normalize_component_class, normalize_qwen_class, normalize_qwen_subtype,
)
from circuitcare.vision.ai_review_service import AIReviewStageError
from circuitcare.frontend.measurement_state import invalidate_legacy_diagnosis
from circuitcare.analysis.models import AnalysisScope
from circuitcare.analysis.state_manager import invalidate_analysis
from circuitcare.frontend.workbench_state import ACTIVE_PHOTO_KEY

APP_DIR = Path(__file__).resolve().parents[3] / "apps" / "pcb-component-recognition"
MODULE_NAME = "circuitcare_pcb_component_app_server"
CLASS_OPTIONS = list(COMPONENT_LABELS)
STATUS_OPTIONS = ["검수 필요", "확인", "제외"]
LAST_EXCLUDED_KEY = "circuitcare_v4_last_excluded_box"
INFERENCE_CONFIG = {"model_pipeline": "yolo26s+doctor-v12n+arshia-yolov8s", "conf": 0.20,
                    "tile": 1024, "overlap": 0.25, "postprocess": "one-to-one-v2",
                    "ocr_engine": "rapidocr-v1", "review_model": "qwen3.5:9b",
                    "review_schema": "qwen-hierarchical-review-v2"}


def _model_cache_fingerprint() -> str:
    root = resolve_pcb_weights_root(APP_DIR)
    specs = ((root / "any_best.pt"),
             (root / "external" / "pcb-ai-doctor" / "best_v12n.pt"),
             (root / "external" / "pcb-yolov8s-detection" / "best.pt"))
    parts = []
    for path in specs:
        try:
            info = path.stat()
            parts.append(f"{path.name}:{info.st_size}:{info.st_mtime_ns}")
        except OSError:
            parts.append(f"{path.name}:missing")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def review_scope(image_bytes: bytes, project_id: str, photo_id: str) -> str:
    # Version the review cache because detector input now keeps its source resolution.
    versioned_options = {**INFERENCE_CONFIG, "weight_fingerprint": _model_cache_fingerprint()}
    digest = hashlib.sha256(image_bytes + json.dumps(versioned_options, sort_keys=True).encode()).hexdigest()[:16]
    return f"{project_id}_{photo_id}_native_{digest}"


def _legacy_secondary_scope(image_bytes: bytes, project_id: str, photo_id: str) -> str:
    """Locate session rows written before the optional second pass was removed."""
    options = {**INFERENCE_CONFIG, "secondary_conf": 0.08, "secondary_tile": 640,
               "secondary_overlap": 0.25, "weight_fingerprint": _model_cache_fingerprint()}
    digest = hashlib.sha256(image_bytes + json.dumps(options, sort_keys=True).encode()).hexdigest()[:16]
    return f"{project_id}_{photo_id}_native_{digest}"


def _migrate_legacy_review_state(state: Any, image_bytes: bytes, project_id: str,
                                 photo_id: str, scope: str) -> None:
    old_scope = _legacy_secondary_scope(image_bytes, project_id, photo_id)
    if old_scope == scope:
        return
    old_keys, new_keys = review_keys(old_scope), review_keys(scope)
    for index, (old_key, new_key) in enumerate(zip(old_keys, new_keys)):
        if old_key in state and new_key not in state:
            state[new_key] = state[old_key]
        elif old_key in state and index == 1:
            previous = state[old_key] if isinstance(state[old_key], list) else []
            current = state.get(new_key) if isinstance(state.get(new_key), list) else []
            by_id = {str(row.get("component_id")): row for row in current if isinstance(row, dict)}
            for row in previous:
                if not isinstance(row, dict):
                    continue
                component_id = str(row.get("component_id") or "")
                current_row = by_id.get(component_id)
                if not current_row:
                    current.append(row)
                    if component_id:
                        by_id[component_id] = row
                elif (row.get("user_edited") or row.get("상태") in {"확인", "제외"}
                      or row.get("source") in {"manual", "qwen_suggestion", "yolo_second_pass"}):
                    current[current.index(current_row)] = row
                    by_id[component_id] = row
            state[new_key] = current
        state.pop(old_key, None)
    old_ocr, old_qwen = review_aux_keys(old_scope)
    new_ocr, _new_qwen = review_aux_keys(scope)
    if old_ocr in state and new_ocr not in state:
        state[new_ocr] = state[old_ocr]
    state.pop(old_ocr, None)
    # Old proposals were generated against a different workflow; keep user rows,
    # but force Qwen to produce a fresh review when requested.
    state.pop(old_qwen, None)


def review_keys(scope: str) -> tuple[str, str, str]:
    return (f"cc_component_recognition_{scope}", f"cc_component_review_rows_{scope}",
            f"cc_component_recognition_error_{scope}")


def review_aux_keys(scope: str) -> tuple[str, str]:
    return f"cc_component_ocr_{scope}", f"cc_component_qwen_review_{scope}"


def _review_detection_batches(detections: list[dict], batch_size: int = 24) -> list[list[dict]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return [detections[index:index + batch_size]
            for index in range(0, len(detections), batch_size)] or [[]]


def _marking_ocr_targets(rows: list[dict], limit: int = 12) -> list[dict]:
    eligible = [row for row in rows if row.get("상태") != "제외"
                and row.get("소자 종류") in {"ic", "resistor", "transistor", "diode", "connector"}]
    eligible.sort(key=lambda row: (
        row.get("소자 종류") not in {"ic", "transistor", "diode"},
        row.get("상태") == "확인",
        float(row.get("type_confidence") or row.get("신뢰도") or 0.0),
    ))
    return eligible[:max(0, limit)]


def _write_analysis_status(progress: Any, message: str) -> None:
    if progress is not None:
        progress.write(message)


@st.cache_resource(show_spinner=False)
def _ollama_review_client() -> Any:
    """Reuse the local Ollama transport across reruns and review batches."""
    from circuitcare.local_llm import OllamaClient
    return OllamaClient(timeout_seconds=300, num_predict=4096)


def invalidate_qwen_review(state: Any, scope: str) -> None:
    """Drop suggestions whenever their detection/OCR evidence changes."""
    _ocr_key, qwen_key = review_aux_keys(scope)
    state.pop(qwen_key, None)


def invalidate_component_review_state(state: Any, *, scope: str, project_id: str, board_id: str | None) -> None:
    """Invalidate diagnosis output; AI caches self-invalidate from actual inference inputs."""
    invalidate_legacy_diagnosis(state, project_id=project_id, board_id=board_id or "")
    invalidate_analysis(
        state,
        scope=AnalysisScope(project_id, board_id or "",
                            str(state.get(ACTIVE_PHOTO_KEY) or "target:default")),
        reason="소자 검수 정보가 변경되었습니다.",
    )
    diagnosis_key = f"cc_symptom_diagnosis_{project_id}"
    diagnosis = state.get(diagnosis_key)
    if isinstance(diagnosis, dict):
        if diagnosis.get("result") is not None:
            history = list(diagnosis.get("history", []))
            history.append({"result": diagnosis["result"],
                            "reason": "소자 검수 정보가 변경되었습니다."})
            diagnosis["history"] = history[-100:]
        diagnosis["result"] = None
        diagnosis.pop("input_fingerprint", None)
        state[diagnosis_key] = diagnosis


def ensure_review_row_for_component(rows: list[dict], component: Any) -> dict:
    """Return the canonical review row for a displayed component, creating it if needed."""
    component_id = str(component.component_id)
    row = next((item for item in rows if str(item.get("component_id")) == component_id), None)
    if row is None:
        x1, y1, x2, y2 = component.bbox_xyxy
        row = _record(component.component_type, [x1, y1, x2, y2],
                      status="확인" if component.presence_status == "CONFIRMED" else "검수 필요",
                      confidence=component.detection_confidence or 0.0,
                      component_id=component_id,
                      location_confidence=component.detection_confidence,
                      type_confidence=component.type_confidence)
        row["source"] = component.overlay_source or "component_map"
        rows.append(row)
    return row


def update_review_row(row: dict, *, component_type: str, status: str,
                      bbox_xyxy: list[int] | tuple[int, int, int, int]) -> None:
    if component_type not in CLASS_OPTIONS or status not in STATUS_OPTIONS:
        raise ValueError("소자 종류 또는 검수 상태가 유효하지 않습니다.")
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    if min(x1, y1) < 0 or x2 - x1 < 4 or y2 - y1 < 4:
        raise ValueError("박스 좌표가 유효하지 않습니다.")
    row.update({"소자 종류": component_type, "상태": status,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "user_edited": True})


def restore_review_row(row: dict) -> None:
    row["상태"] = row.pop("status_before_exclusion", "검수 필요")
    row.pop("exclusion_reason", None)
    row.pop("duplicate_of", None)
    row["user_edited"] = True


def record_box_exclusion(
    state: Any, *, scope: str, row: dict, reason: str = "user_box_exclusion",
    review: dict | None = None, duplicate_of: str | None = None,
) -> None:
    """Snapshot a review row before hiding it so the immediately preceding action can be undone."""
    before = deepcopy(row)
    row["status_before_exclusion"] = row.get("상태", "검수 필요")
    exclude_review_row(row, reason=reason, review=review)
    if duplicate_of:
        row["duplicate_of"] = str(duplicate_of)
    state[f"{LAST_EXCLUDED_KEY}_{scope}"] = {"component_id": row.get("component_id"), "before": before}


def undo_last_box_exclusion(state: Any, *, scope: str, rows: list[dict]) -> str | None:
    """Restore the exact row snapshot from the latest user exclusion, if present."""
    snapshot = state.get(f"{LAST_EXCLUDED_KEY}_{scope}")
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("before"), dict):
        return None
    component_id = snapshot.get("component_id")
    row = next((item for item in rows if item.get("component_id") == component_id), None)
    if row is None:
        return None
    row.clear()
    row.update(deepcopy(snapshot["before"]))
    state.pop(f"{LAST_EXCLUDED_KEY}_{scope}", None)
    return str(component_id)


def _review_fingerprint(rows: list[dict], ocr_result: dict | None) -> str:
    snapshot = [{key: row.get(key) for key in (
        "component_id", "소자 종류", "x1", "y1", "x2", "y2", "신뢰도",
        "ocr_candidates", "reference_designators", "ocr_class_hints", "body_marking",
        "marking_ocr_fingerprint", "possible_part_number",
    )} for row in rows]
    payload = json.dumps({"rows": snapshot, "ocr": (ocr_result or {}).get("records", [])},
                         ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def duplicate_exclusion_enabled(proposal: dict, left_box: list | tuple,
                                right_box: list | tuple, *, protected: bool = False) -> bool:
    """Require Qwen agreement and strong geometric evidence before exclusion."""
    if protected:
        return False
    try:
        confidence = float(proposal.get("confidence", 0))
    except (TypeError, ValueError):
        return False
    return (proposal.get("decision") == "same_component" and confidence >= 0.65
            and same_component_box_pair(left_box, right_box))


@st.cache_resource(show_spinner=False)
def _app_server() -> Any:
    if not APP_DIR.is_dir():
        raise FileNotFoundError(f"PCB recognition app missing: {APP_DIR}")
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    cached = sys.modules.get(MODULE_NAME)
    if cached is not None:
        return cached
    source = APP_DIR / "app_server.py"
    spec = importlib.util.spec_from_file_location(MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load PCB recognition module: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _record(label: str, bbox: list | tuple, status: str = "검수 필요", confidence: float = 0.0,
            component_id: str | None = None, **scores: Any) -> dict:
    return {"component_id": component_id or f"manual-{hashlib.sha256(repr(tuple(round(float(v),1) for v in bbox)).encode()).hexdigest()[:16]}",
            "소자 종류": normalize_component_class(label),
            "상태": status if status in STATUS_OPTIONS else "검수 필요",
            "신뢰도": float(confidence or 0.0),
            "location_confidence": scores.get("location_confidence", confidence),
            "type_confidence": scores.get("type_confidence"), "match_score": scores.get("match_score"),
            "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}


def _store_predictions(state: Any, records_key: str, result: dict) -> None:
    old_rows = state.get(records_key, [])
    detections = [row for row in result.get("detections", []) if isinstance(row, dict)]
    rows = preserve_review_rows(old_rows, detections)
    for row in rows:
        if not row.get("component_id"):
            row["component_id"] = _record(
                row.get("소자 종류", "unknown"),
                [row.get("x1", 0), row.get("y1", 0), row.get("x2", 0), row.get("y2", 0)],
                status=row.get("상태", "검수 필요"), confidence=row.get("신뢰도", 0.0),
            )["component_id"]
    state[records_key] = rows


def ensure_component_detection(st_api: Any, *, image_bytes: bytes, project_id: str,
                               photo_id: str, review_image_bytes: bytes | None = None,
                               review_image_size: tuple[int, int] | None = None,
                               progress: Any = None) -> tuple[dict | None, str | None]:
    """Automatically detect a newly selected photo once per Streamlit session."""
    review_bytes = review_image_bytes or image_bytes
    scope = review_scope(review_bytes, project_id, photo_id)
    _migrate_legacy_review_state(st_api.session_state, review_bytes, project_id, photo_id, scope)
    result_key, records_key, error_key = review_keys(scope)
    old_rows = st_api.session_state.get(records_key, [])
    cleaned_rows = remove_unreviewed_legacy_secondary_rows(old_rows)
    if cleaned_rows != old_rows:
        st_api.session_state[records_key] = cleaned_rows
    if result_key in st_api.session_state:
        _write_analysis_status(progress, "YOLO 소자 탐지 · 재사용")
        return st_api.session_state[result_key], None
    if error_key in st_api.session_state:
        _write_analysis_status(progress, "YOLO 소자 탐지 · 이전 실행 실패")
        return None, st_api.session_state[error_key]
    try:
        _write_analysis_status(progress, "YOLO 소자 탐지 · 진행 중")
        with st_api.spinner("선택한 PCB 사진에서 소자를 자동 탐지하고 있습니다…"):
            engine = _app_server()
            engine.load_component_locator()
            result = engine.run_prediction(image_bytes, {"conf": INFERENCE_CONFIG["conf"],
                "tile": INFERENCE_CONFIG["tile"], "overlap": INFERENCE_CONFIG["overlap"]})
        if review_image_size and result.get("detections"):
            source_width = max(1, int(result.get("image_width", review_image_size[0])))
            source_height = max(1, int(result.get("image_height", review_image_size[1])))
            scale_x = review_image_size[0] / source_width
            scale_y = review_image_size[1] / source_height
            for row in result["detections"]:
                x1, y1, x2, y2 = row["bbox_xyxy"]
                row["bbox_xyxy"] = [x1 * scale_x, y1 * scale_y,
                                    x2 * scale_x, y2 * scale_y]
        st_api.session_state[result_key] = result
        _store_predictions(st_api.session_state, records_key, result)
        _write_analysis_status(progress, f"YOLO 소자 탐지 · 완료 · {len(result.get('detections', []))}개")
        return result, None
    except Exception as exc:
        st_api.session_state[error_key] = str(exc)
        _write_analysis_status(progress, f"YOLO 소자 탐지 · 실패 · {exc}")
        return None, str(exc)


def _run_ai_review_workflow(st_api: Any, *, image_bytes: bytes, scope: str, rows: list[dict],
                            records_key: str, ocr_key: str, qwen_key: str,
                            progress: Any = None) -> None:
    """Run RapidOCR and Qwen review on the existing first-pass YOLO proposals."""
    invalidate_qwen_review(st_api.session_state, scope)
    ocr_result = st_api.session_state.get(ocr_key)
    image_fingerprint = hashlib.sha256(image_bytes).hexdigest()
    try:
        from PIL import Image
        from circuitcare.vision.ocr_backend import run_local_ocr

        if (not isinstance(ocr_result, dict)
                or ocr_result.get("source_image_fingerprint") != image_fingerprint):
            _write_analysis_status(progress, "PCB OCR · 진행 중")
            with st_api.spinner("PCB 실크스크린 OCR 중…"):
                with Image.open(io.BytesIO(image_bytes)) as source_image:
                    ocr_result = run_local_ocr(source_image.convert("RGB"))
            ocr_result["source_image_fingerprint"] = image_fingerprint
            st_api.session_state[ocr_key] = ocr_result
            _write_analysis_status(progress, f"PCB OCR · 완료 · {len(ocr_result.get('records', []))}개 문자 후보")
        else:
            _write_analysis_status(progress, "PCB OCR · 재사용")
        associated = associate_ocr_with_components([
            {**row, "bbox_xyxy": [row["x1"], row["y1"], row["x2"], row["y2"]]}
            for row in rows
        ], ocr_result.get("records", []))
        for row, evidence in zip(rows, associated):
            for key in ("ocr_records", "ocr_candidates", "reference_designators", "ocr_class_hints", "ocr_association_status"):
                row[key] = evidence.get(key)
    except Exception as exc:
        st_api.warning(f"OCR를 실행하지 못했습니다 · {exc} · 기존 검수 결과는 유지됩니다.")

    # Read markings only for informative/high-priority parts. OCR output remains a
    # candidate and never becomes measured or verified electrical evidence.
    try:
        from PIL import Image
        from circuitcare.vision.ocr_backend import run_component_marking_ocr

        image_fingerprint = hashlib.sha256(image_bytes).hexdigest()
        marking_cache_key = f"cc_component_marking_ocr_{scope}"
        marking_cache = st_api.session_state.setdefault(marking_cache_key, {})
        eligible = _marking_ocr_targets(rows)
        with Image.open(io.BytesIO(image_bytes)) as source_image:
            for row in eligible:
                marking_fingerprint = hashlib.sha256(json.dumps({
                    "photo": image_fingerprint, "id": row.get("component_id"),
                    "box": [row.get(key) for key in ("x1", "y1", "x2", "y2")],
                    "class": row.get("소자 종류"), "engine": "rapidocr-marking-v1",
                }, sort_keys=True).encode()).hexdigest()
                cached_marking = marking_cache.get(marking_fingerprint)
                if isinstance(cached_marking, dict):
                    result = cached_marking
                elif row.get("marking_ocr_fingerprint") == marking_fingerprint:
                    continue
                else:
                    try:
                        result = run_component_marking_ocr(
                            source_image, [row["x1"], row["y1"], row["x2"], row["y2"]], row["소자 종류"])
                    except Exception as exc:
                        result = {"status": "OCR_FAILED", "error": str(exc), "candidates": [],
                                  "marking": None, "identification_status": "UNVERIFIED_PART_ID"}
                    marking_cache[marking_fingerprint] = result
                row["marking_ocr"] = result
                row["marking_ocr_fingerprint"] = marking_fingerprint
                if result.get("marking"):
                    row["body_marking"] = result["marking"]
                    row["marking_candidates"] = [item.get("text") for item in result.get("candidates", [])
                                                  if item.get("text")]
                    if row.get("소자 종류") == "ic":
                        row["possible_part_number"] = result["marking"]
                        row["part_identity_status"] = "UNVERIFIED_PART_ID"
                        from circuitcare.symptom_diagnosis.ic_identity import resolve_ic_identity
                        row["ic_identity_candidate"] = resolve_ic_identity(result["marking"])
                    if result.get("inferred_value"):
                        row["ocr_inferred_value"] = result["inferred_value"]
                        row["ocr_inference_method"] = result.get("inference_method")
        while len(marking_cache) > 256:
            marking_cache.pop(next(iter(marking_cache)))
    except Exception as exc:
        st_api.warning(f"소자 각인 OCR를 실행하지 못했습니다 · {exc} · 각인 후보 없이 Qwen 검수는 계속합니다.")

    st_api.session_state[records_key] = rows
    try:
        from circuitcare.local_llm import OllamaClient
        from circuitcare.vision.ai_review_service import DEFAULT_REVIEW_MODEL, review_components_with_qwen

        ollama = _ollama_review_client()
        vision_check = getattr(ollama, "model_supports_vision", None)
        if callable(vision_check) and not vision_check(DEFAULT_REVIEW_MODEL):
            raise RuntimeError(f"{DEFAULT_REVIEW_MODEL} 모델이 Ollama에서 vision 기능을 선언하지 않았습니다.")
        live_rows = [row for row in rows if row.get("상태") != "제외"]
        # Build a deterministic priority queue; each request is bounded to 24
        # detections while every active box is eventually included.
        live_rows = sorted(live_rows, key=lambda row: (
            row.get("소자 종류") != "unknown",
            not (row.get("qwen_component_type_candidate") or row.get("marking_ocr")),
            row.get("상태") == "확인",
            float(row.get("type_confidence") or row.get("신뢰도") or 0.0),
        ))
        detections = [{"component_id": row["component_id"], "class_name": row.get("소자 종류", "unknown"),
                       "confidence": row.get("신뢰도"), "bbox_xyxy": [row["x1"], row["y1"], row["x2"], row["y2"]],
                       "ocr_candidates": row.get("ocr_candidates", []),
                       "ocr_class_hints": row.get("ocr_class_hints", []),
                       "reference_designators": row.get("reference_designators", []),
                       "body_marking": row.get("body_marking"),
                       "marking_candidates": row.get("marking_candidates", []),
                       "marking_ocr": row.get("marking_ocr"),
                       "possible_part_number": row.get("possible_part_number")} for row in live_rows]
        with st_api.spinner(f"Qwen Vision으로 소자 검수 중 · {len(detections)}개"):
            review = {"class_reviews": [], "duplicate_reviews": [], "group_reviews": [],
                      "missed_component_candidates": [], "notes": [], "review_diagnostics": {},
                      "model": DEFAULT_REVIEW_MODEL, "verification_status": "AI_SUGGESTIONS_ONLY",
                      "review_batches": []}
            batch_cache_key = f"cc_component_qwen_batches_{scope}"
            batch_cache = st_api.session_state.setdefault(batch_cache_key, {})
            batches = _review_detection_batches(detections)
            failed_batches = []
            for batch_index, batch in enumerate(batches, start=1):
                _write_analysis_status(progress, f"Qwen 소자 검수 · {batch_index}/{len(batches)} 배치 진행 중")
                batch_fingerprint = hashlib.sha256(json.dumps({
                    "image": image_fingerprint, "detections": batch,
                    "ocr": (ocr_result or {}).get("records", []), "schema": "qwen-review-v2",
                }, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
                cached_batch = batch_cache.get(batch_fingerprint)
                if isinstance(cached_batch, dict) and cached_batch.get("status") == "complete":
                    partial = cached_batch["result"]
                else:
                    try:
                        partial = review_components_with_qwen(ollama, DEFAULT_REVIEW_MODEL,
                            image_bytes, batch, (ocr_result or {}).get("records", []))
                        batch_cache[batch_fingerprint] = {"status": "complete", "result": partial}
                    except Exception as batch_exc:
                        batch_cache[batch_fingerprint] = {"status": "failed", "error": str(batch_exc)}
                        failed_batches.append({"batch": batch_index,
                            "detection_ids": [str(row.get("component_id") or row.get("detection_id")) for row in batch],
                            "status": "failed", "stage": getattr(batch_exc, "stage", "qwen_validation"),
                            "error": str(batch_exc)[:500]})
                        continue
                for key in ("class_reviews", "duplicate_reviews", "group_reviews",
                            "missed_component_candidates", "notes"):
                    review[key].extend(partial.get(key, []))
                review["review_batches"].append({"batch": batch_index, "detection_ids": [
                    str(row.get("component_id") or row.get("detection_id")) for row in batch], "status": "complete",
                    "image_count": partial.get("review_image_count", 0),
                    "review_tiles": partial.get("review_tiles", 0),
                    "duplicate_pair_crop_count": partial.get("duplicate_pair_crop_count", 0)})
                review["review_diagnostics"][f"batch_{batch_index}"] = partial.get("review_diagnostics", {})
            review["failed_batches"] = failed_batches
            search_failed = any(row.get("stage") == "candidate_search" for row in failed_batches)
            batch_diagnostics = [review["review_diagnostics"].get(f"batch_{index}", {})
                                 for index in range(1, len(batches) + 1)]
            candidate_items = [item for data in batch_diagnostics
                               for item in data.get("candidate_search", {}).get("items", [])]
            duplicate_items = [item for data in batch_diagnostics
                               for item in data.get("duplicate_pair_search", {}).get("items", [])]
            qwen_items = [item for data in batch_diagnostics
                          for item in data.get("qwen", {}).get("items", [])]
            review["review_diagnostics"].update({
                "candidate_search": {"status": "failed" if search_failed else "complete",
                                     "error": next((row["error"] for row in failed_batches
                                                    if row.get("stage") == "candidate_search"), None),
                                     "items": candidate_items[:240]},
                "duplicate_pair_search": {"status": "complete",
                                          "pair_count": sum(row["duplicate_pair_crop_count"]
                                                            for row in review["review_batches"]),
                                          "items": duplicate_items[:240]},
                "qwen": {"status": ("failed" if any(row.get("stage") in {"qwen", "qwen_validation"}
                                                        for row in failed_batches)
                                     else "not_reached" if search_failed else "complete"),
                         "error": next((row["error"] for row in failed_batches
                                        if row.get("stage") in {"qwen", "qwen_validation"}), None),
                         "items": qwen_items[:240]},
                "ui": {"status": "pending", "events": []},
            })
            if failed_batches:
                review["verification_status"] = ("AI_REVIEW_FAILED" if not review["review_batches"]
                                                   else "AI_REVIEW_PARTIAL")
            review["review_image_count"] = sum(row["image_count"] for row in review["review_batches"])
            review["review_tiles"] = sum(row["review_tiles"] for row in review["review_batches"])
            review["duplicate_pair_crop_count"] = sum(row["duplicate_pair_crop_count"]
                                                       for row in review["review_batches"])
        review["source_fingerprint"] = _review_fingerprint(rows, ocr_result)
        st_api.session_state[qwen_key] = review
        if failed_batches:
            first_failure = failed_batches[0]
            stage_label = {"candidate_search": "묶음 후보 탐색", "qwen": "Qwen 응답",
                           "qwen_validation": "Qwen 응답 검증"}.get(first_failure.get("stage"), "AI 검수")
            st_api.warning(f"{stage_label} 실패 · {first_failure.get('error')} · 성공한 배치 결과와 사용자 검수 기록은 보존됩니다.")
        _write_analysis_status(progress, f"Qwen 소자 검수 · 완료 · {len(review.get('class_reviews', []))}개 제안")
    except Exception as exc:
        stage = getattr(exc, "stage", "qwen_validation")
        qwen_status = ("failed" if stage == "qwen" else
                       "response_invalid" if stage == "qwen_validation" else "not_reached")
        search_status = "failed" if stage == "candidate_search" else "complete"
        st_api.session_state[qwen_key] = {
            "class_reviews": [], "duplicate_reviews": [], "group_reviews": [],
            "missed_component_candidates": [], "notes": [],
            "model": DEFAULT_REVIEW_MODEL, "verification_status": "AI_REVIEW_FAILED",
            "source_fingerprint": _review_fingerprint(rows, ocr_result),
            "review_diagnostics": {"candidate_search": {"status": search_status,
                                                           "error": str(exc) if stage == "candidate_search" else None},
                                   "qwen": {"status": qwen_status, "error": str(exc)},
                                   "ui": {"status": "not_reached", "events": []}},
        }
        _write_analysis_status(progress, f"Qwen 소자 검수 · 실패 · {exc}")
        stage_label = {"candidate_search": "묶음 후보 탐색", "qwen": "Qwen 응답",
                       "qwen_validation": "Qwen 응답 검증"}.get(stage, "AI 검수")
        st_api.warning(f"{stage_label} 실패 · {exc} · 탐지와 사용자 검수 기록은 보존됩니다.")


def ensure_ai_component_review(st_api: Any, *, image_bytes: bytes, scope: str,
                                rows: list[dict], records_key: str,
                                progress: Any = None) -> str:
    """Run the existing OCR→Qwen review once for each unchanged photo observation."""
    ocr_key, qwen_key = review_aux_keys(scope)
    fingerprint = _review_fingerprint(rows, st_api.session_state.get(ocr_key))
    cached = st_api.session_state.get(qwen_key)
    if isinstance(cached, dict) and cached.get("source_fingerprint") == fingerprint:
        _write_analysis_status(progress, "OCR/Qwen 검수 · 결과 재사용")
        return "failed" if cached.get("verification_status") in {"AI_REVIEW_FAILED", "AI_REVIEW_PARTIAL"} else "complete"
    # A separate in-flight/completed key prevents slow-model retries on every Streamlit
    # rerun, including failures. Explicit re-review clears this key in the UI.
    run_key = f"cc_ai_review_run_{scope}"
    if st_api.session_state.get(run_key) == fingerprint:
        _write_analysis_status(progress, "OCR/Qwen 검수 · 이전 결과 재사용")
        return "failed" if isinstance(cached, dict) and cached.get("verification_status") in {
            "AI_REVIEW_FAILED", "AI_REVIEW_PARTIAL"} else "complete"
    st_api.session_state[run_key] = fingerprint
    _run_ai_review_workflow(st_api, image_bytes=image_bytes, scope=scope, rows=rows,
                            records_key=records_key, ocr_key=ocr_key, qwen_key=qwen_key,
                            progress=progress)
    result = st_api.session_state.get(qwen_key)
    if isinstance(result, dict):
        st_api.session_state[run_key] = _review_fingerprint(
            rows, st_api.session_state.get(ocr_key))
        return "failed" if result.get("verification_status") in {
            "AI_REVIEW_FAILED", "AI_REVIEW_PARTIAL"} else "complete"
    return "unavailable"


def _make_group_child_row(scope: str, parent_id: str, index: int, proposal: dict,
                          rows: list[dict]) -> tuple[dict | None, str | None]:
    """Build a still-unconfirmed user-addable row from a validated Qwen group box."""
    box = proposal.get("bbox_xyxy")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None, "invalid_box_shape"
    try:
        coordinates = [float(value) for value in box]
    except (TypeError, ValueError):
        return None, "non_numeric_coordinates"
    if (coordinates[0] < 0 or coordinates[1] < 0 or coordinates[2] <= coordinates[0]
            or coordinates[3] <= coordinates[1]):
        return None, "invalid_box_bounds"
    if coordinates[2] - coordinates[0] < 4 or coordinates[3] - coordinates[1] < 4:
        return None, "box_below_minimum_size"
    qwen_label = normalize_qwen_class(proposal.get("proposed_class"))
    label = qwen_label if qwen_label in CLASS_OPTIONS else "unknown"
    if label == "not_component":
        return None, "not_component_cannot_be_added_as_box"
    if any(same_component_box_pair(coordinates, [row[key] for key in ("x1", "y1", "x2", "y2")])
           for row in rows if all(isinstance(row.get(key), (int, float))
                                  for key in ("x1", "y1", "x2", "y2"))):
        return None, "duplicates_existing_detection"
    digest = hashlib.sha256(f"{scope}:{parent_id}:{index}:{coordinates}".encode()).hexdigest()[:16]
    row = _record(label, coordinates, confidence=proposal.get("confidence", 0.0),
                  location_confidence=proposal.get("confidence", 0.0),
                  type_confidence=proposal.get("confidence"))
    row.update({"component_id": f"qwen-group-{digest}", "source": "qwen_suggestion",
                "상태": "검수 필요", "group_parent_id": parent_id,
                "ai_evidence": proposal.get("evidence", ""),
                "qwen_component_type_candidate": qwen_label if qwen_label not in CLASS_OPTIONS else None,
                "qwen_component_subtype_candidate": normalize_qwen_subtype(qwen_label, proposal.get("proposed_subtype"))})
    return row, None


def render_component_recognition_panel(st_api: Any, *, image_bytes: bytes, project_id: str,
                                       photo_id: str, board_id: str | None = None,
                                       auto_result: dict | None = None,
                                       auto_error: str | None = None) -> None:
    """Compact controls and export; drawing/editing happens on the shared PCB viewer."""
    scope = review_scope(image_bytes, project_id, photo_id)
    def invalidate_review_state() -> None:
        invalidate_component_review_state(st_api.session_state, scope=scope,
                                          project_id=project_id, board_id=board_id)

    _migrate_legacy_review_state(st_api.session_state, image_bytes, project_id, photo_id, scope)
    result_key, records_key, error_key = review_keys(scope)
    if result_key not in st_api.session_state:
        if auto_result is not None:
            st_api.session_state[result_key] = auto_result
            _store_predictions(st_api.session_state, records_key, auto_result)
        elif auto_error:
            st_api.warning(f"자동 소자 탐지 실패 · {auto_error}")
            if st_api.button("소자 탐지 다시 시도", key=f"cc_component_retry_{scope}"):
                st_api.session_state.pop(error_key, None)
                st_api.rerun()
            return
        else:
            st_api.info("선택한 사진에서 자동 탐지를 실행하고 있습니다.")
            return
    old_rows = st_api.session_state.get(records_key, [])
    cleaned_rows = remove_unreviewed_legacy_secondary_rows(old_rows)
    if cleaned_rows != old_rows:
        st_api.session_state[records_key] = cleaned_rows
    st_api.caption("이미지 위의 ‘박스 그리기’를 눌러 드래그하세요. 박스를 선택하면 오른쪽에서 종류·상태·위치를 수정할 수 있습니다.")
    if st_api.button("선택 사진 다시 탐지", key=f"cc_component_run_{scope}"):
        invalidate_review_state()
        st_api.session_state.pop(result_key, None)
        st_api.session_state.pop(error_key, None)
        st_api.rerun()
    rows = st_api.session_state.get(records_key, [])
    st_api.caption(f"박스 {len(rows)}개 · 확인 {sum(row.get('상태') == '확인' for row in rows)}개 · 제외 {sum(row.get('상태') == '제외' for row in rows)}개")
    ocr_key, qwen_key = review_aux_keys(scope)
    if st_api.button("AI 소자 검수", key=f"cc_component_ai_review_{scope}", type="primary"):
        _run_ai_review_workflow(st_api, image_bytes=image_bytes, scope=scope, rows=rows,
            records_key=records_key, ocr_key=ocr_key, qwen_key=qwen_key)
        st_api.rerun()
    with st_api.expander("개별 실행 · 고급", expanded=False):
        st_api.caption("OCR와 Qwen 결과는 제안으로만 저장합니다. 소자 종류 수정·박스 추가·중복 제외는 사용자가 선택해야 합니다.")
        if st_api.button("PCB 실크스크린 OCR", key=f"cc_component_run_ocr_{scope}"):
            try:
                from PIL import Image
                from circuitcare.vision.ocr_backend import run_local_ocr

                with Image.open(io.BytesIO(image_bytes)) as source_image:
                    ocr_result = run_local_ocr(source_image.convert("RGB"))
                st_api.session_state[ocr_key] = ocr_result
                associated = associate_ocr_with_components([
                    {**row, "bbox_xyxy": [row["x1"], row["y1"], row["x2"], row["y2"]]}
                    for row in rows
                ], ocr_result.get("records", []))
                for old_row, evidence in zip(rows, associated):
                    for key in ("ocr_records", "ocr_candidates", "reference_designators", "ocr_class_hints", "ocr_association_status"):
                        old_row[key] = evidence.get(key)
                st_api.session_state[records_key] = rows
                invalidate_review_state()
                st_api.rerun()
            except Exception as exc:
                from circuitcare.vision.ocr_backend import OCRUnavailable

                if isinstance(exc, (OCRUnavailable, ImportError)):
                    st_api.warning(f"RapidOCR를 사용할 수 없습니다 · {exc}")
                else:
                    st_api.warning(f"실크스크린 OCR 실패 · {exc}")
        ocr_result = st_api.session_state.get(ocr_key)
        if isinstance(ocr_result, dict):
            ocr_rows = ocr_result.get("records", [])
            if ocr_rows:
                st_api.caption(f"OCR 문자 {len(ocr_rows)}개 · 엔진 {ocr_result.get('engine', 'RapidOCR')}")
                st_api.dataframe([{"원문": row.get("raw_text", row.get("text", "")),
                                   "Reference 후보": ", ".join(row.get("reference_candidates", [])),
                                   "OCR 점수": row.get("confidence"),
                                   "연결 상태": next((item.get("association_status") for component in rows
                                                     for item in component.get("ocr_records", [])
                                                     if item.get("raw_text", item.get("text")) == row.get("raw_text", row.get("text"))),
                                                    "주변 부품 미확인")}
                                  for row in ocr_rows], hide_index=True, use_container_width=True)
            else:
                st_api.info("OCR 문자 후보가 없습니다. 탐지 결과는 그대로 유지됩니다.")

        if st_api.button("Qwen3.5 9B로 검수 제안 생성", key=f"cc_component_run_qwen_{scope}"):
            st_api.session_state.pop(f"cc_ai_review_run_{scope}", None)
            st_api.session_state.pop(qwen_key, None)
            ensure_ai_component_review(st_api, image_bytes=image_bytes, scope=scope, rows=rows,
                                       records_key=records_key)

        qwen_review = st_api.session_state.get(qwen_key)
        if isinstance(qwen_review, dict) and qwen_review.get("source_fingerprint") != _review_fingerprint(rows, ocr_result):
            invalidate_review_state()
            qwen_review = None
        if isinstance(qwen_review, dict):
            if qwen_review.get("verification_status") == "AI_REVIEW_FAILED":
                diagnostics = qwen_review.get("review_diagnostics", {})
                st_api.error(f"AI 검수 실패 · 탐색 {diagnostics.get('candidate_search', {}).get('status')} · "
                             f"Qwen {diagnostics.get('qwen', {}).get('status')} · "
                             f"{diagnostics.get('qwen', {}).get('error') or diagnostics.get('candidate_search', {}).get('error')}")
            if qwen_review.get("failed_batches"):
                st_api.warning(f"검수 배치 {len(qwen_review['failed_batches'])}개가 실패했습니다. 다시 실행하면 완료 배치는 재사용하고 실패 배치만 재시도합니다.")
                for failed in qwen_review["failed_batches"]:
                    st_api.caption(f"배치 {failed.get('batch')} · {', '.join(failed.get('detection_ids', []))} · {failed.get('error')}")
            diagnostics = qwen_review.get("review_diagnostics", {})
            failed_items = [item for section in ("candidate_search", "duplicate_pair_search", "qwen")
                            for item in diagnostics.get(section, {}).get("items", [])
                            if item.get("status") == "rejected"]
            if failed_items:
                with st_api.expander(f"박스 검수 상세 · 탈락 {len(failed_items)}건", expanded=False):
                    for item in failed_items:
                        st_api.caption(f"{item.get('stage')} · {item.get('component_id', item.get('parent_id', ''))} · {item.get('reason')}")
            st_api.caption(f"제안 모델 · {qwen_review.get('model', 'Qwen Vision')} · 상태 {qwen_review.get('verification_status', 'AI_SUGGESTIONS_ONLY')} · {len(qwen_review.get('review_batches', []))}개 배치 · 1024px 고해상도 타일 {qwen_review.get('review_tiles', 0)}개 · 모든 AI 제안은 미확인")
            false_positive_count = sum(row.get("proposed_class") == "not_component"
                                        for row in qwen_review.get("class_reviews", []))
            duplicate_count = sum(row.get("decision") == "same_component"
                                  for row in qwen_review.get("duplicate_reviews", []))
            qwen_excluded_count = sum(row.get("상태") == "제외"
                                      and str(row.get("exclusion_reason", "")).startswith("qwen_")
                                      for row in rows)
            pending_count = sum(row.get("review_status") in {"AI_SUGGESTED", "AI_CONFLICTED", "UNKNOWN",
                                                               "NOT_COMPONENT_SUGGESTED"}
                                for row in qwen_review.get("class_reviews", []))
            st_api.caption(f"AI 정리 검토 · 오탐 제안 {false_positive_count} · 중복 후보 {duplicate_count} · 사용자 확인 후 제외 {qwen_excluded_count} · 판정 대기 {pending_count} · 자동 삭제 없음")
            by_id = {str(row.get("component_id")): row for row in rows}
            for index, proposal in enumerate(qwen_review.get("class_reviews", [])):
                target = by_id.get(proposal.get("detection_id"))
                if target is None or target.get("상태") == "제외":
                    continue
                base_class = proposal.get("proposed_base_class", proposal.get("proposed_class", "unknown"))
                title = f"{target['component_id']} · YOLO {target.get('소자 종류', 'unknown')} → Qwen {base_class}"
                if proposal.get("proposed_subtype"):
                    title += f" / {proposal['proposed_subtype']}"
                st_api.markdown(f"**{title}** · Qwen 점수 {proposal['confidence']:.2f} · {proposal.get('review_status', 'AI_SUGGESTED')} · 확인 필요")
                st_api.caption(proposal.get("visual_evidence") or proposal.get("evidence") or "시각 근거 없음")
                if proposal.get("ocr_evidence"):
                    st_api.caption(f"OCR 교차 확인 · {proposal['ocr_evidence']}")
                if proposal.get("conflicting_evidence"):
                    st_api.warning("근거 충돌 · " + " · ".join(proposal["conflicting_evidence"]))
                if proposal.get("alternative_classes"):
                    st_api.caption("대안 후보 · " + ", ".join(proposal["alternative_classes"]))
                if target.get("상태") == "확인" or target.get("user_edited"):
                    st_api.caption("사용자 확인·수정 소자 · AI 제안은 기록만 하며 박스와 종류를 변경하지 않습니다.")
                elif proposal.get("decision") == "suggest" and proposal["proposed_class"] == "not_component":
                    if st_api.button("오탐으로 제외 확인", key=f"cc_qwen_false_positive_{scope}_{index}"):
                        record_box_exclusion(st_api.session_state, scope=scope, row=target,
                                             reason="qwen_not_component", review=proposal)
                        st_api.session_state[records_key] = rows
                        invalidate_review_state()
                        st_api.rerun()
                elif proposal.get("decision") == "suggest" and proposal["proposed_class"] in CLASS_OPTIONS:
                    if st_api.button("종류 제안 반영 · 검수 필요 상태 유지", key=f"cc_qwen_class_apply_{scope}_{index}"):
                        target.update({
                            "qwen_component_type_candidate": proposal["proposed_class"],
                            "qwen_component_subtype_candidate": proposal.get("proposed_subtype"),
                            "qwen_component_type_alternatives": proposal.get("alternative_classes", []),
                            "qwen_component_type_evidence": proposal.get("visual_evidence") or proposal.get("evidence", ""),
                            "qwen_component_type_confidence": proposal.get("confidence"),
                            "qwen_component_type_decision": proposal.get("decision"),
                            "qwen_component_type_status": "AI_PROPOSED_UNVERIFIED",
                            "qwen_review_status": proposal.get("review_status", "AI_SUGGESTED"),
                            "qwen_visual_evidence": proposal.get("visual_evidence") or proposal.get("evidence", ""),
                            "qwen_ocr_evidence": proposal.get("ocr_evidence", ""),
                            "qwen_conflicting_evidence": proposal.get("conflicting_evidence", []),
                            "qwen_marking_candidates": proposal.get("marking_candidates", []),
                            "qwen_part_number_candidates": proposal.get("part_number_candidates", []),
                            "qwen_functional_role_candidates": proposal.get("functional_role_candidates", []),
                            "ai_review": proposal,
                        })
                        st_api.session_state[records_key] = rows
                        invalidate_review_state()
                        st_api.rerun()
                elif (proposal.get("decision") == "suggest"
                      and proposal["proposed_class"] in {"other_component", "board_feature"}):
                    if st_api.button("확장 종류 후보 저장 · YOLO 원본 보존", key=f"cc_qwen_extension_apply_{scope}_{index}"):
                        target.update({
                            "qwen_component_type_candidate": proposal["proposed_class"],
                            "qwen_component_subtype_candidate": proposal.get("proposed_subtype"),
                            "qwen_component_type_alternatives": proposal.get("alternative_classes", []),
                            "qwen_component_type_evidence": proposal.get("visual_evidence") or proposal.get("evidence", ""),
                            "qwen_component_type_confidence": proposal.get("confidence"),
                            "qwen_component_type_decision": proposal.get("decision"),
                            "qwen_component_type_status": "AI_PROPOSED_UNVERIFIED",
                            "qwen_review_status": proposal.get("review_status", "AI_SUGGESTED"),
                            "qwen_visual_evidence": proposal.get("visual_evidence") or proposal.get("evidence", ""),
                            "qwen_ocr_evidence": proposal.get("ocr_evidence", ""),
                            "qwen_conflicting_evidence": proposal.get("conflicting_evidence", []),
                            "qwen_marking_candidates": proposal.get("marking_candidates", []),
                            "qwen_part_number_candidates": proposal.get("part_number_candidates", []),
                            "qwen_functional_role_candidates": proposal.get("functional_role_candidates", []),
                            "qwen_review_id": f"{scope}:{target['component_id']}",
                            "ai_review": proposal,
                        })
                        st_api.session_state[records_key] = rows
                        invalidate_review_state()
                        st_api.rerun()
            for index, proposal in enumerate(qwen_review.get("duplicate_reviews", [])):
                ids = proposal.get("detection_ids", [])
                if len(ids) != 2:
                    continue
                pair_rows = [by_id.get(str(item_id)) for item_id in ids]
                if any(row is None or row.get("상태") == "제외" for row in pair_rows):
                    continue
                left_box = [pair_rows[0][key] for key in ("x1", "y1", "x2", "y2")]
                right_box = [pair_rows[1][key] for key in ("x1", "y1", "x2", "y2")]
                geometric_duplicate = same_component_box_pair(left_box, right_box)
                st_api.markdown(f"**중복 검토 · {ids[0]} / {ids[1]} · {proposal['decision']} · 아직 미확인**")
                st_api.caption(proposal.get("evidence") or "시각 근거 없음")
                if not geometric_duplicate:
                    st_api.warning("박스 위치·크기 관계가 중복 근거와 맞지 않습니다. Qwen 제안만으로는 제외할 수 없습니다.")
                protected = any(row.get("상태") == "확인" or row.get("user_edited") for row in pair_rows)
                can_exclude = duplicate_exclusion_enabled(proposal, left_box, right_box, protected=protected)
                if protected:
                    st_api.info("사용자 확인 박스가 포함되어 AI 중복 제외를 막았습니다.")
                choices = [*ids, "둘 다 유지"] if can_exclude else ["둘 다 유지"]
                choice_key = f"cc_qwen_duplicate_choice_{scope}_{index}_{hashlib.sha256('|'.join(ids).encode()).hexdigest()[:8]}"
                if st_api.session_state.get(choice_key) not in choices:
                    st_api.session_state[choice_key] = "둘 다 유지"
                keep_id = st_api.radio(
                    "중복 제안 선택", choices, index=len(choices) - 1, horizontal=True,
                    key=choice_key,
                )
                if st_api.button("선택 결과 저장", key=f"cc_qwen_duplicate_apply_{scope}_{index}"):
                    if keep_id in ids:
                        exclude_id = ids[1] if keep_id == ids[0] else ids[0]
                        record_box_exclusion(st_api.session_state, scope=scope,
                                             row=by_id[exclude_id], reason="qwen_duplicate_review",
                                             review=proposal, duplicate_of=keep_id)
                    else:
                        for target_id in ids:
                            by_id[target_id]["ai_review"] = proposal
                    st_api.session_state[records_key] = rows
                    invalidate_review_state()
                    st_api.rerun()
            for index, proposal in enumerate(qwen_review.get("group_reviews", [])):
                parent_id = str(proposal.get("parent_id", ""))
                parent = by_id.get(parent_id)
                child_ids = [str(value) for value in proposal.get("child_ids", [])]
                children = [by_id[value] for value in child_ids if value in by_id]
                if parent is None or parent.get("상태") == "제외":
                    continue
                st_api.markdown(f"**묶음 박스 검토 · {parent_id} · {proposal['decision']} · 아직 미확인**")
                if children:
                    st_api.caption("기존 개별 박스 재사용 · " + " · ".join(
                        f"{child['component_id']} ({child.get('소자 종류', 'unknown')})" for child in children))
                st_api.caption(proposal.get("evidence") or "큰 박스와 내부 소자 후보의 관계를 검토했습니다.")
                for event in parent.get("group_review_diagnostics", []):
                    st_api.caption(f"UI 반영 · {event.get('status')} · {event.get('reason', event.get('component_id', ''))}")
                if (proposal.get("decision") == "multiple_components" and len(children) >= 2
                        and parent.get("상태") != "확인" and not parent.get("user_edited")):
                    if st_api.button("묶음 박스만 제외 · 개별 박스 유지", key=f"cc_qwen_group_exclude_{scope}_{index}"):
                        review = {**proposal, "child_ids": [child["component_id"] for child in children]}
                        record_box_exclusion(st_api.session_state, scope=scope, row=parent,
                                             reason="qwen_group_container", review=review)
                        parent["group_split_member_ids"] = review["child_ids"]
                        for child in children:
                            child["group_parent_id"] = parent_id
                        st_api.session_state[records_key] = rows
                        invalidate_review_state()
                        st_api.rerun()
                elif proposal.get("decision") == "single_component":
                    if st_api.button("하나의 부품 패키지로 확인", key=f"cc_qwen_group_single_{scope}_{index}"):
                        parent["group_review_confirmed"] = True
                        parent["group_review_decision"] = "single_component"
                        parent["ai_review"] = proposal
                        st_api.session_state[records_key] = rows
                        invalidate_review_state()
                        st_api.rerun()
                for child_index, candidate in enumerate(proposal.get("new_components", [])):
                    st_api.markdown(f"묶음 내부 신규 후보 · {candidate['proposed_class']} · 점수 {candidate['confidence']:.2f}")
                    st_api.caption(f"원본 좌표 · {', '.join(f'{value:.1f}' for value in candidate['bbox_xyxy'])} · 미확인")
                    st_api.caption(candidate.get("evidence") or "Qwen이 확대 영역에서 제안한 누락 후보")
                    if st_api.button("신규 개별 박스 추가 · 검수 필요", key=f"cc_qwen_group_child_{scope}_{index}_{child_index}"):
                        try:
                            candidate_row, reject_reason = _make_group_child_row(
                                scope, parent_id, child_index, candidate, rows)
                            if reject_reason:
                                parent.setdefault("group_review_diagnostics", []).append({
                                    "stage": "ui_apply", "status": "rejected", "reason": reject_reason,
                                    "candidate_index": child_index,
                                })
                                st_api.session_state[records_key] = rows
                                st_api.warning(f"묶음 후보 박스를 추가하지 않았습니다 · {reject_reason}")
                            else:
                                rows.append(candidate_row)
                                parent.setdefault("group_review_diagnostics", []).append({
                                    "stage": "ui_apply", "status": "added", "component_id": candidate_row["component_id"],
                                })
                                st_api.session_state[records_key] = rows
                                invalidate_review_state()
                                st_api.rerun()
                        except Exception as exc:
                            parent.setdefault("group_review_diagnostics", []).append({
                                "stage": "ui_apply", "status": "failed", "reason": str(exc),
                                "candidate_index": child_index,
                            })
                            st_api.session_state[records_key] = rows
                            st_api.error(f"묶음 후보 UI 반영 실패 · {exc}")
            for index, proposal in enumerate(qwen_review.get("missed_component_candidates", [])):
                box = proposal.get("bbox_xyxy")
                if not isinstance(box, list) or len(box) != 4:
                    continue
                st_api.markdown(f"**미탐지 후보 · {proposal['proposed_class']} · 점수 {proposal['confidence']:.2f} · 검수 필요**")
                st_api.caption(proposal.get("evidence") or "Qwen 위치 후보 · YOLO 확인 필요")
                if proposal.get("possible_duplicate_ids"):
                    st_api.warning("박스 겹침 후보 · " + ", ".join(proposal["possible_duplicate_ids"])
                                   + " · 포함 관계만으로 삭제하지 않고 별도 검수합니다.")
                if st_api.button("미확인 후보 박스로 추가", key=f"cc_qwen_missing_apply_{scope}_{index}"):
                    if any(same_component_box_pair(box, [row["x1"], row["y1"], row["x2"], row["y2"]]) for row in rows):
                        st_api.warning("기존 박스와 같은 소자로 보이는 후보는 추가하지 않았습니다.")
                    else:
                        proposed_base = normalize_qwen_class(proposal.get("proposed_class"))
                        display_class = "unknown"
                        candidate = _record(
                            display_class, box, confidence=proposal["confidence"],
                            location_confidence=proposal["confidence"], type_confidence=proposal["confidence"],
                        )
                        candidate.update({"component_id": f"qwen-{hashlib.sha256(f'{scope}:{index}:{box}'.encode()).hexdigest()[:16]}",
                                          "source": "qwen_suggestion", "상태": "검수 필요",
                                          "ai_evidence": proposal.get("evidence", ""),
                                          "qwen_component_type_candidate": proposed_base,
                                          "qwen_component_subtype_candidate": normalize_qwen_subtype(
                                              proposed_base, proposal.get("proposed_subtype")),
                                          "qwen_component_type_evidence": proposal.get("evidence", ""),
                                          "qwen_component_type_status": "AI_PROPOSED_UNVERIFIED",
                                          "qwen_review_status": "AI_SUGGESTED",
                                          "qwen_review_id": f"{scope}:{index}"})
                        candidate["possible_duplicate_ids"] = [row["component_id"] for row in rows
                            if boxes_overlap_candidate(box, [row["x1"], row["y1"], row["x2"], row["y2"]])]
                        candidate["geometry_review_required"] = bool(candidate["possible_duplicate_ids"])
                        rows.append(candidate)
                        st_api.session_state[records_key] = rows
                        invalidate_review_state()
                        st_api.rerun()
            for note in qwen_review.get("notes", []):
                st_api.caption(f"Qwen 관찰 · {note}")

    export_rows = [{"project_id": project_id, "photo_id": photo_id,
                    "active": row.get("상태") != "제외", "class_name": row["소자 종류"],
                    "review_status": row.get("상태", "검수 필요"), "component_id": row.get("component_id"),
                    "location_confidence": row.get("location_confidence"),
                    "type_confidence": row.get("type_confidence"), "match_score": row.get("match_score"),
                    **{key: row[key] for key in ("x1", "y1", "x2", "y2")}}
                   for row in rows]
    st_api.download_button("검수 결과 CSV 저장", pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{photo_id}_component_review.csv", mime="text/csv",
                           key=f"cc_component_review_download_{scope}")
