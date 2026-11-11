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
from circuitcare.component_taxonomy import COMPONENT_LABELS, normalize_component_class
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
                    "ocr_engine": "rapidocr-v1", "review_model": "qwen3.5:9b"}


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


def invalidate_qwen_review(state: Any, scope: str) -> None:
    """Drop suggestions whenever their detection/OCR evidence changes."""
    _ocr_key, qwen_key = review_aux_keys(scope)
    state.pop(qwen_key, None)


def invalidate_component_review_state(state: Any, *, scope: str, project_id: str, board_id: str | None) -> None:
    """Invalidate derived AI/diagnosis results while leaving measurement records untouched."""
    invalidate_qwen_review(state, scope)
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
        "component_id", "소자 종류", "상태", "x1", "y1", "x2", "y2", "신뢰도",
        "ocr_candidates", "reference_designators", "ocr_class_hints",
    )} for row in rows]
    payload = json.dumps({"rows": snapshot, "ocr": (ocr_result or {}).get("records", [])},
                         ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def duplicate_exclusion_enabled(proposal: dict, left_box: list | tuple,
                                right_box: list | tuple) -> bool:
    """Require Qwen agreement and strong geometric evidence before exclusion."""
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
                               review_image_size: tuple[int, int] | None = None) -> tuple[dict | None, str | None]:
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
        return st_api.session_state[result_key], None
    if error_key in st_api.session_state:
        return None, st_api.session_state[error_key]
    try:
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
        return result, None
    except Exception as exc:
        st_api.session_state[error_key] = str(exc)
        return None, str(exc)


def _run_ai_review_workflow(st_api: Any, *, image_bytes: bytes, scope: str, rows: list[dict],
                            records_key: str, ocr_key: str, qwen_key: str) -> None:
    """Run RapidOCR and Qwen review on the existing first-pass YOLO proposals."""
    invalidate_qwen_review(st_api.session_state, scope)
    ocr_result = st_api.session_state.get(ocr_key)
    image_fingerprint = hashlib.sha256(image_bytes).hexdigest()
    try:
        from PIL import Image
        from circuitcare.vision.ocr_backend import run_local_ocr

        if (not isinstance(ocr_result, dict)
                or ocr_result.get("source_image_fingerprint") != image_fingerprint):
            with st_api.spinner("PCB 실크스크린 OCR 중…"):
                with Image.open(io.BytesIO(image_bytes)) as source_image:
                    ocr_result = run_local_ocr(source_image.convert("RGB"))
            ocr_result["source_image_fingerprint"] = image_fingerprint
            st_api.session_state[ocr_key] = ocr_result
        associated = associate_ocr_with_components([
            {**row, "bbox_xyxy": [row["x1"], row["y1"], row["x2"], row["y2"]]}
            for row in rows
        ], ocr_result.get("records", []))
        for row, evidence in zip(rows, associated):
            for key in ("ocr_records", "ocr_candidates", "reference_designators", "ocr_class_hints", "ocr_association_status"):
                row[key] = evidence.get(key)
    except Exception as exc:
        st_api.warning(f"OCR를 실행하지 못했습니다 · {exc} · 기존 검수 결과는 유지됩니다.")

    st_api.session_state[records_key] = rows
    try:
        from circuitcare.local_llm import OllamaClient
        from circuitcare.vision.ai_review_service import DEFAULT_REVIEW_MODEL, review_components_with_qwen

        ollama = OllamaClient(timeout_seconds=300, num_predict=4096)
        vision_check = getattr(ollama, "model_supports_vision", None)
        if callable(vision_check) and not vision_check(DEFAULT_REVIEW_MODEL):
            raise RuntimeError(f"{DEFAULT_REVIEW_MODEL} 모델이 Ollama에서 vision 기능을 선언하지 않았습니다.")
        live_rows = [row for row in rows if row.get("상태") != "제외"]
        # Prioritize uncertain and low-confidence observations and cap large-board
        # review load; Qwen still receives the full board image plus their crops.
        live_rows = sorted(live_rows, key=lambda row: (
            row.get("소자 종류") != "unknown",
            row.get("상태") == "확인",
            float(row.get("type_confidence") or row.get("신뢰도") or 0.0),
        ))[:24]
        detections = [{"component_id": row["component_id"], "class_name": row.get("소자 종류", "unknown"),
                       "confidence": row.get("신뢰도"), "bbox_xyxy": [row["x1"], row["y1"], row["x2"], row["y2"]],
                       "ocr_candidates": row.get("ocr_candidates", []),
                       "ocr_class_hints": row.get("ocr_class_hints", []),
                       "reference_designators": row.get("reference_designators", []),
                       "body_marking": row.get("body_marking")} for row in live_rows]
        with st_api.spinner("Qwen3.5 9B로 소자 종류·중복·미탐지 후보 검수 중…"):
            review = review_components_with_qwen(ollama, DEFAULT_REVIEW_MODEL,
                image_bytes, detections, (ocr_result or {}).get("records", []))
        review["source_fingerprint"] = _review_fingerprint(rows, ocr_result)
        st_api.session_state[qwen_key] = review
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
        stage_label = {"candidate_search": "묶음 후보 탐색", "qwen": "Qwen 응답",
                       "qwen_validation": "Qwen 응답 검증"}.get(stage, "AI 검수")
        st_api.warning(f"{stage_label} 실패 · {exc} · 탐지와 사용자 검수 기록은 보존됩니다.")


def ensure_ai_component_review(st_api: Any, *, image_bytes: bytes, scope: str,
                                rows: list[dict], records_key: str) -> str:
    """Run the existing OCR→Qwen review once for each unchanged photo observation."""
    ocr_key, qwen_key = review_aux_keys(scope)
    fingerprint = _review_fingerprint(rows, st_api.session_state.get(ocr_key))
    cached = st_api.session_state.get(qwen_key)
    if isinstance(cached, dict) and cached.get("source_fingerprint") == fingerprint:
        return "failed" if cached.get("verification_status") == "AI_REVIEW_FAILED" else "complete"
    # A separate in-flight/completed key prevents slow-model retries on every Streamlit
    # rerun, including failures. Explicit re-review clears this key in the UI.
    run_key = f"cc_ai_review_run_{scope}"
    if st_api.session_state.get(run_key) == fingerprint:
        return "failed" if isinstance(cached, dict) and cached.get("verification_status") == "AI_REVIEW_FAILED" else "complete"
    st_api.session_state[run_key] = fingerprint
    _run_ai_review_workflow(st_api, image_bytes=image_bytes, scope=scope, rows=rows,
                            records_key=records_key, ocr_key=ocr_key, qwen_key=qwen_key)
    result = st_api.session_state.get(qwen_key)
    if isinstance(result, dict):
        st_api.session_state[run_key] = _review_fingerprint(
            rows, st_api.session_state.get(ocr_key))
        return "failed" if result.get("verification_status") == "AI_REVIEW_FAILED" else "complete"
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
    label = normalize_component_class(proposal.get("proposed_class"))
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
                "ai_evidence": proposal.get("evidence", "")})
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
            try:
                from circuitcare.local_llm import OllamaClient
                from circuitcare.vision.ai_review_service import DEFAULT_REVIEW_MODEL, review_components_with_qwen

                live_rows = [row for row in rows if row.get("상태") != "제외"]
                qwen_detections = [{"component_id": row["component_id"], "class_name": row.get("소자 종류", "unknown"),
                                    "confidence": row.get("신뢰도"),
                                    "bbox_xyxy": [row["x1"], row["y1"], row["x2"], row["y2"]],
                                    "ocr_candidates": row.get("ocr_candidates", []),
                                    "ocr_class_hints": row.get("ocr_class_hints", []),
                                    "reference_designators": row.get("reference_designators", []),
                                    "body_marking": row.get("body_marking")}
                                   for row in live_rows]
                with st_api.spinner("Qwen3.5 9B가 사진과 불확실한 후보를 검토하고 있습니다…"):
                    review = review_components_with_qwen(
                        OllamaClient(timeout_seconds=300, num_predict=4096), DEFAULT_REVIEW_MODEL, image_bytes,
                        qwen_detections, (ocr_result or {}).get("records", []),
                    )
                review["source_fingerprint"] = _review_fingerprint(rows, ocr_result)
                st_api.session_state[qwen_key] = review
            except Exception as exc:
                st_api.warning(f"Qwen 검수 제안을 만들지 못했습니다 · {exc} · YOLO 박스와 사용자 검수 기록은 보존됩니다.")

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
            diagnostics = qwen_review.get("review_diagnostics", {})
            failed_items = [item for section in ("candidate_search", "duplicate_pair_search", "qwen")
                            for item in diagnostics.get(section, {}).get("items", [])
                            if item.get("status") == "rejected"]
            if failed_items:
                with st_api.expander(f"박스 검수 상세 · 탈락 {len(failed_items)}건", expanded=False):
                    for item in failed_items:
                        st_api.caption(f"{item.get('stage')} · {item.get('component_id', item.get('parent_id', ''))} · {item.get('reason')}")
            st_api.caption(f"제안 모델 · {qwen_review.get('model', 'Qwen3.5 9B')} · 상태 {qwen_review.get('verification_status', 'AI_SUGGESTIONS_ONLY')} · 1024px 고해상도 타일 {qwen_review.get('review_tiles', 0)}개 확인 · 모든 제안은 미확인")
            by_id = {str(row.get("component_id")): row for row in rows}
            for index, proposal in enumerate(qwen_review.get("class_reviews", [])):
                target = by_id.get(proposal.get("detection_id"))
                if target is None:
                    continue
                ocr_hints = target.get("ocr_class_hints", [])
                mismatch = ocr_hints and target.get("소자 종류") not in ocr_hints
                title = f"{target['component_id']} · YOLO {target.get('소자 종류', 'unknown')} → 제안 {proposal['proposed_class']}"
                if mismatch:
                    title += " · OCR/YOLO 불일치"
                st_api.markdown(f"**{title}** · 점수 {proposal['confidence']:.2f} · 아직 미확인")
                st_api.caption(proposal.get("evidence") or "시각 근거 없음")
                if proposal.get("decision") == "suggest" and proposal["proposed_class"] == "not_component":
                    if st_api.button("오탐으로 제외 확인", key=f"cc_qwen_false_positive_{scope}_{index}"):
                        record_box_exclusion(st_api.session_state, scope=scope, row=target,
                                             reason="qwen_not_component", review=proposal)
                        st_api.session_state[records_key] = rows
                        invalidate_review_state()
                        st_api.rerun()
                elif proposal.get("decision") == "suggest" and proposal["proposed_class"] in CLASS_OPTIONS:
                    if st_api.button("종류 제안 반영 · 검수 필요 상태 유지", key=f"cc_qwen_class_apply_{scope}_{index}"):
                        target["소자 종류"] = proposal["proposed_class"]
                        target["상태"] = "검수 필요"
                        target["user_edited"] = True
                        target["ai_review"] = proposal
                        st_api.session_state[records_key] = rows
                        invalidate_review_state()
                        st_api.rerun()
            for index, proposal in enumerate(qwen_review.get("duplicate_reviews", [])):
                ids = proposal.get("detection_ids", [])
                if len(ids) != 2:
                    continue
                pair_rows = [by_id.get(str(item_id)) for item_id in ids]
                if any(row is None for row in pair_rows):
                    continue
                left_box = [pair_rows[0][key] for key in ("x1", "y1", "x2", "y2")]
                right_box = [pair_rows[1][key] for key in ("x1", "y1", "x2", "y2")]
                geometric_duplicate = same_component_box_pair(left_box, right_box)
                st_api.markdown(f"**중복 검토 · {ids[0]} / {ids[1]} · {proposal['decision']} · 아직 미확인**")
                st_api.caption(proposal.get("evidence") or "시각 근거 없음")
                if not geometric_duplicate:
                    st_api.warning("박스 위치·크기 관계가 중복 근거와 맞지 않습니다. Qwen 제안만으로는 제외할 수 없습니다.")
                can_exclude = duplicate_exclusion_enabled(proposal, left_box, right_box)
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
                if parent is None:
                    continue
                st_api.markdown(f"**묶음 박스 검토 · {parent_id} · {proposal['decision']} · 아직 미확인**")
                if children:
                    st_api.caption("기존 개별 박스 재사용 · " + " · ".join(
                        f"{child['component_id']} ({child.get('소자 종류', 'unknown')})" for child in children))
                st_api.caption(proposal.get("evidence") or "큰 박스와 내부 소자 후보의 관계를 검토했습니다.")
                for event in parent.get("group_review_diagnostics", []):
                    st_api.caption(f"UI 반영 · {event.get('status')} · {event.get('reason', event.get('component_id', ''))}")
                if proposal.get("decision") == "multiple_components" and len(children) >= 2:
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
                        candidate = _record(
                            proposal["proposed_class"], box, confidence=proposal["confidence"],
                            location_confidence=proposal["confidence"], type_confidence=proposal["confidence"],
                        )
                        candidate.update({"component_id": f"qwen-{hashlib.sha256(f'{scope}:{index}:{box}'.encode()).hexdigest()[:16]}",
                                          "source": "qwen_suggestion", "상태": "검수 필요",
                                          "ai_evidence": proposal.get("evidence", ""),
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
