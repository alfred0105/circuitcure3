"""Component detection and manual box/class review for the PCB Workbench."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from functools import lru_cache
import importlib.util
import io
import os
import sys
import time
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
from circuitcare.vision.ai_review_service import (
    AIReviewStageError, _duplicate_pair_candidates,
)
from circuitcare.frontend.measurement_state import invalidate_legacy_diagnosis
from circuitcare.analysis.models import AnalysisScope
from circuitcare.analysis.state_manager import invalidate_analysis
from circuitcare.vision.ai_review_jobs import get_ai_review_job, submit_ai_review
from circuitcare.frontend.workbench_state import ACTIVE_PHOTO_KEY, SELECTION_KEY

APP_DIR = Path(__file__).resolve().parents[3] / "apps" / "pcb-component-recognition"
MODULE_NAME = "circuitcare_pcb_component_app_server"
CLASS_OPTIONS = list(COMPONENT_LABELS)
STATUS_OPTIONS = ["검수 필요", "확인", "제외"]
LAST_EXCLUDED_KEY = "circuitcare_v4_last_excluded_box"
INFERENCE_CONFIG = {"model_pipeline": "yolo26s+doctor-v12n+arshia-yolov8s", "conf": 0.20,
                    "tile": 1024, "overlap": 0.25, "postprocess": "one-to-one-v2",
                    "ocr_engine": "rapidocr-v1", "review_model": "qwen3.5:9b",
                    "review_schema": "qwen-hierarchical-review-v3"}
AUTO_CLEANUP_VALIDATION = None  # No independent PCB dataset/calibration is registered.


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


def _review_detection_batches(detections: list[dict], batch_size: int = 6) -> list[list[dict]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return [detections[index:index + batch_size]
            for index in range(0, len(detections), batch_size)] or [[]]


def _missing_class_review_ids(batch: list[dict], partial: dict) -> list[str]:
    """Return requested boxes for which Qwen produced no validated class decision."""
    requested = {str(row.get("component_id") or row.get("detection_id") or "") for row in batch}
    reviewed = {str(row.get("detection_id") or "") for row in partial.get("class_reviews", [])
                if isinstance(row, dict)}
    return sorted(value for value in requested - reviewed if value)


def _qwen_review_batch_size() -> int:
    try:
        value = int(os.environ.get("CIRCUITCARE_QWEN_BATCH_SIZE", "4"))
    except (TypeError, ValueError):
        return 4
    return value if value in {2, 4, 6, 8} else 4


def _pending_duplicate_pair_candidates(
    candidates: list[dict], reviewed_pair_ids: set[str], *, limit: int = 12,
) -> list[dict]:
    """Page the deterministic board-wide pair queue without discarding deferred pairs."""
    if limit <= 0:
        return []
    pending = []
    for pair in candidates:
        ids = sorted(str(row.get("detection_id") or row.get("component_id") or "")
                     for row in pair.get("detections", ()))
        pair_id = "|".join(ids)
        if len(ids) == 2 and pair_id not in reviewed_pair_ids:
            pending.append(pair)
        if len(pending) >= limit:
            break
    return pending


def _duplicate_pair_id(pair: dict) -> str:
    ids = sorted(str(row.get("detection_id") or row.get("component_id") or "")
                 for row in pair.get("detections", ()))
    return "|".join(ids) if len(ids) == 2 and ids[0] and ids[1] and ids[0] != ids[1] else ""


def _duplicate_pair_input_fingerprint(pair: dict, *, image_fingerprint: str,
                                      model: str, schema_version: str) -> str:
    rows = sorted(({
        "id": str(row.get("detection_id") or row.get("component_id") or ""),
        "box": row.get("bbox_xyxy", [row.get(key) for key in ("x1", "y1", "x2", "y2")]),
        "class": row.get("class_name", row.get("소자 종류", "unknown")),
    } for row in pair.get("detections", ())), key=lambda item: item["id"])
    payload = {"image": image_fingerprint, "detections": rows, "model": model,
               "schema": schema_version}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     default=str).encode()).hexdigest()


def _sync_duplicate_pair_queue(
    candidates: list[dict], queue: Any, *, image_fingerprint: str,
    model: str = "qwen3.5:9b", schema_version: str = "qwen-review-v2-board-pairs",
) -> dict:
    """Reconcile board-wide pair candidates with versioned, resumable review states."""
    if not isinstance(queue, dict) or queue.get("version") != 2:
        queue = {"version": 2, "pairs": {}, "candidate_ids": []}
    else:
        queue = deepcopy(queue)
    pairs = queue.setdefault("pairs", {})
    current = {}
    for candidate in candidates:
        pair_id = _duplicate_pair_id(candidate)
        if not pair_id:
            continue
        fingerprint = _duplicate_pair_input_fingerprint(
            candidate, image_fingerprint=image_fingerprint, model=model,
            schema_version=schema_version)
        current[pair_id] = fingerprint
        previous = pairs.get(pair_id)
        if not isinstance(previous, dict):
            pairs[pair_id] = {"fingerprint": fingerprint, "status": "PENDING",
                              "attempts": 0, "proposal": None, "error": None,
                              "history": []}
        elif previous.get("fingerprint") != fingerprint:
            history = list(previous.get("history", []))
            history.append({"status": "STALE", "fingerprint": previous.get("fingerprint")})
            pairs[pair_id] = {"fingerprint": fingerprint, "status": "PENDING",
                              "attempts": 0, "proposal": None, "error": None,
                              "history": history[-10:]}
        elif previous.get("status") == "RUNNING":
            # A Streamlit rerun/process interruption must not strand a claimed job.
            previous["status"] = "PENDING"
    for pair_id, previous in pairs.items():
        if pair_id not in current and previous.get("status") in {
                "PENDING", "RUNNING", "DEFERRED", "FAILED"}:
            previous["status"] = "STALE"
            previous["error"] = "박스 후보가 현재 입력에서 사라졌습니다."
    queue["candidate_ids"] = sorted(current)
    return queue


def _duplicate_pair_queue_summary(queue: dict) -> dict[str, Any]:
    active = [queue.get("pairs", {}).get(pair_id, {}) for pair_id in queue.get("candidate_ids", [])]
    counts = {status.lower(): sum(row.get("status") == status for row in active)
              for status in ("PENDING", "RUNNING", "REVIEWED", "FAILED", "DEFERRED", "STALE")}
    counts["total"] = len(active)
    counts["confirmed_duplicate"] = sum(
        row.get("status") == "REVIEWED" and isinstance(row.get("proposal"), dict)
        and row["proposal"].get("decision") == "same_component" for row in active)
    counts["complete"] = bool(active) and counts["reviewed"] == counts["total"]
    return counts


def _claim_duplicate_pair_page(candidates: list[dict], queue: dict, *, limit: int = 12) -> list[dict]:
    """Claim a bounded page; leave all unclaimed candidates resumable as DEFERRED."""
    records = queue.setdefault("pairs", {})
    selected = []
    retryable = [candidate for candidate in candidates
                 if records.get(_duplicate_pair_id(candidate), {}).get("status") == "FAILED"]
    deferred = [candidate for candidate in candidates
                if records.get(_duplicate_pair_id(candidate), {}).get("status") in {"PENDING", "DEFERRED"}]
    ordered_candidates = [*retryable, *deferred]
    for candidate in ordered_candidates:
        pair_id = _duplicate_pair_id(candidate)
        record = records.get(pair_id, {})
        if record.get("status") not in {"PENDING", "DEFERRED", "FAILED"}:
            continue
        if len(selected) >= max(0, limit):
            if record.get("status") in {"PENDING", "DEFERRED"}:
                record["status"] = "DEFERRED"
            continue
        record["status"] = "RUNNING"
        record["attempts"] = int(record.get("attempts", 0)) + 1
        selected.append(candidate)
    selected_ids = {_duplicate_pair_id(candidate) for candidate in selected}
    for candidate in candidates:
        pair_id = _duplicate_pair_id(candidate)
        record = records.get(pair_id, {})
        if pair_id not in selected_ids and record.get("status") in {"PENDING", "DEFERRED"}:
            record["status"] = "DEFERRED"
    return selected


def _review_duplicate_pair_page(client: Any, *, image_bytes: bytes, detections: list[dict],
                                pairs: list[dict], ocr_records: list[dict] | None = None,
                                progress: Any = None) -> tuple[dict[str, dict], dict[str, dict]]:
    """Review pair crops one at a time so a valid answer cannot be omitted in a large JSON response."""
    from circuitcare.vision.ai_review_service import DEFAULT_REVIEW_MODEL, review_components_with_qwen

    indexed = {str(row.get("detection_id") or row.get("component_id")): row for row in detections}
    completed, failed = {}, {}
    for index, pair in enumerate(pairs, start=1):
        pair_id = _duplicate_pair_id(pair)
        ids = sorted(pair_id.split("|")) if pair_id else []
        if len(ids) != 2 or any(item not in indexed for item in ids):
            failed[pair_id] = {"stage": "qwen_validation", "error": "중복 후보 ID가 현재 탐지 목록에 없습니다."}
            continue
        _write_analysis_status(progress, f"Qwen 중복 후보 검토 · {index}/{len(pairs)}쌍")
        try:
            result = review_components_with_qwen(
                client, DEFAULT_REVIEW_MODEL, image_bytes, [indexed[item] for item in ids],
                ocr_records or [], duplicate_pair_candidates=[pair], duplicate_pairs_only=True,
            )
            proposal = next((item for item in result.get("duplicate_reviews", [])
                            if sorted(map(str, item.get("detection_ids", []))) == ids), None)
            if not isinstance(proposal, dict):
                raise RuntimeError("Qwen duplicate response did not include the requested pair.")
            completed[pair_id] = proposal
        except Exception as exc:
            failed[pair_id] = {"stage": getattr(exc, "stage", "qwen"), "error": str(exc)[:500]}
    return completed, failed


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


def _record_vision_metric(state: Any, stage: str, duration_ms: float, status: str) -> None:
    events = state.setdefault("cc_performance_events", [])
    events.append({"stage": stage, "duration_ms": round(duration_ms, 2), "status": status})
    del events[:-50]


@lru_cache(maxsize=1)
def _ollama_review_client() -> Any:
    """Reuse the thread-safe local transport without invoking Streamlit in workers."""
    from circuitcare.local_llm import OllamaClient
    # Structured per-box fields, OCR evidence and split proposals can exceed a
    # 4096-token completion even for a six-box batch on Qwen3.5 9B.
    # Keep an upper bound on JSON generation. Large output budgets let a malformed
    # or overly verbose batch occupy the single MLX worker for several minutes.
    return OllamaClient(auto_start=True, timeout_seconds=300, num_predict=4096)


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
    row.setdefault("detector_class", row.get("소자 종류", "unknown"))
    row.update({"소자 종류": component_type, "상태": status,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "user_edited": True})
    # A manual type edit is an explicit override of any older Qwen extension
    # confirmation. Keep AI evidence for audit, but make the effective class
    # and Registry status follow the user edit.
    row.update({"confirmed_base_class": component_type, "confirmed_subtype": None,
                "effective_class": component_type, "diagnosis_class": component_type,
                "component_type_status": "USER_CONFIRMED",
                "type_confirmation_source": "USER_MANUAL"})
    row.pop("qwen_user_confirmed_extension", None)


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
    if not isinstance(snapshot, dict):
        return None
    batch_ids = snapshot.get("component_ids")
    batch_before = snapshot.get("before_rows")
    if isinstance(batch_ids, list) and isinstance(batch_before, dict):
        restored = []
        for component_id in batch_ids:
            before = batch_before.get(str(component_id))
            row = next((item for item in rows
                        if str(item.get("component_id")) == str(component_id)), None)
            if row is None or not isinstance(before, dict):
                continue
            row.clear()
            row.update(deepcopy(before))
            restored.append(str(component_id))
        if restored:
            state.pop(f"{LAST_EXCLUDED_KEY}_{scope}", None)
            return restored[0]
        return None
    if not isinstance(snapshot.get("before"), dict):
        return None
    component_id = snapshot.get("component_id")
    row = next((item for item in rows if item.get("component_id") == component_id), None)
    if row is None:
        return None
    row.clear()
    row.update(deepcopy(snapshot["before"]))
    state.pop(f"{LAST_EXCLUDED_KEY}_{scope}", None)
    return str(component_id)


def _observation_fingerprint(rows: list[dict]) -> str:
    """Stable version of YOLO observations, independent of later OCR/review output."""
    snapshot = [{key: row.get(key) for key in (
        "component_id", "detector_class", "소자 종류", "x1", "y1", "x2", "y2", "신뢰도",
    )} for row in rows]
    payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _ocr_evidence_fingerprint(ocr_result: dict | None) -> str:
    """Version OCR output separately so producing OCR cannot mutate the YOLO job input."""
    result = ocr_result if isinstance(ocr_result, dict) else {}
    payload = {"source_image_fingerprint": result.get("source_image_fingerprint"),
               "status": result.get("status"), "engine": result.get("engine"),
               "records": result.get("records", [])}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     default=str).encode()).hexdigest()


def _inference_fingerprint(rows: list[dict], ocr_result: dict | None = None) -> str:
    """Fingerprint YOLO observations plus OCR evidence for AI-result cache validation."""
    ai_evidence = [{key: row.get(key) for key in (
        "ocr_candidates", "reference_designators", "ocr_class_hints", "body_marking",
        "marking_candidates", "marking_ocr_fingerprint", "possible_part_number",
        "ocr_source", "ocr_score", "qwen_marking_candidates", "qwen_part_number_candidates",
        "qwen_ocr_evidence", "qwen_conflicting_evidence",
    )} for row in rows]
    payload = {"observation": _observation_fingerprint(rows), "row_ocr": ai_evidence,
               "ocr_evidence": _ocr_evidence_fingerprint(ocr_result)}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     default=str).encode()).hexdigest()


_USER_REVIEW_FIELDS = (
    "component_id", "board_component_id", "소자 종류", "상태", "status_before_exclusion", "user_edited",
    "user_confirmed", "user_confirmed_class", "confirmed_base_class", "confirmed_subtype",
    "confirmed_part_number", "part_number_status", "part_identity_status",
    "reference_designator", "user_reference_designator", "fault_role", "fault_role_status",
    "measurement_ids", "measurement_id", "measurement_mapping_status",
)
_AI_REVIEW_ROW_FIELDS = (
    "ocr_records", "ocr_candidates", "reference_designators", "ocr_class_hints",
    "ocr_association_status", "marking_ocr", "marking_ocr_fingerprint", "marking_ocr_status",
    "body_marking", "marking_candidates", "ocr_source", "ocr_score", "possible_part_number",
    "ic_identity_candidate", "ocr_inferred_value", "ocr_inference_method",
)


def _review_user_version(rows: list[dict]) -> str:
    protected = [{key: row.get(key) for key in _USER_REVIEW_FIELDS}
                 for row in rows if isinstance(row, dict)]
    return hashlib.sha256(json.dumps(protected, ensure_ascii=False, sort_keys=True,
                                     default=str).encode()).hexdigest()


def _review_fingerprint(rows: list[dict], ocr_result: dict | None) -> str:
    """Job input fingerprint: YOLO observation plus user decisions, not worker OCR output."""
    payload = {"observation": _observation_fingerprint(rows),
               "user_review_version": _review_user_version(rows)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def merge_ai_review_rows(current_rows: list[dict], worker_rows: list[dict]) -> list[dict]:
    """Merge only AI-owned OCR evidence into current rows, preserving user state."""
    merged = deepcopy([row for row in current_rows if isinstance(row, dict)])
    by_id = {str(row.get("component_id")): row for row in merged if row.get("component_id") is not None}
    for proposed in worker_rows:
        if not isinstance(proposed, dict) or proposed.get("component_id") is None:
            continue
        target = by_id.get(str(proposed["component_id"]))
        if target is None:
            # Worker snapshots are not authoritative for IDs or manually added boxes.
            continue
        for key in _AI_REVIEW_ROW_FIELDS:
            if key not in proposed:
                continue
            if key == "possible_part_number" and target.get("confirmed_part_number"):
                continue
            if key == "body_marking" and target.get("user_confirmed"):
                continue
            target[key] = deepcopy(proposed[key])
    return merged


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


def qwen_cleanup_disposition(
    proposal: dict, *, protected: bool = False, geometry_valid: bool = False,
    has_conflict: bool = False, is_group: bool = False,
    validation: dict[str, Any] | None = AUTO_CLEANUP_VALIDATION,
) -> tuple[str, str]:
    """Return a reversible cleanup state; never auto-exclude on uncalibrated scores."""
    if protected:
        return "USER_CONFIRMED", "사용자가 확인·수정한 박스"
    decision = str(proposal.get("decision") or "uncertain")
    if decision == "different_components":
        return "KEEP", "서로 다른 소자로 제안됨"
    if is_group or has_conflict or decision == "uncertain":
        return "REVIEW_REQUIRED", "묶음 박스·근거 충돌·불확실 판정"
    if decision == "same_component" and not geometry_valid:
        return "REVIEW_REQUIRED", "기하학적 정합이 확인되지 않음"
    if decision == "suggest" and proposal.get("proposed_class") == "not_component":
        return "REVIEW_REQUIRED", "실제 데이터로 검증된 자동 정리 기준이 없어 사용자 확인 필요"
    if decision not in {"same_component", "not_component"}:
        return "REVIEW_REQUIRED", "지원하지 않는 정리 판정"
    if decision == "not_component" and proposal.get("proposed_class") != "not_component":
        return "REVIEW_REQUIRED", "오탐으로 확인된 종류 제안이 아님"
    if not str(proposal.get("visual_evidence") or proposal.get("evidence") or "").strip():
        return "REVIEW_REQUIRED", "박스 자체의 시각 근거가 기록되지 않음"
    if isinstance(validation, dict) and _cleanup_validation_passes(validation, decision):
        return "AUTO_EXCLUDED", "검증된 자동 정리 정책 통과"
    return "REVIEW_REQUIRED", "실제 데이터로 검증된 자동 정리 기준이 없어 사용자 확인 필요"


def _cleanup_validation_passes(validation: dict, decision: str) -> bool:
    """Require an independently measured, versioned safety policy for auto-exclusion."""
    if (validation.get("validated") is not True
            or not str(validation.get("policy_version") or "").strip()
            or not str(validation.get("independent_dataset_fingerprint") or "").strip()
            or validation.get("evaluation_mode") != "REAL_LABELED_PCB"):
        return False
    metrics = validation.get("metrics")
    if not isinstance(metrics, dict) or int(metrics.get("sample_count", 0) or 0) < 1:
        return False
    def metric(name: str, default: float) -> float:
        try:
            return float(metrics[name])
        except (KeyError, TypeError, ValueError):
            return default
    if decision == "same_component":
        return (metric("duplicate_precision_lower_bound", 0) >= .995
                and metric("normal_component_exclusion_rate_upper_bound", 1) <= .001
                and metric("group_or_connector_false_exclusion_rate_upper_bound", 1) <= .001)
    if decision == "not_component":
        return (metric("false_positive_precision_lower_bound", 0) >= .995
                and metric("normal_component_exclusion_rate_upper_bound", 1) <= .001
                and metric("ocr_conflict_false_exclusion_rate_upper_bound", 1) <= .001)
    return False


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
        _record_vision_metric(st_api.session_state, "yolo_detection", 0.0, "cache_hit")
        _write_analysis_status(progress, "YOLO 소자 탐지 · 재사용")
        return st_api.session_state[result_key], None
    if error_key in st_api.session_state:
        _record_vision_metric(st_api.session_state, "yolo_detection", 0.0, "cached_failure")
        _write_analysis_status(progress, "YOLO 소자 탐지 · 이전 실행 실패")
        return None, st_api.session_state[error_key]
    started = time.perf_counter()
    try:
        _write_analysis_status(progress, "YOLO 소자 탐지 · 진행 중")
        # The workbench owns a persistent workflow_status_slot. Showing a
        # second spinner here makes the initial loader disappear/reappear as
        # YOLO hands off to the asynchronous OCR/Qwen panel.
        from contextlib import nullcontext
        progress_context = nullcontext() if progress is not None else st_api.spinner(
            "선택한 PCB 사진에서 소자를 자동 탐지하고 있습니다…")
        with progress_context:
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
        _record_vision_metric(st_api.session_state, "yolo_detection",
                              (time.perf_counter() - started) * 1000, "complete")
        _write_analysis_status(progress, f"YOLO 소자 탐지 · 완료 · {len(result.get('detections', []))}개")
        return result, None
    except Exception as exc:
        _record_vision_metric(st_api.session_state, "yolo_detection",
                              (time.perf_counter() - started) * 1000, "failed")
        st_api.session_state[error_key] = str(exc)
        _write_analysis_status(progress, f"YOLO 소자 탐지 · 실패 · {exc}")
        return None, str(exc)


def _run_selected_classification(st_api: Any, *, image_bytes: bytes, scope: str, rows: list[dict],
                                qwen_key: str, selected_component_ids: list[str],
                                progress: Any = None) -> None:
    """Classify only user-selected boxes; do not run OCR or duplicate analysis."""
    from circuitcare.vision.ai_review_service import DEFAULT_REVIEW_MODEL, review_components_with_qwen

    selected_ids = {str(value) for value in selected_component_ids}
    targets = [row for row in rows if str(row.get("component_id")) in selected_ids
               and row.get("상태") != "제외"]
    if not targets:
        raise ValueError("선택한 사진에서 분류할 수 있는 활성 박스를 찾지 못했습니다.")
    detections = _selected_classification_detections(rows, selected_ids)
    fingerprint = _selected_classification_fingerprint(image_bytes, detections, selected_ids)
    batch_cache_key = f"cc_component_qwen_batches_{scope}"
    batch_cache = st_api.session_state.setdefault(batch_cache_key, {})
    cached = batch_cache.get(fingerprint)
    if isinstance(cached, dict) and cached.get("status") == "complete":
        partial = cached["result"]
        _write_analysis_status(progress, "선택 소자 분류 · 캐시 결과 재사용")
    else:
        if progress is not None and callable(getattr(progress, "set_stage", None)):
            progress.set_stage("QWEN_CLASS_REVIEW", f"Qwen 선택 소자 분류 · {len(targets)}개 박스",
                               completed_batches=0, total_batches=1, current_batch=1)
        client = _ollama_review_client()
        vision_check = getattr(client, "model_supports_vision", None)
        if callable(vision_check) and not vision_check(DEFAULT_REVIEW_MODEL):
            raise RuntimeError(f"{DEFAULT_REVIEW_MODEL} 모델이 Ollama에서 vision 기능을 선언하지 않았습니다.")
        started = time.perf_counter()
        partial = review_components_with_qwen(
            client, DEFAULT_REVIEW_MODEL, image_bytes, detections, [],
            classification_only=True, include_board_context=False,
        )
        batch_cache[fingerprint] = {"status": "complete", "result": partial}
        _write_analysis_status(progress, f"선택 소자 분류 · 완료 · {len(partial.get('class_reviews', []))}개 제안")
        if progress is not None and callable(getattr(progress, "set_stage", None)):
            progress.set_stage("RESULT_VALIDATION", "선택 소자 분류 결과 확인 중",
                               completed_batches=1, total_batches=1, current_batch=1)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)

    previous = st_api.session_state.get(qwen_key)
    review = deepcopy(previous) if (isinstance(previous, dict)
                                    and previous.get("review_mode") == "selected_class_only") else {
        "class_reviews": [], "duplicate_reviews": [], "group_reviews": [],
        "missed_component_candidates": [], "notes": [], "review_batches": [],
        "review_diagnostics": {},
    }
    by_id = {str(item.get("detection_id")): item for item in review.get("class_reviews", [])
             if isinstance(item, dict) and item.get("detection_id")}
    for proposal in partial.get("class_reviews", []):
        if isinstance(proposal, dict) and str(proposal.get("detection_id")) in selected_ids:
            by_id[str(proposal["detection_id"])] = proposal
    review.update({
        "class_reviews": list(by_id.values()),
        "duplicate_reviews": [], "group_reviews": [], "missed_component_candidates": [],
        "model": partial.get("model", DEFAULT_REVIEW_MODEL),
        "review_mode": "selected_class_only",
        "selected_component_ids": sorted(selected_ids),
        "selected_source_fingerprint": fingerprint,
        "review_diagnostics": {"selected_classification": {"status": "complete"}},
        "verification_status": "AI_SUGGESTIONS_ONLY",
    })
    review.setdefault("review_batches", []).append({
        "mode": "selected_class_only", "detection_ids": sorted(selected_ids),
        "status": "complete", "image_count": partial.get("review_image_count", len(targets)),
        "duration_ms": partial.get("ollama_metrics", {}).get("total_duration_ns", 0) / 1_000_000,
        "ollama_metrics": partial.get("ollama_metrics", {}),
    })
    review["review_batches"] = review["review_batches"][-64:]
    review["ollama_metrics"] = partial.get("ollama_metrics", {})
    review["source_fingerprint"] = _inference_fingerprint(rows, None)
    review["performance"] = {
        "batch_size": len(targets), "class_review_batches": 1,
        "class_review_requests": 0 if isinstance(cached, dict) else 1,
        "class_batch_cache_hits": 1 if isinstance(cached, dict) else 0,
        "qwen_image_count": partial.get("review_image_count", len(targets)),
        "qwen_native_tile_count": 0,
        "output_tokens_per_second": None,
        "duration_ms": duration_ms if not isinstance(cached, dict) else 0,
    }
    st_api.session_state[qwen_key] = review


def _selected_classification_fingerprint(image_bytes: bytes, detections: list[dict],
                                         selected_ids: set[str] | list[str]) -> str:
    """Fingerprint only the selected YOLO observations and image, excluding OCR/UI state."""
    from circuitcare.vision.ai_review_service import DEFAULT_REVIEW_MODEL

    return hashlib.sha256(json.dumps({
        "image": hashlib.sha256(image_bytes).hexdigest(),
        "selected": sorted(map(str, selected_ids)), "detections": detections,
        "model": DEFAULT_REVIEW_MODEL, "mode": "selected-class-only-v1",
    }, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def _selected_classification_detections(rows: list[dict], selected_ids: set[str] | list[str]) -> list[dict]:
    selected = {str(value) for value in selected_ids}
    return [{"component_id": str(row["component_id"]),
             "class_name": str(row.get("소자 종류") or "unknown"),
             "confidence": row.get("신뢰도"),
             "bbox_xyxy": [row["x1"], row["y1"], row["x2"], row["y2"]],
             "ocr_candidates": row.get("ocr_candidates", []),
             "ocr_class_hints": row.get("ocr_class_hints", []),
             "reference_designators": row.get("reference_designators", []),
             "body_marking": row.get("body_marking") or (
                 row.get("marking_ocr", {}).get("marking")
                 if isinstance(row.get("marking_ocr"), dict) else None),
             "marking_candidates": row.get("marking_candidates", row.get("body_marking_candidates", [])),
             "possible_part_number": row.get("possible_part_number")}
            for row in rows if str(row.get("component_id")) in selected
            and row.get("상태") != "제외"]


def _run_ai_review_workflow(st_api: Any, *, image_bytes: bytes, scope: str, rows: list[dict],
                            records_key: str, ocr_key: str, qwen_key: str,
                            progress: Any = None,
                            selected_component_ids: list[str] | None = None) -> None:
    """Classify selected YOLO boxes only; OCR and duplicate review are separate actions."""
    selected_ids = sorted({str(value) for value in (selected_component_ids or [])})
    if not selected_ids:
        raise ValueError("Qwen 종류 분류 전 PCB에서 소자 박스를 선택해야 합니다.")
    _run_selected_classification(st_api, image_bytes=image_bytes, scope=scope, rows=rows,
                                 qwen_key=qwen_key, selected_component_ids=selected_ids,
                                 progress=progress)


def ensure_ai_component_review(st_api: Any, *, image_bytes: bytes, scope: str,
                                rows: list[dict], records_key: str,
                                progress: Any = None,
                                selected_component_ids: list[str] | None = None) -> str:
    """Run Qwen classification only for explicitly selected component boxes."""
    ocr_key, qwen_key = review_aux_keys(scope)
    selected_ids = sorted({str(value) for value in (selected_component_ids or [])})
    if not selected_ids:
        _write_analysis_status(progress, "Qwen 분류 대기 · PCB에서 소자 박스를 먼저 선택하세요")
        return "selection_required"
    detections = _selected_classification_detections(rows, selected_ids)
    if len(detections) != len(selected_ids):
        return "stale_selection"
    request_fingerprint = _selected_classification_fingerprint(image_bytes, detections, selected_ids)
    run_key = f"cc_ai_review_selected_run_{scope}_{hashlib.sha256('|'.join(selected_ids).encode()).hexdigest()[:12]}"
    cached = st_api.session_state.get(qwen_key)
    if (isinstance(cached, dict) and cached.get("review_mode") == "selected_class_only"
            and cached.get("selected_source_fingerprint") == request_fingerprint):
        _write_analysis_status(progress, "선택 소자 분류 · 결과 재사용")
        return "complete"
    if st_api.session_state.get(run_key) == request_fingerprint:
        _write_analysis_status(progress, "선택 소자 분류 · 이전 결과 재사용")
        return "complete" if isinstance(cached, dict) else "unavailable"
    _run_ai_review_workflow(st_api, image_bytes=image_bytes, scope=scope, rows=rows,
                            records_key=records_key, ocr_key=ocr_key, qwen_key=qwen_key,
                            progress=progress, selected_component_ids=selected_ids)
    result = st_api.session_state.get(qwen_key)
    if isinstance(result, dict):
        st_api.session_state[run_key] = request_fingerprint
        return "failed" if result.get("verification_status") in {
            "AI_REVIEW_FAILED", "AI_REVIEW_PARTIAL"} else "complete"
    return "unavailable"


def queue_ai_component_review(st_api: Any, *, image_bytes: bytes, scope: str,
                              rows: list[dict], records_key: str,
                              owner: str | None = None,
                              selected_component_ids: list[str] | None = None,
                              retry_failed: bool = False) -> tuple[str | None, str, str]:
    """Queue selected-box Qwen classification using an immutable row snapshot."""
    import secrets

    state = st_api.session_state
    session_owner_key = "cc_ai_review_session_owner"
    token = state.setdefault(session_owner_key, secrets.token_urlsafe(20))
    project_id = str(state.get("circuitcare_v2_current_project_id", "unknown-project"))
    job_owner = owner or f"{token}|{project_id}"
    selected_ids = sorted({str(value) for value in (selected_component_ids or [])})
    if not selected_ids:
        return None, job_owner, "SELECT_COMPONENT_REQUIRED"
    known_ids = {str(row.get("component_id")) for row in rows
                 if isinstance(row, dict) and row.get("상태") != "제외"}
    if not set(selected_ids) <= known_ids:
        return None, job_owner, "STALE_SELECTION"
    fingerprint = _observation_fingerprint(rows)
    ocr_key, qwen_key = review_aux_keys(scope)
    cached = state.get(qwen_key)
    selected_detections = _selected_classification_detections(rows, selected_ids)
    selected_fingerprint = _selected_classification_fingerprint(image_bytes, selected_detections, selected_ids)
    if (isinstance(cached, dict) and cached.get("review_mode") == "selected_class_only"
            and cached.get("selected_source_fingerprint") == selected_fingerprint
            and cached.get("verification_status") not in {"AI_REVIEW_FAILED", "AI_REVIEW_PARTIAL"}):
        state[f"cc_ai_review_job_status_{scope}"] = "COMPLETE"
        return None, "", "COMPLETE"
    variant = "selected_class_only:" + ",".join(selected_ids)
    previous_job_id = state.get(f"cc_ai_review_job_{scope}")
    if (not retry_failed and previous_job_id
            and state.get(f"cc_ai_review_job_owner_{scope}") == job_owner
            and state.get(f"cc_ai_review_job_fingerprint_{scope}") == fingerprint
            and state.get(f"cc_ai_review_job_variant_{scope}") == variant):
        existing = get_ai_review_job(owner=job_owner, job_id=previous_job_id)
        if existing is not None:
            # Preserve the original submission versions and avoid copying OCR
            # and batch caches just to discover that the job already exists.
            status = str(existing.get("status", "FAILED"))
            if state.get(f"cc_ai_review_job_status_{scope}") == "STALE":
                status = "STALE"
            state[f"cc_ai_review_job_status_{scope}"] = status
            return previous_job_id, job_owner, status
    review_fingerprint = _review_fingerprint(rows, None)
    ocr_fingerprint = _ocr_evidence_fingerprint(state.get(ocr_key))
    allowed = (records_key, ocr_key, qwen_key, f"cc_component_marking_ocr_{scope}",
               f"cc_component_qwen_batches_{scope}", f"cc_component_qwen_pair_queue_{scope}")
    worker_state = {key: deepcopy(state[key]) for key in allowed if key in state}
    worker_state.pop(f"cc_ai_review_run_{scope}", None)
    if retry_failed:
        # Keep successful batch/OCR caches, but force the worker past a failed or
        # partial top-level result so only failed batches are retried.
        worker_state.pop(qwen_key, None)
    job_id, status = submit_ai_review(
        owner=job_owner, scope=scope, fingerprint=fingerprint, image_bytes=image_bytes,
        rows=rows, records_key=records_key, state_snapshot=worker_state,
        user_version=_review_user_version(rows), review_fingerprint=review_fingerprint,
        selected_component_ids=selected_ids,
        retry_failed=retry_failed,
    )
    if job_id:
        state[f"cc_ai_review_job_{scope}"] = job_id
        state[f"cc_ai_review_job_owner_{scope}"] = job_owner
        state[f"cc_ai_review_job_fingerprint_{scope}"] = fingerprint
        state[f"cc_ai_review_job_variant_{scope}"] = variant
        if job_id != previous_job_id:
            state[f"cc_ai_review_job_review_fingerprint_{scope}"] = review_fingerprint
            state[f"cc_ai_review_job_ocr_fingerprint_{scope}"] = ocr_fingerprint
        state[f"cc_ai_review_job_status_{scope}"] = status
    return job_id, job_owner, status


def continue_duplicate_pair_review(st_api: Any, *, image_bytes: bytes, scope: str,
                                   rows: list[dict], ocr_result: dict | None = None,
                                   progress: Any = None) -> dict[str, Any]:
    """Review the next bounded duplicate-pair page without rerunning OCR or class review."""
    from circuitcare.vision.ai_review_service import DEFAULT_REVIEW_MODEL, review_components_with_qwen

    image_fingerprint = hashlib.sha256(image_bytes).hexdigest()
    indexed = [dict(row, detection_id=str(row.get("component_id") or row.get("detection_id")))
               for row in rows if row.get("상태") != "제외"]
    candidates = _duplicate_pair_candidates(indexed, limit=None)
    queue_key = f"cc_component_qwen_pair_queue_{scope}"
    queue = _sync_duplicate_pair_queue(candidates, st_api.session_state.get(queue_key),
                                       image_fingerprint=image_fingerprint,
                                       model=DEFAULT_REVIEW_MODEL)
    records = queue["pairs"]
    pending = _claim_duplicate_pair_page(candidates, queue, limit=12)
    if not pending:
        st_api.session_state[queue_key] = queue
        return _duplicate_pair_queue_summary(queue)

    status = progress or st_api.status("Qwen 중복 후보 검토 중", expanded=True)
    status.write(f"중복 후보 추가 배치 처리 중 · {len(pending)}쌍")
    client_error = None
    client = None
    try:
        client = _ollama_review_client()
        vision_check = getattr(client, "model_supports_vision", None)
        if callable(vision_check) and not vision_check(DEFAULT_REVIEW_MODEL):
            raise RuntimeError(f"{DEFAULT_REVIEW_MODEL} 모델이 Ollama에서 vision 기능을 선언하지 않았습니다.")
    except Exception as exc:
        client_error = {"stage": getattr(exc, "stage", "qwen"), "error": str(exc)[:500]}
    if client_error:
        reviewed_pairs, failed_pairs = {}, {_duplicate_pair_id(pair): client_error for pair in pending}
    else:
        reviewed_pairs, failed_pairs = _review_duplicate_pair_page(
            client, image_bytes=image_bytes, detections=indexed, pairs=pending,
            ocr_records=(ocr_result or {}).get("records", []), progress=status,
        )
    for pair in pending:
        pair_id = _duplicate_pair_id(pair)
        if pair_id in reviewed_pairs:
            records[pair_id].update({"status": "REVIEWED", "proposal": reviewed_pairs[pair_id],
                                     "error": None, "error_stage": None})
        else:
            failure = failed_pairs.get(pair_id, {"stage": "qwen", "error": "Qwen pair review failed."})
            records[pair_id].update({"status": "FAILED", "error": failure["error"],
                                     "error_stage": failure["stage"]})
    if failed_pairs:
        status.write(f"중복 후보 검토 실패 · {len(failed_pairs)}쌍 · 실패 항목은 재시도할 수 있습니다.")
    st_api.session_state[queue_key] = queue
    summary = _duplicate_pair_queue_summary(queue)
    qwen_key = review_aux_keys(scope)[1]
    review = st_api.session_state.get(qwen_key)
    if isinstance(review, dict):
        valid_pair_ids = set(queue["candidate_ids"])
        previous = [proposal for proposal in review.get("duplicate_reviews", [])
                    if _duplicate_pair_id({"detections": tuple({"detection_id": item}
                        for item in proposal.get("detection_ids", []))}) in valid_pair_ids]
        merged = {"|".join(sorted(map(str, proposal.get("detection_ids", [])))): proposal
                  for proposal in previous}
        for pair_id in queue["candidate_ids"]:
            proposal = records[pair_id].get("proposal")
            if records[pair_id].get("status") == "REVIEWED" and isinstance(proposal, dict):
                merged[pair_id] = proposal
        review["duplicate_reviews"] = list(merged.values())
        review.setdefault("review_diagnostics", {})["cross_batch_duplicate_search"] = {
            "status": "complete" if summary["complete"] else "failed" if summary["failed"] else "pending",
            "candidate_count": summary["total"],
            "reviewed_count": summary["reviewed"], "pending_count": summary["pending"],
            "running_count": summary["running"], "failed_count": summary["failed"],
            "deferred_count": summary["deferred"], "stale_count": summary["stale"],
            "confirmed_duplicate_count": summary["confirmed_duplicate"],
            "complete": summary["complete"],
            "failed_items": [{"pair_id": pair_id, "stage": records[pair_id].get("error_stage", "qwen"),
                              "error": records[pair_id].get("error")}
                             for pair_id in queue["candidate_ids"]
                             if records[pair_id].get("status") == "FAILED"][:12],
        }
        pair_failure_batch = {"batch": "board-wide duplicate pairs", "detection_ids": [],
                              "status": "failed", "stage": "qwen",
                              "error": "Qwen 중복 후보 응답을 검증하지 못했습니다."}
        existing_failed = [row for row in review.get("failed_batches", [])
                           if row.get("batch") != "board-wide duplicate pairs"]
        if summary["failed"]:
            first_failed = next((records[pair_id] for pair_id in queue["candidate_ids"]
                                 if records[pair_id].get("status") == "FAILED"), {})
            pair_failure_batch.update({"stage": first_failed.get("error_stage", "qwen"),
                                       "error": first_failed.get("error") or pair_failure_batch["error"]})
            review["failed_batches"] = [*existing_failed, pair_failure_batch]
            review["verification_status"] = "AI_REVIEW_PARTIAL"
        else:
            review["failed_batches"] = existing_failed
            review["verification_status"] = "AI_REVIEW_PARTIAL" if existing_failed else "AI_SUGGESTIONS_ONLY"
        st_api.session_state[qwen_key] = review
    if summary["failed"]:
        status.update(label="중복 후보 일부 검토 실패 · 실패 항목을 다시 시도할 수 있습니다.",
                      state="error", expanded=True)
    elif summary["complete"]:
        status.update(label=f"중복 후보 검토 완료 · {summary['reviewed']}/{summary['total']}쌍",
                      state="complete", expanded=False)
    else:
        status.update(label=(f"중복 후보 추가 배치 완료 · 검토 {summary['reviewed']}/{summary['total']}쌍 · "
                             f"대기 {summary['pending'] + summary['deferred']}쌍"),
                      state="complete", expanded=False)
    return summary


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
    selection = st_api.session_state.get(SELECTION_KEY)
    selected_id = str(getattr(selection, "selected_component", None) or "")
    selected_row = next((row for row in rows if str(row.get("component_id") or "") == selected_id
                         and row.get("상태") != "제외"), None)
    if selected_row:
        st_api.caption(f"Qwen 분석 대상 · {selected_id} · {selected_row.get('소자 종류', 'unknown')}")
        st_api.caption("선택 박스와 주변 실크스크린·확보된 OCR 후보를 함께 참고해 종류를 추정합니다. 이미지 해석은 오래 걸릴 수 있으며 결과는 사용자 확인 전까지 미확정입니다.")
    else:
        st_api.info("PCB에서 소자 박스를 선택하면 해당 박스의 종류만 Qwen으로 확인할 수 있습니다.")
    if st_api.button("선택 박스 주변 문맥으로 종류 추정", key=f"cc_component_ai_review_{scope}", type="primary",
                     disabled=selected_row is None):
        token = st_api.session_state.setdefault("cc_ai_review_session_owner", os.urandom(20).hex())
        queue_ai_component_review(
            st_api, image_bytes=image_bytes, scope=scope, rows=rows,
            records_key=records_key, owner=f"{token}|{project_id}",
            selected_component_ids=[selected_id], retry_failed=True,
        )
        st_api.rerun(scope="fragment")
    with st_api.expander("개별 실행 · 고급", expanded=False):
        st_api.caption("OCR는 별도 일괄 실행합니다. Qwen은 선택한 박스의 종류만 제안하며, 확정·제외는 사용자가 결정합니다.")
        if st_api.button("PCB OCR + 각인 OCR 일괄 실행", key=f"cc_component_run_ocr_{scope}"):
            try:
                from PIL import Image
                from circuitcare.vision.ocr_backend import run_local_ocr, run_component_marking_ocr_batch

                image_fingerprint = hashlib.sha256(image_bytes).hexdigest()
                status = st_api.status("RapidOCR 일괄 처리 준비 중", expanded=True)
                with Image.open(io.BytesIO(image_bytes)) as source_image:
                    source_image = source_image.convert("RGB")
                    ocr_result = st_api.session_state.get(ocr_key)
                    if (not isinstance(ocr_result, dict)
                            or ocr_result.get("source_image_fingerprint") != image_fingerprint):
                        status.write("PCB 전체 실크스크린 OCR · 진행 중")
                        ocr_result = run_local_ocr(source_image)
                        ocr_result["source_image_fingerprint"] = image_fingerprint
                    else:
                        status.write("PCB 전체 OCR · 기존 결과 재사용")
                associated = associate_ocr_with_components([
                    {**row, "bbox_xyxy": [row["x1"], row["y1"], row["x2"], row["y2"]]}
                    for row in rows
                ], ocr_result.get("records", []))
                for old_row, evidence in zip(rows, associated):
                    for key in ("ocr_records", "ocr_candidates", "reference_designators", "ocr_class_hints", "ocr_association_status"):
                        old_row[key] = evidence.get(key)
                eligible = _marking_ocr_targets(rows, limit=len(rows))
                marking_cache_key = f"cc_component_marking_ocr_{scope}"
                marking_cache = st_api.session_state.setdefault(marking_cache_key, {})
                targets = []
                fingerprints = {}
                for row in eligible:
                    marking_fingerprint = hashlib.sha256(json.dumps({
                        "photo": image_fingerprint, "id": row.get("component_id"),
                        "box": [row.get(key) for key in ("x1", "y1", "x2", "y2")],
                        "class": row.get("소자 종류"), "engine": "rapidocr-marking-batch-v1",
                    }, sort_keys=True).encode()).hexdigest()
                    fingerprints[str(row.get("component_id"))] = marking_fingerprint
                    cached = marking_cache.get(marking_fingerprint)
                    if isinstance(cached, dict):
                        row["marking_ocr"] = cached
                        if cached.get("marking"):
                            row["body_marking"] = cached["marking"]
                            row["marking_candidates"] = [item.get("text") for item in cached.get("candidates", [])
                                                          if isinstance(item, dict) and item.get("text")]
                            row["ocr_source"] = "RapidOCR"
                            candidates = cached.get("candidates", [])
                            row["ocr_score"] = (candidates[0].get("ocr_score", candidates[0].get("confidence"))
                                                 if candidates and isinstance(candidates[0], dict) else None)
                        if cached.get("inferred_value"):
                            row["ocr_inferred_value"] = cached["inferred_value"]
                            row["ocr_inference_method"] = cached.get("inference_method")
                        row["marking_ocr_status"] = cached.get("status", "REVIEWED")
                    else:
                        targets.append({"component_id": str(row.get("component_id")),
                                        "component_type": str(row.get("소자 종류") or "unknown"),
                                        "bbox_xyxy": [row["x1"], row["y1"], row["x2"], row["y2"]]})
                if targets:
                    status.write(f"소자 각인 OCR · {len(targets)}개 박스를 4개씩 묶어 처리합니다")
                    with Image.open(io.BytesIO(image_bytes)) as source_image:
                        batch_results = run_component_marking_ocr_batch(
                            source_image.convert("RGB"), targets, batch_size=4,
                            on_batch=lambda done, total: status.update(
                                label=f"소자 각인 OCR · {done}/{total} 묶음 완료",
                                state="running", expanded=True),
                        )
                    row_by_id = {str(row.get("component_id")): row for row in rows}
                    for component_id, result in batch_results.items():
                        target = row_by_id.get(component_id)
                        if target is None:
                            continue
                        marking_cache[fingerprints[component_id]] = result
                        target["marking_ocr"] = result
                        target["marking_ocr_fingerprint"] = fingerprints[component_id]
                        target["marking_ocr_status"] = result.get("status", "REVIEWED")
                        if result.get("marking"):
                            target["body_marking"] = result["marking"]
                            target["marking_candidates"] = [item.get("text") for item in result.get("candidates", [])]
                            candidates = result.get("candidates", [])
                            target["ocr_score"] = (candidates[0].get("ocr_score", candidates[0].get("confidence"))
                                                    if candidates and isinstance(candidates[0], dict) else None)
                            target["ocr_source"] = "RapidOCR"
                            if result.get("inferred_value"):
                                target["ocr_inferred_value"] = result["inferred_value"]
                                target["ocr_inference_method"] = result.get("inference_method")
                ocr_result["source_image_fingerprint"] = image_fingerprint
                st_api.session_state[ocr_key] = ocr_result
                while len(marking_cache) > 256:
                    marking_cache.pop(next(iter(marking_cache)))
                st_api.session_state[records_key] = rows
                status.update(label="RapidOCR 일괄 처리 완료", state="complete", expanded=False)
                st_api.rerun(scope="fragment")
            except Exception as exc:
                from circuitcare.vision.ocr_backend import OCRUnavailable

                if isinstance(exc, (OCRUnavailable, ImportError)):
                    st_api.warning(f"RapidOCR를 사용할 수 없습니다 · {exc}")
                else:
                    st_api.warning(f"실크스크린 OCR 실패 · {exc}")
        ocr_result = st_api.session_state.get(ocr_key)
        if isinstance(ocr_result, dict):
            ocr_rows = ocr_result.get("records", [])
            marking_pending = sum(row.get("상태") != "제외"
                                  and row.get("marking_ocr_status") == "PENDING_PRIORITY" for row in rows)
            marking_failed = sum(row.get("상태") != "제외"
                                 and row.get("marking_ocr_status") == "OCR_FAILED" for row in rows)
            st_api.caption(f"소자 본체 각인 OCR · 상위 12개 우선, 이후 12개씩 추가 검수 · 미검수 대기 {marking_pending} · 실패 {marking_failed}")
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

        qwen_review = st_api.session_state.get(qwen_key)
        if isinstance(qwen_review, dict):
            selected_ids = sorted({str(value) for value in qwen_review.get("selected_component_ids", [])})
            selected_detections = _selected_classification_detections(rows, selected_ids)
            expected_fingerprint = (_selected_classification_fingerprint(
                image_bytes, selected_detections, selected_ids) if selected_ids else None)
            if (qwen_review.get("review_mode") != "selected_class_only"
                    or qwen_review.get("selected_source_fingerprint") != expected_fingerprint):
                invalidate_qwen_review(st_api.session_state, scope)
                qwen_review = None
        if isinstance(qwen_review, dict):
            selected_ids = set(map(str, qwen_review.get("selected_component_ids", [])))
            st_api.caption("선택 소자 종류 검수 · OCR 및 중복 검수는 실행하지 않았습니다.")
            if qwen_review.get("verification_status") == "AI_REVIEW_FAILED":
                diagnostics = qwen_review.get("review_diagnostics", {}).get("selected_classification", {})
                st_api.error(f"Qwen 분류 실패 · {diagnostics.get('error') or '요청을 완료하지 못했습니다.'}")
            performance = qwen_review.get("performance") if isinstance(qwen_review.get("performance"), dict) else {}
            metrics = qwen_review.get("ollama_metrics") if isinstance(qwen_review.get("ollama_metrics"), dict) else {}
            if performance:
                total_s = (metrics.get("total_duration_ns") or 0) / 1e9
                out_tokens = metrics.get("eval_count") or 0
                out_duration = metrics.get("eval_duration_ns") or 0
                rate = out_tokens * 1e9 / out_duration if out_duration else 0
                st_api.caption(f"Qwen · {total_s:.1f}초 · {out_tokens} 출력 토큰 · {rate:.1f} 토큰/초")
            proposals = [proposal for proposal in qwen_review.get("class_reviews", [])
                         if str(proposal.get("detection_id")) in selected_ids]
            if not proposals:
                st_api.info("선택 박스에 대한 분류 제안이 없습니다.")
            by_id = {str(row.get("component_id")): row for row in rows}
            for index, proposal in enumerate(proposals):
                target = by_id.get(str(proposal.get("detection_id")))
                if target is None or target.get("상태") == "제외":
                    continue
                base_class = proposal.get("proposed_base_class", proposal.get("proposed_class", "unknown"))
                title = f"{target['component_id']} · YOLO {target.get('소자 종류', 'unknown')} → Qwen {base_class}"
                if proposal.get("proposed_subtype"):
                    title += f" / {proposal['proposed_subtype']}"
                st_api.markdown(f"**{title}** · {proposal.get('confidence', 0) * 100:.0f}/100 · 사용자 확인 필요")
                st_api.caption(proposal.get("visual_evidence") or proposal.get("evidence") or "시각 근거 없음")
                if target.get("상태") == "확인" or target.get("user_edited"):
                    st_api.info("사용자가 확정하거나 수정한 박스입니다. AI 제안은 자동 반영되지 않습니다.")
                    continue
                proposed_class = proposal.get("proposed_class")
                if proposed_class == "not_component":
                    if st_api.button("오탐으로 제외 확인", key=f"cc_qwen_false_positive_{scope}_{index}"):
                        record_box_exclusion(st_api.session_state, scope=scope, row=target,
                                             reason="qwen_not_component", review=proposal)
                        st_api.session_state[records_key] = rows
                        invalidate_review_state()
                        st_api.rerun(scope="fragment")
                elif proposed_class in CLASS_OPTIONS:
                    if st_api.button("종류 제안 반영 · 검수 필요 상태 유지",
                                     key=f"cc_qwen_class_apply_{scope}_{index}"):
                        target.update({
                            "qwen_component_type_candidate": proposed_class,
                            "qwen_component_subtype_candidate": proposal.get("proposed_subtype"),
                            "qwen_component_type_alternatives": proposal.get("alternative_classes", []),
                            "qwen_component_type_evidence": proposal.get("visual_evidence") or proposal.get("evidence", ""),
                            "qwen_component_type_confidence": proposal.get("confidence"),
                            "qwen_component_type_status": "AI_PROPOSED_UNVERIFIED",
                            "qwen_review_status": proposal.get("review_status", "AI_SUGGESTED"),
                            "qwen_visual_evidence": proposal.get("visual_evidence") or proposal.get("evidence", ""),
                            "ai_review": proposal,
                        })
                        st_api.session_state[records_key] = rows
                        invalidate_review_state()
                        st_api.rerun(scope="fragment")

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
