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
from circuitcare.vision.ai_review_jobs import submit_ai_review
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


def _qwen_review_batch_size() -> int:
    try:
        value = int(os.environ.get("CIRCUITCARE_QWEN_BATCH_SIZE", "6"))
    except (TypeError, ValueError):
        return 6
    return value if value in {2, 4, 6, 8} else 6


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
    return OllamaClient(timeout_seconds=300, num_predict=8192)


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


def _inference_fingerprint(rows: list[dict], ocr_result: dict | None = None) -> str:
    """Fingerprint only the visual/model inputs; review actions don't rerun vision."""
    snapshot = [{key: row.get(key) for key in (
        "component_id", "detector_class", "소자 종류", "x1", "y1", "x2", "y2", "신뢰도",
        "ocr_candidates", "reference_designators", "ocr_class_hints", "body_marking",
        "marking_candidates", "marking_ocr_fingerprint", "possible_part_number",
        "ocr_source", "ocr_score", "qwen_marking_candidates", "qwen_part_number_candidates",
        "qwen_ocr_evidence", "qwen_conflicting_evidence",
    )} for row in rows]
    payload = json.dumps({"rows": snapshot, "ocr": (ocr_result or {}).get("records", [])},
                         ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


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
    """Full review snapshot fingerprint, including user decisions and identity links."""
    payload = {"inference": _inference_fingerprint(rows, ocr_result),
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

        _write_analysis_status(progress, "소자 각인 OCR · 진행 중")
        image_fingerprint = hashlib.sha256(image_bytes).hexdigest()
        marking_cache_key = f"cc_component_marking_ocr_{scope}"
        marking_cache = st_api.session_state.setdefault(marking_cache_key, {})
        # Prioritize the existing top twelve, then continue through later OCR
        # batches so a large board never leaves the remaining markings silently
        # unreviewed. Each target is independently fingerprint-cached.
        eligible = _marking_ocr_targets(rows, limit=len(rows))
        eligible_ids = {str(row.get("component_id")) for row in eligible}
        for row in rows:
            if row.get("상태") == "제외":
                continue
            if str(row.get("component_id")) not in eligible_ids:
                row.setdefault("marking_ocr_status", "PENDING_PRIORITY")
        with Image.open(io.BytesIO(image_bytes)) as source_image:
            for target_index, row in enumerate(eligible):
                row["marking_ocr_batch"] = target_index // 12 + 1
                marking_fingerprint = hashlib.sha256(json.dumps({
                    "photo": image_fingerprint, "id": row.get("component_id"),
                    "box": [row.get(key) for key in ("x1", "y1", "x2", "y2")],
                    "class": row.get("소자 종류"), "engine": "rapidocr-marking-v1", "version": 2,
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
                row["marking_ocr_status"] = result.get("status", "REVIEWED")
                if result.get("marking"):
                    row["body_marking"] = result["marking"]
                    row["marking_candidates"] = [item.get("text") for item in result.get("candidates", [])
                                                  if item.get("text")]
                    row["ocr_source"] = "RapidOCR"
                    candidates = result.get("candidates", [])
                    row["ocr_score"] = (candidates[0].get("confidence") if candidates
                                         and isinstance(candidates[0], dict) else None)
                    if row.get("소자 종류") == "ic" and not row.get("confirmed_part_number"):
                        row["possible_part_number"] = result["marking"]
                        row["part_identity_status"] = "UNVERIFIED_PART_ID"
                        row["part_number_status"] = "UNVERIFIED_PART_ID"
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
            requested_batch_size = _qwen_review_batch_size()
            batches = _review_detection_batches(detections, batch_size=requested_batch_size)
            failed_batches = []
            batch_cache_hits = 0
            qwen_requests = 0
            for batch_index, batch in enumerate(batches, start=1):
                _write_analysis_status(progress, f"Qwen 소자 검수 · {batch_index}/{len(batches)} 배치 진행 중")
                batch_fingerprint = hashlib.sha256(json.dumps({
                    "image": image_fingerprint, "detections": batch,
                    "ocr": (ocr_result or {}).get("records", []), "model": DEFAULT_REVIEW_MODEL,
                    "schema": "qwen-review-v2", "input_version": 2,
                }, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
                cached_batch = batch_cache.get(batch_fingerprint)
                if isinstance(cached_batch, dict) and cached_batch.get("status") == "complete":
                    batch_cache_hits += 1
                    partial = cached_batch["result"]
                else:
                    try:
                        qwen_requests += 1
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
                    "duplicate_pair_crop_count": partial.get("duplicate_pair_crop_count", 0),
                    "ollama_metrics": partial.get("ollama_metrics", {})})
                stage_update = getattr(progress, "set_stage", None)
                if callable(stage_update):
                    stage_update("QWEN_CLASS_REVIEW", f"Qwen 소자 검수 · {batch_index}/{len(batches)} 배치 완료",
                                 completed_batches=batch_index, total_batches=len(batches),
                                 current_batch=batch_index)
                review["review_diagnostics"][f"batch_{batch_index}"] = partial.get("review_diagnostics", {})
            # Search the complete active detection set before considering batch
            # boundaries. Cross-batch pairs receive their own focused paired crops.
            indexed_detections = [dict(row, detection_id=str(row.get("component_id") or row.get("detection_id")))
                                  for row in detections]
            all_pairs = _duplicate_pair_candidates(indexed_detections, limit=None)
            membership = {}
            for batch_index, batch in enumerate(batches):
                for row in batch:
                    membership[str(row.get("component_id") or row.get("detection_id"))] = batch_index
            batch_reviewed_pair_ids = {
                "|".join(sorted(map(str, row.get("detection_ids", []))))
                for row in review["duplicate_reviews"] if len(row.get("detection_ids", [])) == 2
            }
            pair_queue_key = f"cc_component_qwen_pair_queue_{scope}"
            pair_queue = _sync_duplicate_pair_queue(
                all_pairs, st_api.session_state.get(pair_queue_key),
                image_fingerprint=image_fingerprint, model=DEFAULT_REVIEW_MODEL,
            )
            for candidate in all_pairs:
                pair_id = _duplicate_pair_id(candidate)
                if pair_id in batch_reviewed_pair_ids:
                    record = pair_queue["pairs"].get(pair_id, {})
                    record.update({"status": "REVIEWED", "proposal": next((row for row in review["duplicate_reviews"]
                        if _duplicate_pair_id({"detections": tuple({"detection_id": item}
                                                                    for item in row.get("detection_ids", []))})
                        == pair_id), None), "error": None})
            pair_records = pair_queue["pairs"]
            pending_pairs = _claim_duplicate_pair_page(all_pairs, pair_queue, limit=12)
            pair_error = None
            if pending_pairs:
                reviewed_pairs, failed_pairs = _review_duplicate_pair_page(
                    ollama, image_bytes=image_bytes, detections=indexed_detections,
                    pairs=pending_pairs, ocr_records=(ocr_result or {}).get("records", []),
                    progress=progress,
                )
                for candidate in pending_pairs:
                    pair_id = _duplicate_pair_id(candidate)
                    record = pair_records[pair_id]
                    if pair_id in reviewed_pairs:
                        record.update({"status": "REVIEWED", "proposal": reviewed_pairs[pair_id],
                                       "error": None, "error_stage": None})
                    else:
                        failure = failed_pairs.get(pair_id, {"stage": "qwen", "error": "Qwen pair review failed."})
                        record.update({"status": "FAILED", "error": failure["error"],
                                       "error_stage": failure["stage"]})
                if failed_pairs:
                    pair_error = f"{len(failed_pairs)}개 중복 후보 검토 실패"
                    failed_batches.extend({"batch": "board-wide duplicate pairs", "detection_ids": pair_id.split("|"),
                        "status": "failed", "stage": failure["stage"], "error": failure["error"]}
                        for pair_id, failure in failed_pairs.items())
            st_api.session_state[pair_queue_key] = pair_queue
            pair_summary = _duplicate_pair_queue_summary(pair_queue)
            pair_reviews = [record["proposal"] for pair_id, record in pair_records.items()
                            if pair_id in pair_queue.get("candidate_ids", [])
                            and record.get("status") == "REVIEWED" and isinstance(record.get("proposal"), dict)]
            merged_duplicate_reviews = {"|".join(sorted(map(str, row.get("detection_ids", [])))): row
                                        for row in [*review["duplicate_reviews"], *pair_reviews]}
            review["duplicate_reviews"] = list(merged_duplicate_reviews.values())
            cross_batch_count = sum(
                membership.get(str(pair["detections"][0].get("detection_id")))
                != membership.get(str(pair["detections"][1].get("detection_id")))
                for pair in all_pairs
            )
            review["review_diagnostics"]["cross_batch_duplicate_search"] = {
                "status": ("failed" if pair_error or pair_summary["failed"] else
                           "complete" if pair_summary["complete"] else "pending"),
                "candidate_count": len(all_pairs), "cross_batch_candidate_count": cross_batch_count,
                "reviewed_count": pair_summary["reviewed"],
                "pending_count": pair_summary["pending"],
                "running_count": pair_summary["running"],
                "failed_count": pair_summary["failed"],
                "deferred_count": pair_summary["deferred"],
                "stale_count": pair_summary["stale"],
                "confirmed_duplicate_count": pair_summary["confirmed_duplicate"],
                "complete": pair_summary["complete"],
                "error": pair_error,
            }
            review["failed_batches"] = failed_batches
            _write_analysis_status(progress, "Qwen 중복 후보 검토 · 진행 중")
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
                                                            for row in review["review_batches"])
                                                      + pair_summary["reviewed"],
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
            metric_names = ("total_duration_ns", "load_duration_ns", "prompt_eval_count",
                            "prompt_eval_duration_ns", "eval_count", "eval_duration_ns")
            review["ollama_metrics"] = {name: sum(
                (row.get("ollama_metrics", {}).get(name) or 0) for row in review["review_batches"]
                if isinstance(row.get("ollama_metrics", {}).get(name), (int, float)))
                for name in metric_names}
            eval_count = review["ollama_metrics"].get("eval_count") or 0
            eval_duration_ns = review["ollama_metrics"].get("eval_duration_ns") or 0
            review["performance"] = {
                "batch_size": requested_batch_size,
                "class_review_batches": len(batches),
                "class_review_requests": qwen_requests,
                "class_batch_cache_hits": batch_cache_hits,
                "qwen_image_count": sum(int(row.get("image_count") or 0)
                                         for row in review["review_batches"]),
                "qwen_native_tile_count": review["review_tiles"],
                "output_tokens_per_second": (round(eval_count * 1e9 / eval_duration_ns, 3)
                                              if eval_duration_ns else None),
            }
        review["source_fingerprint"] = _inference_fingerprint(rows, ocr_result)
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
            "source_fingerprint": _inference_fingerprint(rows, ocr_result),
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
    fingerprint = _inference_fingerprint(rows, st_api.session_state.get(ocr_key))
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
        st_api.session_state[run_key] = _inference_fingerprint(
            rows, st_api.session_state.get(ocr_key))
        return "failed" if result.get("verification_status") in {
            "AI_REVIEW_FAILED", "AI_REVIEW_PARTIAL"} else "complete"
    return "unavailable"


def queue_ai_component_review(st_api: Any, *, image_bytes: bytes, scope: str,
                              rows: list[dict], records_key: str,
                              owner: str | None = None,
                              retry_failed: bool = False) -> tuple[str | None, str, str]:
    """Queue OCR/Qwen review using only an allowlisted snapshot of session data."""
    import secrets

    state = st_api.session_state
    session_owner_key = "cc_ai_review_session_owner"
    token = state.setdefault(session_owner_key, secrets.token_urlsafe(20))
    project_id = str(state.get("circuitcare_v2_current_project_id", "unknown-project"))
    job_owner = owner or f"{token}|{project_id}"
    fingerprint = _inference_fingerprint(rows, None)
    ocr_key, qwen_key = review_aux_keys(scope)
    review_fingerprint = _review_fingerprint(rows, state.get(ocr_key))
    cached = state.get(qwen_key)
    if (isinstance(cached, dict)
            and cached.get("verification_status") not in {"AI_REVIEW_FAILED", "AI_REVIEW_PARTIAL"}
            and cached.get("source_fingerprint") == _inference_fingerprint(rows, state.get(ocr_key))):
        return None, "", "COMPLETE"
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
        retry_failed=retry_failed,
    )
    if job_id:
        state[f"cc_ai_review_job_{scope}"] = job_id
        state[f"cc_ai_review_job_owner_{scope}"] = job_owner
        state[f"cc_ai_review_job_fingerprint_{scope}"] = fingerprint
        state[f"cc_ai_review_job_review_fingerprint_{scope}"] = review_fingerprint
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
    if st_api.button("AI 소자 검수", key=f"cc_component_ai_review_{scope}", type="primary"):
        token = st_api.session_state.setdefault("cc_ai_review_session_owner", os.urandom(20).hex())
        queue_ai_component_review(
            st_api, image_bytes=image_bytes, scope=scope, rows=rows,
            records_key=records_key, owner=f"{token}|{project_id}", retry_failed=True,
        )
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

        if st_api.button("Qwen3.5 9B로 검수 제안 생성", key=f"cc_component_run_qwen_{scope}"):
            st_api.session_state.pop(f"cc_ai_review_run_{scope}", None)
            st_api.session_state.pop(qwen_key, None)
            ensure_ai_component_review(st_api, image_bytes=image_bytes, scope=scope, rows=rows,
                                       records_key=records_key)

        qwen_review = st_api.session_state.get(qwen_key)
        if isinstance(qwen_review, dict) and qwen_review.get("source_fingerprint") != _inference_fingerprint(rows, ocr_result):
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
            performance = qwen_review.get("performance") if isinstance(qwen_review.get("performance"), dict) else {}
            ollama_metrics = qwen_review.get("ollama_metrics") if isinstance(qwen_review.get("ollama_metrics"), dict) else {}
            if performance:
                total_s = (ollama_metrics.get("total_duration_ns") or 0) / 1e9
                load_s = (ollama_metrics.get("load_duration_ns") or 0) / 1e9
                prompt_tokens = ollama_metrics.get("prompt_eval_count") or 0
                output_tokens = ollama_metrics.get("eval_count") or 0
                st_api.caption(
                    f"Qwen 실측 · 요청 {performance.get('class_review_requests', 0)}회 · "
                    f"입력 이미지 {performance.get('qwen_image_count', 0)}장 · "
                    f"Ollama {total_s:.1f}초 (모델 로드 {load_s:.2f}초) · "
                    f"프롬프트 {prompt_tokens:,} / 생성 {output_tokens:,} 토큰 · "
                    f"생성 {performance.get('output_tokens_per_second') or 0:.1f} 토큰/초"
                )
            cross_batch = diagnostics.get("cross_batch_duplicate_search", {})
            pair_queue = st_api.session_state.get(f"cc_component_qwen_pair_queue_{scope}", {})
            pair_summary = _duplicate_pair_queue_summary(pair_queue) if pair_queue.get("version") == 2 else {
                "total": int(cross_batch.get("candidate_count", 0)),
                "reviewed": int(cross_batch.get("reviewed_count", 0)),
                "pending": int(cross_batch.get("pending_count", 0)),
                "running": int(cross_batch.get("running_count", 0)),
                "failed": int(cross_batch.get("failed_count", 0)),
                "deferred": int(cross_batch.get("deferred_count", 0)),
                "confirmed_duplicate": int(cross_batch.get("confirmed_duplicate_count", 0)),
                "complete": bool(cross_batch.get("complete", False)),
            }
            if pair_summary["total"] or cross_batch.get("candidate_count"):
                pair_label = ("전체 중복 후보 검토 완료" if pair_summary["complete"] else
                              "중복 후보 검토 실패" if pair_summary["failed"] else "중복 후보 검토 진행 중")
                st_api.caption(f"{pair_label} · 후보 {pair_summary['total']}쌍 · 검토 완료 {pair_summary['reviewed']} · "
                               f"대기 {pair_summary['pending'] + pair_summary['deferred']} · 실패 {pair_summary['failed']} · "
                               f"중복 제안 {pair_summary['confirmed_duplicate']} · 배치 경계 {cross_batch.get('cross_batch_candidate_count', 0)}")
                for failed_pair in cross_batch.get("failed_items", [])[:4]:
                    st_api.caption(f"중복 후보 검토 실패 · {failed_pair.get('pair_id')} · "
                                   f"{failed_pair.get('stage')} · {failed_pair.get('error')}")
                remaining = pair_summary["pending"] + pair_summary["deferred"] + pair_summary["failed"]
                if remaining and st_api.button(
                        f"{'실패 포함 ' if pair_summary['failed'] else ''}남은 중복 후보 검토 · {remaining}쌍",
                        key=f"cc_component_continue_duplicate_review_{scope}"):
                    with st_api.status("Qwen 중복 후보 검토 중", expanded=True) as duplicate_progress:
                        continue_duplicate_pair_review(
                            st_api, image_bytes=image_bytes, scope=scope, rows=rows,
                            ocr_result=ocr_result, progress=duplicate_progress)
                    st_api.rerun()
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
                st_api.markdown(f"**{title}** · Qwen 판단 점수 {proposal['confidence'] * 100:.0f}/100 · {proposal.get('review_status', 'AI_SUGGESTED')} · 확인 필요")
                cleanup_status, cleanup_reason = qwen_cleanup_disposition(
                    proposal, protected=target.get("상태") == "확인" or bool(target.get("user_edited")),
                    has_conflict=bool(proposal.get("conflicting_evidence")),
                    is_group=bool(target.get("group_parent_id")),
                )
                st_api.caption(f"오탐 정리 상태 · {cleanup_status} · {cleanup_reason}")
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
                group_member_ids = set()
                for group in qwen_review.get("group_reviews", []):
                    group_member_ids.add(str(group.get("parent_id") or ""))
                    group_member_ids.update(str(value) for value in group.get("child_ids", []))
                is_group_pair = any(str(component_id) in group_member_ids for component_id in ids)
                has_conflict = any(bool(row.get("qwen_conflicting_evidence")
                                       or row.get("ocr_conflicts")
                                       or (row.get("ai_review") or {}).get("conflicting_evidence"))
                                   for row in pair_rows)
                cleanup_status, cleanup_reason = qwen_cleanup_disposition(
                    proposal, protected=protected, geometry_valid=geometric_duplicate,
                    has_conflict=has_conflict, is_group=is_group_pair,
                )
                st_api.caption(f"중복 정리 상태 · {cleanup_status} · {cleanup_reason}")
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
