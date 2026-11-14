"""Archive 2: isolated comparison of local vision-language models."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import statistics
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from circuitcare.component_taxonomy import (
    COMPONENT_CLASSES, QWEN_CLASSIFICATION_CLASSES, UNKNOWN_CLASS,
    normalize_qwen_class, normalize_qwen_subtype,
)
from circuitcare.local_llm import OllamaClient

MODEL_CATALOG = {
    "qwen35_9b": {"label": "Qwen3.5 9B · 기준 모델", "ollama_id": "qwen3.5:9b",
                  "mlx_id": "mlx-community/Qwen3.5-9B-4bit"},
    "qwen35_4b": {"label": "Qwen3.5 4B", "ollama_id": "qwen3.5:4b",
                  "mlx_id": "mlx-community/Qwen3.5-4B-4bit"},
    "lfm3_vl_2b": {"label": "LFM3-VL 2B", "ollama_id": "lfm3-vl:2b",
                   "mlx_id": "LiquidAI/LFM3-VL-2B-MLX-8bit"},
    "minicpm_v47_1b": {"label": "MiniCPM-V 4.7 1B", "ollama_id": "openbmb/minicpm-v4.7:1b",
                       "mlx_id": "mlx-community/MiniCPM-V-4.7-1B-8bit"},
}
CROP_SIZES = (384, 512, 720)
PADDING_RATIOS = (0.0, 0.25, 0.75)
OUTPUT_BUDGETS = (192, 256, 384)
FULL_BOARD_BATCH_SIZE = 10
MODEL_BATCH_SIZES = {"lfm3_vl_2b": 5, "minicpm_v47_1b": 5}
PROMPT_VERSION = "full-board-fast-classification-v2"
SCHEMA_VERSION = "full-board-fast-classification-schema-v2"
OUTPUT_NORMALIZATION_VERSION = "archive2-class-output-v3"


def _normalize_experiment_chat_http_response(response: dict[str, Any]) -> tuple[dict[str, Any], Any, list[str]]:
    """Wrap array-shaped VLM JSON before OllamaClient enforces an object root."""
    message = response.get("message")
    raw_text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(raw_text, str):
        return response, None, []
    candidate = raw_text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return response, raw_text, []
    if not isinstance(parsed, list):
        return response, parsed, []
    normalized = dict(response)
    normalized_message = dict(message)
    normalized_message["content"] = json.dumps({"class_reviews": parsed}, ensure_ascii=False)
    normalized["message"] = normalized_message
    return normalized, parsed, ["root JSON array wrapped as class_reviews"]


def _normalize_experiment_response(response: dict[str, Any], detection_id: str) -> dict[str, Any]:
    """Normalize VLM-specific response envelopes for Archive 2 only.

    The original object is retained for audit. Missing or unusable classifications
    become an explicit unknown result, never a guessed class.
    """
    content = response.get("content")
    response["archive2_raw_content"] = json.loads(json.dumps(content, ensure_ascii=False))
    notes: list[str] = []
    raw_rows: list[Any] = []
    if isinstance(content, dict):
        candidate = content.get("class_reviews")
        if isinstance(candidate, list):
            raw_rows = candidate
        elif isinstance(content.get("selected_detections"), list):
            raw_rows = content["selected_detections"]
            notes.append("selected_detections envelope normalized")
        elif content.get("detection_id"):
            raw_rows = [content]
            notes.append("single-result envelope normalized")
    row = next((item for item in raw_rows if isinstance(item, dict)
                and str(item.get("detection_id", detection_id)) == detection_id), None)
    schema_valid = row is not None and "class_reviews" in (content if isinstance(content, dict) else {})
    if row is None:
        row = {}
        notes.append("classification result missing; recorded as unknown")

    proposed_raw = next((row[key] for key in (
        "proposed_class", "proposed_base_class", "component_type", "component_class",
        "predicted_class", "prediction", "class", "type",
    ) if row.get(key) is not None), None)
    proposed = normalize_qwen_class(proposed_raw) if proposed_raw is not None else UNKNOWN_CLASS
    if proposed_raw is None:
        notes.append("proposed_class missing; recorded as unknown")
        schema_valid = False
    elif "proposed_class" not in row and "proposed_base_class" not in row:
        notes.append("component class alias normalized")
        schema_valid = False
    confidence = row.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        confidence = float(confidence)
        if 1 < confidence <= 100:
            confidence /= 100
            notes.append("confidence percentage converted to 0..1")
            schema_valid = False
        if not 0 <= confidence <= 1:
            confidence = None
            notes.append("confidence outside supported range omitted")
            schema_valid = False
    else:
        confidence = None
        notes.append("confidence missing or invalid")
        schema_valid = False
    original_decision = str(row.get("decision") or "").strip().casefold()
    model_uncertain = original_decision in {"uncertain", "unsure", "not_sure", "inconclusive"}
    if original_decision == "suggest":
        decision = "suggest"
    elif original_decision in {"confirmed", "confirm", "identified", "classified", "likely"}:
        decision = "suggest"
        notes.append("model confirmation normalized to an unconfirmed suggestion")
        schema_valid = False
    elif original_decision in {"uncertain", "unsure", "not_sure", "inconclusive"}:
        # Keep an explicit class prediction visible while preserving uncertainty
        # as a separate flag. The review helper otherwise replaces that class with
        # UNKNOWN whenever decision=uncertain.
        decision = "suggest" if proposed != UNKNOWN_CLASS else "uncertain"
        notes.append("model uncertainty retained separately from its proposed class")
        schema_valid = False
    elif not original_decision and proposed != UNKNOWN_CLASS:
        decision = "suggest"
        notes.append("missing decision normalized to an unconfirmed suggestion")
        schema_valid = False
    else:
        decision = "uncertain"
        model_uncertain = True
        notes.append("decision missing or invalid; normalized to uncertain")
        schema_valid = False
    evidence = row.get("evidence", row.get("evidence_sentence", row.get("visual_evidence")))
    if "evidence" not in row and evidence:
        notes.append("evidence field alias normalized")
        schema_valid = False
    if not isinstance(evidence, str) or not evidence.strip():
        evidence = "모델이 판별 근거를 반환하지 않아 분류를 확인할 수 없습니다."
        notes.append("evidence missing")
        schema_valid = False
    allowed_classes = set(QWEN_CLASSIFICATION_CLASSES)
    raw_class_token = str(proposed_raw or "").strip().casefold().replace("-", "_").replace(" ", "_")
    unsupported_raw_class = (proposed == UNKNOWN_CLASS
                             and raw_class_token not in {"", "unknown", "unknown_component"})
    if proposed not in allowed_classes or unsupported_raw_class:
        proposed = UNKNOWN_CLASS
        model_uncertain = True
        decision = "uncertain"
        notes.append("class outside the 18-class selection taxonomy; recorded as unknown")
    proposed = proposed if proposed in allowed_classes else UNKNOWN_CLASS
    subtype_raw = next((row[key] for key in ("proposed_subtype", "subtype", "component_subtype")
                        if row.get(key) is not None), None)
    model_uncertain = bool(model_uncertain or proposed == UNKNOWN_CLASS)
    normalized = {"detection_id": detection_id, "decision": decision,
                  "proposed_class": proposed,
                  "proposed_subtype": normalize_qwen_subtype(proposed, subtype_raw),
                  "confidence": confidence if confidence is not None else 0.0,
                  "evidence": evidence[:500],
                  "uncertain": model_uncertain}
    content = dict(content) if isinstance(content, dict) else {}
    content["class_reviews"] = [normalized]
    response["content"] = content
    response["archive2_normalization"] = {"schema_valid_before_normalization": schema_valid,
                                          "original_decision": original_decision or None,
                                          "model_uncertain": model_uncertain,
                                          "notes": notes}
    return response


def _normalize_experiment_response_batch(
    response: dict[str, Any], detection_ids: list[str],
) -> dict[str, Any]:
    """Normalize model envelopes without turning absent rows into unknown parts."""
    raw_content = json.loads(json.dumps(response.get("content"), ensure_ascii=False))
    id_set = set(detection_ids)
    notes: list[str] = []
    raw_rows: list[Any] = []
    if isinstance(raw_content, list):
        raw_rows = raw_content
        notes.append("root array normalized")
    elif isinstance(raw_content, dict):
        for key in ("class_reviews", "detections", "results", "selected_detections", "boxes"):
            value = raw_content.get(key)
            if isinstance(value, list):
                raw_rows = value
                if key != "class_reviews":
                    notes.append(f"{key} envelope normalized")
                break
        if not raw_rows and raw_content.get("detection_id"):
            raw_rows = [raw_content]
            notes.append("single row normalized")

    by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        detection_id = str(raw.get("detection_id") or raw.get("component_id") or raw.get("id") or "")
        if detection_id not in id_set:
            continue
        # Some VLMs wrap each row's useful fields in class_reviews.
        nested = raw.get("class_reviews")
        if isinstance(nested, list) and nested:
            inner = next((item for item in nested if isinstance(item, dict)), {})
            raw = {**raw, **inner, "detection_id": detection_id}
            notes.append("nested class_reviews row normalized")
        class_raw = next((raw.get(key) for key in (
            "proposed_class", "proposed_base_class", "component_type", "component_class",
            "predicted_class", "prediction", "class", "type",
        ) if raw.get(key) is not None), None)
        proposed = normalize_qwen_class(class_raw) if class_raw is not None else UNKNOWN_CLASS
        class_token = str(class_raw or "").strip().casefold().replace("-", "_").replace(" ", "_")
        if proposed not in QWEN_CLASSIFICATION_CLASSES or (
                proposed == UNKNOWN_CLASS and class_token not in {"", "unknown", "unknown_component"}):
            proposed = UNKNOWN_CLASS
        raw_status = raw.get("object_status", raw.get("presence_status"))
        if raw_status is None and raw.get("is_physical_component") is True:
            raw_status = "COMPONENT"
        elif raw_status is None and raw.get("is_physical_component") is False:
            raw_status = "NOT_COMPONENT"
        if raw_status is None and str(raw.get("decision") or "").casefold() == "not_component":
            raw_status = "NOT_COMPONENT"
        status_aliases = {"COMPONENT": "COMPONENT", "PRESENT": "COMPONENT", "PHYSICAL_COMPONENT": "COMPONENT",
                          "NOT_COMPONENT": "NOT_COMPONENT", "FALSE_POSITIVE": "NOT_COMPONENT", "FP": "NOT_COMPONENT",
                          "UNCERTAIN": "UNCERTAIN", "UNKNOWN": "UNCERTAIN", "UNSURE": "UNCERTAIN"}
        object_status = status_aliases.get(str(raw_status or "").strip().upper(), "UNCERTAIN")
        if object_status == "NOT_COMPONENT":
            proposed = UNKNOWN_CLASS
        by_id.setdefault(detection_id, {
            "detection_id": detection_id, "proposed_class": proposed,
            "proposed_subtype": normalize_qwen_subtype(proposed, raw.get("proposed_subtype", raw.get("subtype"))),
            "object_status": object_status,
            "decision": "suggest" if proposed != UNKNOWN_CLASS else "uncertain",
            "uncertain": object_status == "UNCERTAIN" or proposed == UNKNOWN_CLASS,
            "response_status": "RECEIVED",
        })

    normalized_rows = []
    for detection_id in detection_ids:
        row = by_id.get(detection_id)
        if row is None:
            row = {"detection_id": detection_id, "proposed_class": UNKNOWN_CLASS,
                   "proposed_subtype": None, "object_status": "NO_RESPONSE",
                   "decision": "uncertain", "uncertain": True, "response_status": "NO_RESPONSE"}
        normalized_rows.append(row)
    missing_ids = [row["detection_id"] for row in normalized_rows if row["response_status"] == "NO_RESPONSE"]
    if missing_ids:
        notes.append(f"missing detection responses: {len(missing_ids)}")
    normalized = dict(response)
    normalized_content = dict(raw_content) if isinstance(raw_content, dict) else {}
    normalized_content["class_reviews"] = normalized_rows
    normalized["content"] = normalized_content
    normalized["archive2_raw_content"] = raw_content
    normalized["archive2_normalization"] = {
        "schema_valid_before_normalization": bool(raw_rows) and not notes,
        "model_uncertain_by_id": {row["detection_id"]: row["uncertain"] for row in normalized_rows},
        "missing_ids": missing_ids,
        "notes": notes,
    }
    return normalized


def _result_path() -> Path:
    root = Path(os.environ.get("CIRCUITCARE_ARCHIVE2_RESULTS_DIR",
                               Path.home() / ".circuitcare-ai" / "archive2")).expanduser()
    return root / "model_comparison_runs.json"


def _load_runs() -> list[dict[str, Any]]:
    try:
        payload = json.loads(_result_path().read_text(encoding="utf-8"))
        return payload.get("runs", []) if isinstance(payload, dict) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save_runs(rows: list[dict[str, Any]]) -> None:
    path = _result_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps({"schema_version": 1, "runs": rows[-5000:]},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _probe_models(client: OllamaClient, backend: str) -> dict[str, dict[str, Any]]:
    try:
        installed = client.list_models()
    except Exception as exc:
        return {key: {"status": "UNAVAILABLE", "reason": f"서버 연결 실패: {str(exc)[:200]}"}
                for key in MODEL_CATALOG}
    installed = {str(row.get("name") or row.get("model") or ""): row for row in installed
                 if isinstance(row, dict)}
    output = {}
    for key, spec in MODEL_CATALOG.items():
        model_id = spec["mlx_id"] if backend == "MLX" else spec["ollama_id"]
        if backend == "MLX" and key != "qwen35_9b":
            output[key] = {"status": "UNAVAILABLE", "reason": "현재 연결된 MLX 서버가 이 모델을 제공하지 않습니다. 해당 MLX 모델 서버를 별도로 설정한 뒤 다시 확인하세요.",
                           "model_id": model_id}
            continue
        runtime_id = "qwen3.5:9b" if backend == "MLX" else model_id
        installed_row = installed.get(runtime_id)
        if not installed_row:
            output[key] = {"status": "UNAVAILABLE", "reason": f"로컬 설치 항목 없음: {model_id}. Ollama 지원 시 `ollama pull {model_id}` 후 다시 확인하세요. 자동 다운로드는 하지 않습니다.",
                           "model_id": model_id}
            continue
        try:
            info = client.show_model(runtime_id)
            caps = info.get("capabilities", [])
            if not isinstance(caps, list) or "vision" not in caps:
                output[key] = {"status": "UNAVAILABLE", "reason": "로컬 모델이 이미지 입력을 선언하지 않음",
                               "model_id": model_id}
            else:
                output[key] = {"status": "READY", "reason": "이미지 입력 확인", "model_id": model_id,
                               "runtime_id": runtime_id, "digest": installed_row.get("digest"),
                               "size_bytes": installed_row.get("size"),
                               "think_supported": "thinking" in caps}
        except Exception as exc:
            output[key] = {"status": "UNAVAILABLE", "reason": f"모델 호환성 확인 실패: {str(exc)[:200]}",
                           "model_id": model_id}
    return output


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") in {"COMPLETE", "FAILED", "UNAVAILABLE"}:
            key = (row.get("backend"), row.get("model_key"), row.get("crop_px"),
                   row.get("padding_ratio"), row.get("output_budget"))
            groups.setdefault(key, []).append(row)
    result = []
    for key, items in groups.items():
        completed = [row for row in items if row.get("status") == "COMPLETE"]
        prediction_samples = []
        for item in completed:
            if isinstance(item.get("predictions"), list):
                prediction_samples.extend(
                    {**item, **prediction, "prediction": prediction.get("proposed_class"),
                     "prediction_subtype": prediction.get("proposed_subtype")}
                    for prediction in item["predictions"] if isinstance(prediction, dict)
                )
            else:
                prediction_samples.append(item)
        labeled = [row for row in prediction_samples if row.get("ground_truth") in COMPONENT_CLASSES]
        correct = sum(row.get("prediction") == row.get("ground_truth") for row in labeled)
        classes = sorted({str(row["ground_truth"]) for row in labeled})
        f1 = []
        for label in classes:
            tp = sum(row.get("prediction") == label and row.get("ground_truth") == label for row in labeled)
            fp = sum(row.get("prediction") == label and row.get("ground_truth") != label for row in labeled)
            fn = sum(row.get("prediction") != label and row.get("ground_truth") == label for row in labeled)
            p, r = tp / (tp + fp) if tp + fp else 0, tp / (tp + fn) if tp + fn else 0
            f1.append(2 * p * r / (p + r) if p + r else 0)
        timings = sorted(float(row["wall_ms"]) for row in items if isinstance(row.get("wall_ms"), (int, float)))
        warm = [float(row["wall_ms"]) for row in items
                if row.get("status") == "COMPLETE" and row.get("repeat_index", 0) > 0]
        confirmed_cold = [float(row["wall_ms"]) for row in items
                          if row.get("status") == "COMPLETE" and row.get("repeat_index") == 0
                          and isinstance(row.get("load_ms"), (int, float)) and row.get("load_ms") > 0]
        subtype = [row for row in labeled if row.get("ground_truth_subtype")]
        summary = {
            "backend": key[0], "model": MODEL_CATALOG[key[1]]["label"],
            "model_id": items[0].get("model_id"), "crop_px": key[2],
            "padding_ratio": key[3], "output_budget": key[4], "sample_n": len(labeled),
            "accuracy": correct / len(labeled) if labeled else None,
            "macro_f1": statistics.mean(f1) if f1 else None,
            "class_n": len(classes),
            "subtype_n": len(subtype),
            "subtype_accuracy": (sum(row.get("prediction_subtype") == row.get("ground_truth_subtype")
                                     for row in subtype) / len(subtype)) if subtype else None,
            "ic_accuracy": _subset_accuracy(labeled, {"ic"}),
            "connector_accuracy": _subset_accuracy(labeled, {"connector"}),
            "small_smd_accuracy": _subset_accuracy([row for row in labeled if row.get("small_smd")],
                                                    set(COMPONENT_CLASSES)),
            "unknown_rate": (sum(row.get("prediction", row.get("proposed_class")) == UNKNOWN_CLASS
                                  for row in prediction_samples) / len(prediction_samples)
                             if prediction_samples else None),
            "misclassification_rate": (sum(row.get("prediction") != row.get("ground_truth")
                                           for row in labeled) / len(labeled)) if labeled else None,
            "json_valid_rate": sum(bool(row.get("json_valid")) for row in items) / len(items),
            "schema_valid_rate": sum(bool(row.get("schema_valid")) for row in items) / len(items),
            "first_run_ms": next((row.get("wall_ms") for row in items if row.get("repeat_index") == 0), None),
            "confirmed_cold_run_ms": confirmed_cold[0] if confirmed_cold else None,
            "actual_load_ms": next((row.get("load_ms") for row in items if row.get("load_ms") not in (None, 0)), None),
            "warm_mean_ms": statistics.mean(warm) if warm else None,
            "mean_ms": statistics.mean(timings) if timings else None,
            "p50_ms": statistics.median(timings) if timings else None,
            "p95_ms": timings[min(len(timings) - 1, max(0, int(.95 * len(timings) + .5) - 1))] if timings else None,
            "mean_prefill_ms": _mean(items, "prompt_eval_ms"),
            "mean_decode_ms": _mean(items, "eval_ms"),
            "tokens_per_second": _mean(items, "tokens_per_second"),
            "model_memory_mb": _mean(items, "model_memory_mb"),
        }
        result.append(summary)
    return result


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    return statistics.mean(values) if values else None


def _subset_accuracy(rows: list[dict[str, Any]], classes: set[str]) -> float | None:
    selected = [row for row in rows if row.get("ground_truth") in classes]
    return sum(row.get("prediction") == row.get("ground_truth") for row in selected) / len(selected) if selected else None


def render_model_comparison_workspace(st: Any, project: Any, photos: list[dict[str, Any]]) -> None:
    """Compare selected-component crops in an isolated Archive 2 workspace."""
    from circuitcare.dashboard.views.unified_workspace import _cached_photo_pixels
    from circuitcare.vision.component_recognition_panel import (
        ensure_component_detection, review_aux_keys, review_keys, review_scope,
    )
    from circuitcare.vision.ai_review_service import (
        AIReviewStageError, FULL_BOARD_CLASSIFICATION_SCHEMA, _FULL_BOARD_CLASSIFICATION_SYSTEM,
        _duplicate_pair_candidates, review_components_with_qwen,
    )
    from circuitcare.vision.box_reconciliation import same_component_box_pair

    st.markdown("### Archive 2 · 소자별 이미지 AI 비교")
    st.caption("전체 PCB 사진과 모든 활성 YOLO 박스를 네 모델에 동일하게 입력해 종류를 비교합니다. 결과는 실험 기록에만 저장됩니다.")
    photos = [row for row in photos if isinstance(row, dict) and row.get("image_id")
              and isinstance(row.get("data"), bytes)]
    if not photos:
        st.info("PCB 사진을 먼저 올려 주세요.")
        return

    photo_ids = [str(row["image_id"]) for row in photos]
    photo_id = st.selectbox("PCB 사진", photo_ids, key=f"cc_archive2_photo_v2_{project.project_id}",
                            format_func=lambda value: f"사진 {photo_ids.index(value) + 1} · {value}")
    photo = next(row for row in photos if str(row["image_id"]) == photo_id)
    try:
        _image, _source_size, image_size, image_bytes = _cached_photo_pixels(photo["data"])
    except Exception as exc:
        st.error(f"사진을 준비하지 못했습니다 · {exc}")
        return
    scope = review_scope(image_bytes, project.project_id, photo_id)
    result_key, rows_key, error_key = review_keys(scope)
    if result_key not in st.session_state and error_key not in st.session_state:
        with st.status("선택한 사진에서 YOLO 소자 탐지 중", expanded=True) as yolo_status:
            detected, detection_error = ensure_component_detection(
                st, image_bytes=photo["data"], project_id=project.project_id, photo_id=photo_id,
                review_image_bytes=image_bytes, review_image_size=image_size, progress=yolo_status,
            )
            if detection_error or detected is None:
                yolo_status.update(label="YOLO 탐지 실패", state="error", expanded=True)
            else:
                yolo_status.update(label=f"YOLO 탐지 완료 · {len(detected.get('detections', []))}개 후보",
                                   state="complete", expanded=False)
    detections = st.session_state.get(rows_key, [])
    if error_key in st.session_state and result_key not in st.session_state:
        st.error(f"자동 소자 탐지에 실패했습니다 · {st.session_state[error_key]}")
        if st.button("YOLO 다시 실행", key=f"cc_archive2_yolo_retry_{project.project_id}_{photo_id}"):
            st.session_state.pop(error_key, None)
            st.rerun()
        return
    if not isinstance(detections, list) or not detections:
        st.info("YOLO 탐지는 완료됐지만 선택할 수 있는 소자 박스가 없습니다.")
        return
    active = [row for row in detections if isinstance(row, dict) and row.get("상태") != "제외"
              and row.get("component_id")]
    if not active:
        st.info("비교할 수 있는 활성 박스가 없습니다.")
        return
    comparison_detections = []
    source_by_id = {}
    for source in active:
        component_id = str(source["component_id"])
        try:
            bbox = [float(source[name]) for name in ("x1", "y1", "x2", "y2")]
        except (KeyError, TypeError, ValueError):
            continue
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        detector_class = str(source.get("detector_class") or source.get("class_name")
                             or source.get("소자 종류") or UNKNOWN_CLASS)
        detection = {"detection_id": component_id, "component_id": component_id,
                     "class_name": detector_class, "bbox_xyxy": bbox,
                     "confidence": source.get("신뢰도", source.get("confidence")),
                     "ocr_candidates": source.get("ocr_candidates", []),
                     "ocr_text": source.get("ocr_candidates", []),
                     "ocr_class_hints": source.get("ocr_class_hints", []),
                     "reference_designators": source.get("reference_designators", []),
                     "body_marking": source.get("body_marking"),
                     "marking_candidates": source.get("marking_candidates", []),
                     "possible_part_number": source.get("possible_part_number"),
                     "user_component_knowledge": []}
        comparison_detections.append(detection)
        source_by_id[component_id] = source
    if not comparison_detections:
        st.error("전체 사진 비교에 사용할 좌표가 유효한 YOLO 박스가 없습니다.")
        return
    component_id = "__FULL_BOARD__"
    detection_ids = [row["detection_id"] for row in comparison_detections]
    model_detection_ids = [str(index) for index in range(1, len(detection_ids) + 1)]
    model_detection_id_by_real_id = dict(zip(detection_ids, model_detection_ids))
    real_detection_by_photo_number = {str(index): row for index, row in enumerate(comparison_detections, start=1)}
    model_detections = [{**row, "detection_id": model_id, "component_id": model_id}
                        for row, model_id in zip(comparison_detections, model_detection_ids)]
    bbox_by_id = {row["detection_id"]: row["bbox_xyxy"] for row in comparison_detections}
    ground_truth_by_id = {
        component_id: str(source_by_id[component_id].get("소자 종류") or "").casefold()
        for component_id in detection_ids
        if source_by_id[component_id].get("상태") == "확인"
        and str(source_by_id[component_id].get("소자 종류") or "").casefold() in COMPONENT_CLASSES
    }
    if len(comparison_detections) < len(active):
        st.warning(f"좌표가 잘못된 박스 {len(active) - len(comparison_detections)}개는 이번 비교에서 제외했습니다.")

    ocr_key, _ = review_aux_keys(scope)
    ocr_state = st.session_state.get(ocr_key, {})
    ocr_rows = ocr_state.get("records", []) if isinstance(ocr_state, dict) else []
    backend_code = "Ollama"
    client = OllamaClient(timeout_seconds=4, num_predict=256, keep_alive="10m", auto_start=False)
    availability_key = f"cc_archive2_model_status_v2_{backend_code}"
    availability = st.session_state.get(availability_key, {})
    if not isinstance(availability, dict) or not availability:
        availability = _probe_models(client, backend_code)
        st.session_state[availability_key] = availability
    if st.button("모델 상태 새로고침", key=f"cc_archive2_probe_v2_{project.project_id}"):
        availability = _probe_models(client, backend_code)
        st.session_state[availability_key] = availability

    selected_models = st.multiselect(
        "비교할 모델", list(MODEL_CATALOG), default=list(MODEL_CATALOG),
        key=f"cc_archive2_models_v2_{project.project_id}",
        format_func=lambda value: f"{MODEL_CATALOG[value]['label']} · "
                                  f"{'서버 대기 · 실행 시 시작' if '서버 연결 실패' in str(availability.get(value, {}).get('reason', '')) else availability.get(value, {}).get('status', '확인 중')}",
    )
    with st.expander("고급 설정", expanded=False):
        budget = st.selectbox("배치당 응답 토큰 상한", OUTPUT_BUDGETS, index=OUTPUT_BUDGETS.index(256),
                              key=f"cc_archive2_budget_v2_{project.project_id}")

    image_sha = hashlib.sha256(photo["data"]).hexdigest()
    prompt_fingerprint = hashlib.sha256(_FULL_BOARD_CLASSIFICATION_SYSTEM.encode("utf-8")).hexdigest()
    full_board_schema = deepcopy(FULL_BOARD_CLASSIFICATION_SCHEMA)
    full_board_schema["properties"]["class_reviews"]["minItems"] = 0
    full_board_schema["properties"]["class_reviews"]["maxItems"] = len(comparison_detections)
    schema_fingerprint = hashlib.sha256(json.dumps(
        full_board_schema, ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()
    board_preview = _image.convert("RGB").copy()
    board_draw = ImageDraw.Draw(board_preview)
    box_width = max(2, round(min(board_preview.size) / 500))
    for detection, model_id in zip(comparison_detections, model_detection_ids):
        x1, y1, x2, y2 = (round(value) for value in detection["bbox_xyxy"])
        board_draw.rectangle((x1, y1, x2, y2), outline="#77d8ff", width=box_width)
        label_y = max(0, y1 - max(18, box_width * 8))
        board_draw.rectangle((x1, label_y, x1 + max(36, len(model_id) * 8),
                              label_y + max(18, box_width * 8)), fill="#77d8ff")
        board_draw.text((x1 + 4, label_y + 2), model_id, fill="#17202a")
    top, bottom = st.columns([2, 1])
    top.image(board_preview, caption=f"전체 PCB · 탐지 박스 {len(comparison_detections)}개", use_container_width=True)
    bottom.markdown(f"**전체 사진 비교 대상** · {len(comparison_detections)}개 YOLO 박스")
    bottom.caption(f"모든 모델에 동일한 전체 사진과 박스 ID를 보냅니다 · "
                   f"사용자 확정 정답 {len(ground_truth_by_id)}개는 모델 입력에서 제외 · "
                   f"OCR 문맥 {len(ocr_rows)}개 · {image_size[0]}×{image_size[1]}px")
    st.caption("사진 위 번호는 아래 ‘박스 번호’와 같습니다. YOLO 원본 ID는 별도 열에 함께 표시합니다.")
    with st.expander("사진 번호로 박스 확대 확인", expanded=False):
        selected_photo_number = st.selectbox(
            "확대할 번호", model_detection_ids,
            key=f"cc_archive2_box_crop_{project.project_id}_{photo_id}",
        )
        selected_detection = real_detection_by_photo_number[selected_photo_number]
        raw_x1, raw_y1, raw_x2, raw_y2 = [int(round(value)) for value in selected_detection["bbox_xyxy"]]
        left, top_px = max(0, min(_image.width, raw_x1)), max(0, min(_image.height, raw_y1))
        right, bottom_px = max(0, min(_image.width, raw_x2)), max(0, min(_image.height, raw_y2))
        if right <= left or bottom_px <= top_px:
            st.warning(f"번호 {selected_photo_number} 박스가 이미지 영역 밖에 있어 확대할 수 없습니다. YOLO 원본 ID: {selected_detection['detection_id']}")
        else:
            crop = _image.convert("RGB").crop((left, top_px, right, bottom_px))
            crop.thumbnail((720, 720), Image.Resampling.LANCZOS)
            st.image(crop, caption=f"번호 {selected_photo_number} · {selected_detection['detection_id']} · "
                                   f"YOLO {selected_detection['class_name']}", width="stretch")
        st.caption("AI의 ‘오탐 후보’는 확정 판정이 아닙니다. 확대 박스와 전체 기판을 함께 보고 사용자가 확인하세요.")
    duplicate_pairs = _duplicate_pair_candidates(comparison_detections, limit=24)
    duplicate_pair_review_state_key = f"cc_archive2_duplicate_model_results_{project.project_id}_{photo_id}"
    duplicate_pair_fingerprint = hashlib.sha256(json.dumps({
        "image": image_sha,
        "detections": [{"id": row["detection_id"], "box": row["bbox_xyxy"]}
                       for row in comparison_detections],
        "pairs": [[item["detection_id"] for item in pair["detections"]]
                  for pair in duplicate_pairs],
        "models": selected_models,
        "mode": "duplicate-and-physical-presence-v1",
    }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    duplicate_pair_state = st.session_state.get(duplicate_pair_review_state_key)
    if (not isinstance(duplicate_pair_state, dict)
            or duplicate_pair_state.get("fingerprint") != duplicate_pair_fingerprint):
        duplicate_pair_state = {"fingerprint": duplicate_pair_fingerprint, "pairs": {}}
        st.session_state[duplicate_pair_review_state_key] = duplicate_pair_state
    pair_model_results = duplicate_pair_state["pairs"]
    pair_label_to_candidate = {}
    for pair in duplicate_pairs:
        left_id, right_id = (str(item["detection_id"]) for item in pair["detections"])
        left_number = model_detection_id_by_real_id.get(left_id, "?")
        right_number = model_detection_id_by_real_id.get(right_id, "?")
        pair_label_to_candidate[
            f"{left_number}번 (YOLO {left_id}) ↔ {right_number}번 (YOLO {right_id})"
        ] = pair
    with st.expander(f"겹침·중복 의심 박스 · {len(duplicate_pairs)}쌍", expanded=bool(duplicate_pairs)):
        if duplicate_pairs:
            st.caption("같은 모델 선택을 사용해 모든 선택 모델이 같은 박스 쌍을 순차 비교합니다. 모델의 실물/오탐 판단은 제안이며 정답 확정이나 박스 자동 삭제가 아닙니다.")
            st.dataframe([{
                "박스 번호 A": model_detection_id_by_real_id.get(
                    str(pair["detections"][0]["detection_id"]), "—"),
                "YOLO 원본 ID A": pair["detections"][0]["detection_id"],
                "박스 번호 B": model_detection_id_by_real_id.get(
                    str(pair["detections"][1]["detection_id"]), "—"),
                "YOLO 원본 ID B": pair["detections"][1]["detection_id"],
                "기하 분류": "중복 가능성 높음" if same_component_box_pair(
                    pair["detections"][0]["bbox_xyxy"], pair["detections"][1]["bbox_xyxy"]
                ) else "겹침만 확인 · 별도 소자일 수 있음",
                "모델 검토 완료": sum(
                    pair_model_results.get("|".join(sorted(item["detection_id"] for item in pair["detections"])), {})
                    .get(model_key, {}).get("status") in {"COMPLETE", "PARTIAL"}
                    for model_key in selected_models
                ),
                "IoU": round(pair["metrics"]["iou"], 3),
                "작은 박스 포함률": round(pair["metrics"]["smaller_coverage"], 3),
            } for pair in duplicate_pairs], use_container_width=True, hide_index=True)
            selected_pair_labels = st.multiselect(
                "AI로 확인할 겹침 쌍 (한 번에 최대 3쌍)", list(pair_label_to_candidate), max_selections=3,
                key=f"cc_archive2_duplicate_pair_select_{project.project_id}_{photo_id}",
            )
            review_all_pairs = st.button(
                f"선택 모델 전체로 상위 후보 비교 · {min(len(duplicate_pairs), 12)}쌍",
                key=f"cc_archive2_duplicate_run_all_{project.project_id}_{photo_id}",
                disabled=not duplicate_pairs or not selected_models,
            )
            review_selected_pairs = st.button(
                "선택 쌍을 선택 모델 전체로 비교",
                key=f"cc_archive2_duplicate_run_{project.project_id}_{photo_id}",
                disabled=not selected_pair_labels or not selected_models,
            )
            if review_all_pairs or review_selected_pairs:
                duplicate_status = st.status("선택한 모델로 박스 쌍을 비교 중", expanded=True)
                try:
                    selected_candidates = ([pair_label_to_candidate[label] for label in selected_pair_labels]
                                           if review_selected_pairs else duplicate_pairs[:12])
                    batches = [selected_candidates[index:index + 4]
                               for index in range(0, len(selected_candidates), 4)]
                    pair_count = sum(len(batch) for batch in batches)
                    unavailable_count = 0
                    failed_count = 0
                    for model_index, model_key in enumerate(selected_models, start=1):
                        model_id = MODEL_CATALOG[model_key]["ollama_id"]
                        model_state = availability.get(model_key, {})
                        duplicate_status.update(
                            label=f"{MODEL_CATALOG[model_key]['label']} · 모델 {model_index}/{len(selected_models)}",
                            state="running", expanded=True,
                        )
                        if model_state.get("status") != "READY":
                            unavailable_count += pair_count
                            for pair in selected_candidates:
                                ids = tuple(sorted(item["detection_id"] for item in pair["detections"]))
                                pair_key = "|".join(ids)
                                pair_model_results.setdefault(pair_key, {})[model_key] = {
                                    "status": "UNAVAILABLE",
                                    "reason": model_state.get("reason", "설치 또는 실행 불가"),
                                    "component_status": {},
                                }
                            continue
                        duplicate_client = OllamaClient(timeout_seconds=300, num_predict=2048,
                                                        keep_alive="10m", auto_start=True)
                        for batch_index, candidate_batch in enumerate(batches, start=1):
                            duplicate_status.update(
                                label=f"{MODEL_CATALOG[model_key]['label']} · {batch_index}/{len(batches)} 묶음",
                                state="running", expanded=True,
                            )
                            started = time.perf_counter_ns()
                            batch_failure_reason = None
                            try:
                                duplicate_result = review_components_with_qwen(
                                    duplicate_client, model_id, image_bytes, comparison_detections, [],
                                    duplicate_pair_candidates=candidate_batch, duplicate_pairs_only=True,
                                    duplicate_output_budget=2048, include_component_presence=True,
                                )
                                candidate_results = duplicate_result
                            except AIReviewStageError as batch_error:
                                if batch_error.stage != "qwen_validation" or "omitted" not in str(batch_error):
                                    candidate_results = None
                                else:
                                    candidate_results = {"duplicate_reviews": [], "component_presence_reviews": []}
                                    for pair_index, pair in enumerate(candidate_batch, start=1):
                                        duplicate_status.update(
                                            label=f"{MODEL_CATALOG[model_key]['label']} · 누락 쌍 재확인 "
                                                  f"{pair_index}/{len(candidate_batch)}",
                                            state="running", expanded=True,
                                        )
                                        try:
                                            pair_result = review_components_with_qwen(
                                                duplicate_client, model_id, image_bytes, comparison_detections, [],
                                                duplicate_pair_candidates=[pair], duplicate_pairs_only=True,
                                                duplicate_output_budget=768, include_component_presence=True,
                                            )
                                            candidate_results["duplicate_reviews"].extend(
                                                pair_result["duplicate_reviews"])
                                            candidate_results["component_presence_reviews"].extend(
                                                pair_result["component_presence_reviews"])
                                        except Exception:
                                            ids = tuple(sorted(item["detection_id"] for item in pair["detections"]))
                                            pair_model_results.setdefault("|".join(ids), {})[model_key] = {
                                                "status": "NO_RESPONSE", "decision": None,
                                                "component_status": {}, "reason": "응답 누락 또는 형식 오류",
                                            }
                                            failed_count += 1
                            except Exception as model_error:
                                candidate_results = None
                                batch_failure_reason = f"{type(model_error).__name__}: {str(model_error)[:180]}"
                                failed_count += len(candidate_batch)

                            elapsed_ms = round((time.perf_counter_ns() - started) / 1_000_000, 1)
                            duplicate_by_pair = {
                                tuple(sorted(row["detection_ids"])): row
                                for row in candidate_results["duplicate_reviews"]
                            } if candidate_results else {}
                            presence_by_id = {
                                row["detection_id"]: row
                                for row in candidate_results["component_presence_reviews"]
                            } if candidate_results else {}
                            for pair in candidate_batch:
                                ids = tuple(sorted(item["detection_id"] for item in pair["detections"]))
                                pair_key = "|".join(ids)
                                existing = pair_model_results.get(pair_key, {}).get(model_key, {})
                                if existing.get("status") == "NO_RESPONSE":
                                    continue
                                duplicate = duplicate_by_pair.get(ids)
                                component_status = {}
                                for detection_id in ids:
                                    presence = presence_by_id.get(detection_id)
                                    component_status[detection_id] = {
                                        "status": presence.get("object_status", "NO_RESPONSE")
                                        if presence else "NO_RESPONSE",
                                        "evidence": presence.get("evidence", "") if presence else "",
                                    }
                                pair_model_results.setdefault(pair_key, {})[model_key] = {
                                    "status": ("COMPLETE" if duplicate and all(
                                        item["status"] != "NO_RESPONSE" for item in component_status.values())
                                        else "PARTIAL" if duplicate or any(
                                            item["status"] != "NO_RESPONSE" for item in component_status.values())
                                        else "NO_RESPONSE"),
                                    "decision": duplicate.get("decision") if duplicate else None,
                                    "evidence": duplicate.get("evidence", "") if duplicate else "",
                                    "component_status": component_status,
                                    "wall_ms": elapsed_ms,
                                    "reason": None if duplicate else (
                                        batch_failure_reason or "모델 응답에서 쌍 판정이 누락됨"),
                                }
                        st.session_state[duplicate_pair_review_state_key] = duplicate_pair_state
                    deferred = max(0, len(duplicate_pairs) - 12) if review_all_pairs else 0
                    duplicate_status.update(
                        label=f"선택 모델 비교 완료 · {len(selected_candidates)}쌍" +
                              (f" · 나머지 {deferred}쌍은 대기" if deferred else ""),
                        state="complete", expanded=bool(deferred),
                    )
                    if unavailable_count or failed_count:
                        st.warning(f"모델 미설치/실행 불가 결과 {unavailable_count}개 · "
                                   f"응답 누락 또는 오류 {failed_count}개. 모델별 표에서 상태를 확인하세요.")
                    if deferred:
                        st.info(f"후보가 12쌍을 넘어 상위 12쌍만 이번에 검토했습니다. 남은 {deferred}쌍은 선택 검토로 확인하세요.")
                    st.rerun()
                except Exception as exc:
                    duplicate_status.update(label="겹침 쌍 비교 실패", state="error", expanded=True)
                    st.error(f"{type(exc).__name__}: {str(exc)[:400]}")
            status_labels = {
                "COMPONENT": "실물 부품 후보",
                "NOT_COMPONENT": "오탐/기판 요소 후보",
                "UNCERTAIN": "실물 여부 미확정",
                "NO_RESPONSE": "응답 누락",
            }
            model_status_labels = {
                "COMPLETE": "완료",
                "PARTIAL": "일부 응답",
                "NO_RESPONSE": "응답 누락",
                "UNAVAILABLE": "모델 사용 불가",
            }
            decision_labels = {
                "same_component": "같은 부품 후보",
                "different_components": "서로 다른 부품 후보",
                "uncertain": "중복 여부 미확정",
            }
            pair_result_rows = []
            for pair in duplicate_pairs:
                ids = tuple(sorted(item["detection_id"] for item in pair["detections"]))
                pair_key = "|".join(ids)
                model_results = pair_model_results.get(pair_key, {})
                for model_key in selected_models:
                    model_result = model_results.get(model_key, {})
                    components = model_result.get("component_status", {})
                    left = components.get(ids[0], {})
                    right = components.get(ids[1], {})
                    pair_result_rows.append({
                        "박스 쌍": f"{model_detection_id_by_real_id.get(ids[0], '?')} ↔ "
                                   f"{model_detection_id_by_real_id.get(ids[1], '?')}",
                        "모델": MODEL_CATALOG[model_key]["label"],
                        "모델 상태": model_status_labels.get(model_result.get("status"), "미실행"),
                        "같은 부품 판단": decision_labels.get(model_result.get("decision"),
                                                             "판정 미응답" if model_result else "미실행"),
                        f"{model_detection_id_by_real_id.get(ids[0], '?')}번 실물 여부":
                            status_labels.get(left.get("status"), "미실행"),
                        f"{model_detection_id_by_real_id.get(ids[1], '?')}번 실물 여부":
                            status_labels.get(right.get("status"), "미실행"),
                        "간단 근거": model_result.get("evidence") or model_result.get("reason") or "",
                        "요청 묶음 응답시간(ms)": model_result.get("wall_ms"),
                    })
            if pair_result_rows:
                st.markdown("**선택 모델별 겹침·실물 여부 비교**")
                st.dataframe(pair_result_rows, use_container_width=True, hide_index=True)
                viewed_pair_label = st.selectbox(
                    "모델별 결과 카드로 볼 박스 쌍", list(pair_label_to_candidate),
                    key=f"cc_archive2_duplicate_pair_cards_{project.project_id}_{photo_id}",
                )
                viewed_pair = pair_label_to_candidate[viewed_pair_label]
                viewed_ids = tuple(sorted(item["detection_id"] for item in viewed_pair["detections"]))
                viewed_key = "|".join(viewed_ids)
                viewed_model_results = pair_model_results.get(viewed_key, {})
                for detection_id in viewed_ids:
                    counts = {status: sum(
                        viewed_model_results.get(model_key, {}).get("component_status", {})
                        .get(detection_id, {}).get("status") == status
                        for model_key in selected_models
                    ) for status in ("COMPONENT", "NOT_COMPONENT", "UNCERTAIN", "NO_RESPONSE")}
                    st.caption(
                        f"박스 {model_detection_id_by_real_id.get(detection_id, '?')} 실물 여부 모델 응답 집계 "
                        f"· 실물 후보 {counts['COMPONENT']} · 오탐/기판 요소 후보 {counts['NOT_COMPONENT']} "
                        f"· 불확실 {counts['UNCERTAIN']} · 미응답 {counts['NO_RESPONSE']} "
                        "· 모델 합의 수이며 정답 판정은 아님"
                    )
                cards = st.columns(2)
                for model_index, model_key in enumerate(selected_models):
                    model_result = pair_model_results.get(viewed_key, {}).get(model_key, {})
                    with cards[model_index % 2]:
                        with st.container(border=True):
                            st.markdown(f"**{MODEL_CATALOG[model_key]['label']}**")
                            st.caption(model_status_labels.get(model_result.get("status"), "비교 전"))
                            st.markdown("동일 소자 여부 · " +
                                        decision_labels.get(model_result.get("decision"), "판정 응답 없음"))
                            for detection_id in viewed_ids:
                                source = source_by_id.get(detection_id, {})
                                object_status = model_result.get("component_status", {}).get(detection_id, {})
                                number = model_detection_id_by_real_id.get(detection_id, "?")
                                st.write(
                                    f"박스 {number} · YOLO {source.get('detector_class') or source.get('class_name') or 'unknown'}"
                                    f" → {status_labels.get(object_status.get('status'), '미응답')}"
                                )
                            if model_result.get("evidence"):
                                st.caption(model_result["evidence"])
                            if model_result.get("reason"):
                                st.caption(str(model_result["reason"]))
                            if model_result.get("wall_ms") is not None:
                                st.caption(f"요청 묶음 응답 {model_result['wall_ms']} ms")
            st.caption("실물/오탐 결과도 모델의 시각적 제안입니다. 사용자 확인 데이터가 정답 기준이며, 모델 의견만으로 박스를 제외하지 않습니다.")
        else:
            st.caption("현재 겹침 기준에 맞는 후보가 없습니다.")

    # Archive 2 quality pass: use the shared validator for oversized groups and
    # missed boxes. Results stay in this experiment's session state and never
    # modify the production component review rows.
    quality_key = f"cc_archive2_quality_review_{project.project_id}_{photo_id}"
    quality_fingerprint = hashlib.sha256(json.dumps({
        "image": image_sha,
        "detections": [{"id": row["detection_id"], "box": row["bbox_xyxy"], "class": row["class_name"]}
                       for row in comparison_detections],
        "model": MODEL_CATALOG["qwen35_9b"]["ollama_id"],
        "mode": "group-and-missed-candidates-v1",
    }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    quality_state = st.session_state.get(quality_key)
    if not isinstance(quality_state, dict) or quality_state.get("fingerprint") != quality_fingerprint:
        quality_state = None
    with st.expander("큰 박스 묶음 · 미탐지 소자 검사", expanded=False):
        st.caption("Qwen이 전체 PCB와 고해상도 타일을 보고 큰 박스 안의 여러 소자 및 YOLO가 놓친 후보를 찾습니다. 결과는 실험 제안으로만 저장되며 검수 박스를 바꾸지 않습니다.")
        if st.button("묶음·미탐지 AI 검사 실행", key=f"cc_archive2_quality_run_{project.project_id}_{photo_id}"):
            quality_status = st.status("전체 PCB 묶음·미탐지 검사 중", expanded=True)
            try:
                quality_client = OllamaClient(timeout_seconds=600, num_predict=1536,
                                              keep_alive="10m", auto_start=True)
                quality_result = review_components_with_qwen(
                    quality_client, MODEL_CATALOG["qwen35_9b"]["ollama_id"], image_bytes,
                    comparison_detections, ocr_rows,
                    duplicate_pair_candidates=[], include_board_context=True,
                )
                st.session_state[quality_key] = {
                    "fingerprint": quality_fingerprint,
                    "model": MODEL_CATALOG["qwen35_9b"]["ollama_id"],
                    "result": quality_result,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                quality_status.update(label="묶음·미탐지 검사 완료", state="complete", expanded=False)
                st.rerun()
            except Exception as exc:
                quality_status.update(label="묶음·미탐지 검사 실패", state="error", expanded=True)
                st.error(f"{type(exc).__name__}: {str(exc)[:500]}")
        if quality_state is None:
            st.info("이 사진에서 묶음·미탐지 검사를 아직 실행하지 않았습니다.")
        else:
            quality_result = quality_state.get("result", {})
            group_reviews = quality_result.get("group_reviews", [])
            missed_candidates = quality_result.get("missed_component_candidates", [])
            group_diagnostics = quality_result.get("review_diagnostics", {}).get("candidate_search", {})
            deferred_groups = sum(item.get("reason") == "per_batch_group_crop_limit"
                                  for item in group_diagnostics.get("items", []))
            st.caption(f"그룹 후보 {group_diagnostics.get('candidate_count', 0)}개 · "
                       f"Qwen 검토 {len(group_reviews)}개 · 미탐지 후보 {len(missed_candidates)}개 · "
                       f"검토 보류 그룹 {deferred_groups}개")
            quality_preview = _image.convert("RGB").copy()
            quality_draw = ImageDraw.Draw(quality_preview)
            for group_index, review in enumerate(group_reviews, start=1):
                parent = real_detection_by_photo_number.get(
                    model_detection_id_by_real_id.get(str(review.get("parent_id")), ""))
                if parent:
                    x1, y1, x2, y2 = [int(round(value)) for value in parent["bbox_xyxy"]]
                    quality_draw.rectangle((x1, y1, x2, y2), outline="#ffb02e", width=5)
                    quality_draw.text((x1 + 3, max(0, y1 - 20)), f"G{group_index}", fill="#ffb02e",
                                      stroke_width=2, stroke_fill="#17202a")
                for child_index, proposal in enumerate(review.get("new_components", []), start=1):
                    box = proposal.get("bbox_xyxy")
                    if isinstance(box, (list, tuple)) and len(box) == 4:
                        bx1, by1, bx2, by2 = [int(round(value)) for value in box]
                        quality_draw.rectangle((bx1, by1, bx2, by2), outline="#ff5d8f", width=4)
                        quality_draw.text((bx1 + 2, max(0, by1 - 16)), f"G{group_index}.{child_index}",
                                          fill="#ff5d8f", stroke_width=2, stroke_fill="#17202a")
            for missed_index, proposal in enumerate(missed_candidates, start=1):
                box = proposal.get("bbox_xyxy")
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    bx1, by1, bx2, by2 = [int(round(value)) for value in box]
                    quality_draw.rectangle((bx1, by1, bx2, by2), outline="#ff5d8f", width=4)
                    quality_draw.text((bx1 + 2, max(0, by1 - 16)), f"M{missed_index}", fill="#ff5d8f",
                                      stroke_width=2, stroke_fill="#17202a")
            st.image(quality_preview, caption="노랑 G: 묶음 박스 · 분홍 G.*: 누락된 묶음 자식 후보 · 분홍 M: 미탐지 후보",
                     width="stretch")
            if group_reviews:
                st.markdown("**큰 박스 검토 결과**")
                group_rows = []
                for index, review in enumerate(group_reviews, start=1):
                    parent_id = str(review.get("parent_id", ""))
                    group_rows.append({
                        "표시": f"G{index} · 박스 {model_detection_id_by_real_id.get(parent_id, '—')}",
                        "YOLO 원본 ID": parent_id,
                        "AI 판단": {"multiple_components": "여러 소자일 수 있음",
                                    "single_component": "단일 부품 후보", "uncertain": "판정 보류"}.get(
                                        review.get("decision"), "판정 보류"),
                        "기존 내부 박스": ", ".join(review.get("child_ids", [])) or "없음",
                        "신규 위치 제안": len(review.get("new_components", [])),
                        "근거": review.get("evidence", ""),
                    })
                st.dataframe(group_rows, use_container_width=True, hide_index=True)
            if missed_candidates:
                st.markdown("**YOLO 미탐지 후보**")
                missed_rows = [{"표시": f"M{index}", "종류 제안": row.get("proposed_class", UNKNOWN_CLASS),
                                "상세 유형": row.get("proposed_subtype") or "—",
                                "AI 점수 · 확률 아님": row.get("confidence"),
                                "겹친 기존 박스": ", ".join(row.get("possible_duplicate_ids", [])) or "없음",
                                "근거": row.get("evidence", "")}
                               for index, row in enumerate(missed_candidates, start=1)]
                st.dataframe(missed_rows, use_container_width=True, hide_index=True)
                selected_missed = st.selectbox(
                    "확대해서 확인할 미탐지 제안", list(range(len(missed_candidates))),
                    format_func=lambda value: f"M{value + 1} · {missed_candidates[value].get('proposed_class', UNKNOWN_CLASS)}",
                    key=f"cc_archive2_quality_missed_{project.project_id}_{photo_id}",
                )
                missed_box = missed_candidates[selected_missed].get("bbox_xyxy")
                if isinstance(missed_box, (list, tuple)) and len(missed_box) == 4:
                    mx1, my1, mx2, my2 = [int(round(value)) for value in missed_box]
                    left, top_px = max(0, min(_image.width, mx1)), max(0, min(_image.height, my1))
                    right, bottom_px = max(0, min(_image.width, mx2)), max(0, min(_image.height, my2))
                    if right > left and bottom_px > top_px:
                        missed_crop = _image.convert("RGB").crop((left, top_px, right, bottom_px))
                        missed_crop.thumbnail((720, 720), Image.Resampling.LANCZOS)
                        st.image(missed_crop, caption=f"M{selected_missed + 1} 확대 · AI 제안, 사용자 확인 필요",
                                 width="stretch")
            st.caption("Qwen이 제안하지 않은 위치도 미탐지라고 단정할 수 없습니다. 결과는 후보 탐색이며 실제 정답 검증에는 사람이 사진을 확인해야 합니다.")

    run_key = f"cc_archive2_last_board_run_v3_{project.project_id}_{photo_id}"
    all_rows = _load_runs()
    if st.button("전체 PCB 모델 비교 실행", type="primary",
                  key=f"cc_archive2_run_board_v3_{project.project_id}_{photo_id}",
                  disabled=not selected_models):
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_client = OllamaClient(timeout_seconds=600, num_predict=256, keep_alive="10m", auto_start=True)
        availability = _probe_models(run_client, backend_code)
        st.session_state[availability_key] = availability
        try:
            runtime_version = run_client._request("GET", "/api/version").get("version")
        except Exception:
            runtime_version = None
        runtime_environment = {"python_version": sys.version.split()[0], "platform": platform.platform(),
                               "machine": platform.machine(),
                               "code_revision": os.environ.get("CIRCUITCARE_GIT_SHA")}
        progress = st.progress(0, text="비교 시작")
        status = st.status("선택한 소자 모델 비교 중", expanded=True)
        produced = []
        for index, model_key in enumerate(selected_models, start=1):
            spec = MODEL_CATALOG[model_key]
            model_id = spec["ollama_id"]
            model_state = availability.get(model_key, {})
            model_batch_size = MODEL_BATCH_SIZES.get(model_key, FULL_BOARD_BATCH_SIZE)
            common = {"run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                      "execution_mode": "REAL_MODEL", "backend": backend_code,
                      "model_key": model_key, "model_id": model_id, "runtime_id": model_id,
                      "ollama_model_id": model_id, "runtime_version": runtime_version,
                      "runtime_environment": runtime_environment, "project_id": project.project_id,
                      "board_id": project.board_id, "photo_id": photo_id, "component_id": component_id,
                      "detection_ids": detection_ids, "bbox_xyxy_by_id": bbox_by_id,
                      "detector_class_by_id": {row["detection_id"]: row["class_name"]
                                                for row in comparison_detections},
                      "ground_truth_by_id": ground_truth_by_id,
                      "image_sha256": image_sha, "input_mode": "FULL_BOARD",
                      "crop_px": "full_board", "padding_ratio": 0.0,
                      "output_budget": int(budget),
                      "output_budget_total": int(budget) * ((len(detection_ids) + model_batch_size - 1)
                                                              // model_batch_size),
                      "repeat_index": 0,
                      "prompt_version": PROMPT_VERSION, "prompt_fingerprint": prompt_fingerprint,
                      "schema_version": SCHEMA_VERSION, "schema_fingerprint": schema_fingerprint,
                      "normalization_version": OUTPUT_NORMALIZATION_VERSION,
                      "model_digest": model_state.get("digest"), "json_valid": False,
                      "schema_valid": False, "prediction": None, "prediction_subtype": None,
                      "predictions": [],
                      "confidence_uncalibrated": None, "decision": None, "evidence": None,
                      "uncertain": None, "normalization_notes": [], "raw_model_response": None,
                      "wall_ms": None, "load_ms": None, "prompt_eval_ms": None,
                      "eval_ms": None, "prompt_eval_count": None, "eval_count": None,
                      "tokens_per_second": None, "model_memory_mb": None}
            start_ns = time.perf_counter_ns()
            if model_state.get("status") != "READY":
                result = {**common, "status": "UNAVAILABLE", "wall_ms": None,
                          "error": model_state.get("reason") or
                                  f"미설치 상태입니다. `ollama pull {model_id}` 후 모델 설치 상태를 다시 확인하세요."}
            else:
                original_chat_json = run_client.chat_json
                original_request = run_client._request
                batch_review_rows = []
                raw_batches = []
                batch_errors = []
                batch_metrics = []
                try:
                    batch_count = (len(model_detections) + model_batch_size - 1) // model_batch_size
                    for batch_index, batch_start in enumerate(range(0, len(model_detections), model_batch_size), start=1):
                        global_batch = model_detections[batch_start:batch_start + model_batch_size]
                        global_model_ids = [row["detection_id"] for row in global_batch]
                        batch_ids = [str(local_index) for local_index in range(1, len(global_batch) + 1)]
                        local_to_global_id = dict(zip(batch_ids, global_model_ids))
                        batch_detections = [{**row, "detection_id": local_id, "component_id": local_id}
                                             for row, local_id in zip(global_batch, batch_ids)]
                        raw_response: dict[str, Any] = {}

                        def archive2_request(method: str, path: str, payload: dict | None = None) -> dict:
                            response = original_request(method, path, payload)
                            if method == "POST" and path == "/api/chat":
                                adapted, original_content, notes = _normalize_experiment_chat_http_response(response)
                                if original_content is not None:
                                    raw_response["content"] = original_content
                                if notes:
                                    raw_response["http_normalization_notes"] = notes
                                    raw_response["schema_valid_before_normalization"] = False
                                return adapted
                            return response

                        def archive2_chat_json(*args: Any, **kwargs: Any) -> dict[str, Any]:
                            response = original_chat_json(*args, **kwargs)
                            normalized = _normalize_experiment_response_batch(response, batch_ids)
                            raw_response.update(normalized.get("archive2_normalization", {}))
                            if raw_response.get("http_normalization_notes"):
                                raw_response["schema_valid_before_normalization"] = False
                            raw_response["notes"] = [*raw_response.get("http_normalization_notes", []),
                                                      *normalized.get("archive2_normalization", {}).get("notes", [])]
                            raw_response.setdefault("content", normalized.get("archive2_raw_content"))
                            return normalized

                        run_client._request = archive2_request
                        run_client.chat_json = archive2_chat_json
                        try:
                            reviewed_batch = review_components_with_qwen(
                                run_client, model_id, image_bytes, batch_detections, [],
                                classification_only=True, full_board_classification=True,
                                include_board_context=True, classification_output_budget=int(budget),
                                classification_think=False,
                            )
                            for review in reviewed_batch["class_reviews"]:
                                global_id = local_to_global_id.get(str(review.get("detection_id")))
                                if global_id is not None:
                                    batch_review_rows.append({**review, "detection_id": global_id})
                            batch_metrics.append(reviewed_batch.get("ollama_metrics", {}))
                        except Exception as exc:
                            batch_errors.append(f"배치 {batch_index}/{batch_count}: {type(exc).__name__}: {str(exc)[:240]}")
                        finally:
                            run_client.chat_json = original_chat_json
                            run_client._request = original_request
                        raw_batches.append({"detection_ids": global_model_ids, "request_ids": batch_ids,
                                            "response": raw_response.get("content"),
                                            "notes": raw_response.get("notes", []),
                                            "error": batch_errors[-1] if batch_errors and batch_errors[-1].startswith(
                                                f"배치 {batch_index}/") else None})
                        progress.progress((index - 1 + batch_index / batch_count) / len(selected_models),
                                          text=f"{MODEL_CATALOG[model_key]['label']} · 배치 {batch_index}/{batch_count}")

                    # MiniCPM sometimes returns one valid row for a five-box batch
                    # without an error. Retry only the omitted IDs as one-box requests.
                    if model_key == "minicpm_v47_1b":
                        replied_ids = {row["detection_id"] for row in batch_review_rows
                                       if row.get("response_status") != "NO_RESPONSE"
                                       and row.get("object_status") != "NO_RESPONSE"}
                        retry_global_ids = [model_id for model_id in model_detection_ids
                                            if model_id not in replied_ids]
                        for retry_index, global_id in enumerate(retry_global_ids, start=1):
                            original_detection = next(row for row in model_detections
                                                      if row["detection_id"] == global_id)
                            retry_detection = {**original_detection, "detection_id": "1", "component_id": "1"}
                            retry_ids = ["1"]
                            raw_response = {}

                            def archive2_retry_request(method: str, path: str,
                                                       payload: dict | None = None) -> dict:
                                response = original_request(method, path, payload)
                                if method == "POST" and path == "/api/chat":
                                    adapted, original_content, notes = _normalize_experiment_chat_http_response(response)
                                    if original_content is not None:
                                        raw_response["content"] = original_content
                                    if notes:
                                        raw_response["http_normalization_notes"] = notes
                                    return adapted
                                return response

                            def archive2_retry_chat(*args: Any, **kwargs: Any) -> dict[str, Any]:
                                response = original_chat_json(*args, **kwargs)
                                normalized = _normalize_experiment_response_batch(response, retry_ids)
                                raw_response.update(normalized.get("archive2_normalization", {}))
                                raw_response.setdefault("content", normalized.get("archive2_raw_content"))
                                return normalized

                            run_client._request = archive2_retry_request
                            run_client.chat_json = archive2_retry_chat
                            retry_error = None
                            try:
                                retry_result = review_components_with_qwen(
                                    run_client, model_id, image_bytes, [retry_detection], [],
                                    classification_only=True, full_board_classification=True,
                                    classification_output_budget=128, classification_think=False,
                                )
                                retry_row = retry_result["class_reviews"][0]
                                retry_row = {**retry_row, "detection_id": global_id}
                                if retry_row.get("response_status") != "NO_RESPONSE" and retry_row.get("object_status") != "NO_RESPONSE":
                                    batch_review_rows.append(retry_row)
                                    batch_metrics.append(retry_result.get("ollama_metrics", {}))
                            except Exception as exc:
                                retry_error = f"MiniCPM 단일 박스 재시도 {global_id}: {type(exc).__name__}: {str(exc)[:180]}"
                                batch_errors.append(retry_error)
                            finally:
                                run_client.chat_json = original_chat_json
                                run_client._request = original_request
                            raw_batches.append({"detection_ids": [global_id], "request_ids": retry_ids,
                                                "retry": True, "response": raw_response.get("content"),
                                                "notes": raw_response.get("notes", []), "error": retry_error})

                    returned_by_id = {row["detection_id"]: row for row in batch_review_rows}
                    batch_review_rows = [returned_by_id.get(model_id, {
                        "detection_id": model_id, "proposed_class": UNKNOWN_CLASS,
                        "object_status": "NO_RESPONSE", "response_status": "NO_RESPONSE",
                        "uncertain": True,
                    }) for model_id in model_detection_ids]
                    wall_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                    prediction_by_id = {row["detection_id"]: row for row in batch_review_rows}
                    prediction_rows = [{"detection_id": detection_id,
                                        "detector_class": next(item["class_name"] for item in comparison_detections
                                                                if item["detection_id"] == detection_id),
                                        "ground_truth": ground_truth_by_id.get(detection_id),
                                        "proposed_class": prediction_by_id[model_detection_id_by_real_id[detection_id]].get("proposed_class", UNKNOWN_CLASS),
                                        "proposed_subtype": prediction_by_id[model_detection_id_by_real_id[detection_id]].get("proposed_subtype"),
                                        "object_status": prediction_by_id[model_detection_id_by_real_id[detection_id]].get("object_status", "UNCERTAIN"),
                                        "response_status": prediction_by_id[model_detection_id_by_real_id[detection_id]].get("response_status", "RECEIVED"),
                                        "decision": prediction_by_id[model_detection_id_by_real_id[detection_id]].get("decision"),
                                        "uncertain": prediction_by_id[model_detection_id_by_real_id[detection_id]].get("uncertain", False),
                                        }
                                       for detection_id in detection_ids]
                    metric_sums = {key: sum(value.get(key, 0) for value in batch_metrics
                                            if isinstance(value.get(key), (int, float)))
                                   for key in ("load_duration_ns", "prompt_eval_duration_ns", "eval_duration_ns",
                                               "prompt_eval_count", "eval_count")}
                    eval_ms = _ns_ms(metric_sums["eval_duration_ns"])
                    eval_count = metric_sums["eval_count"]
                    partial = any(item["response_status"] == "NO_RESPONSE" for item in prediction_rows)
                    if not batch_metrics:
                        raise RuntimeError("모든 박스 배치가 실패했습니다 · " + "; ".join(batch_errors))
                    result = {**common, "status": "PARTIAL" if partial or batch_errors else "COMPLETE", "json_valid": True,
                              "schema_valid": all(bool(batch.get("response")) and not batch.get("error")
                                                  for batch in raw_batches),
                              "normalization_notes": [note for batch in raw_batches for note in batch.get("notes", [])],
                              "batch_errors": batch_errors, "raw_model_response": {"batches": raw_batches},
                              "prediction": None, "predictions": prediction_rows,
                              "decision": None, "evidence": None, "uncertain": any(
                                  item["uncertain"] for item in prediction_rows),
                              "wall_ms": round(wall_ms, 3),
                              "load_ms": _ns_ms(metric_sums["load_duration_ns"]),
                              "prompt_eval_ms": _ns_ms(metric_sums["prompt_eval_duration_ns"]),
                              "eval_ms": eval_ms, "prompt_eval_count": metric_sums["prompt_eval_count"],
                              "eval_count": eval_count,
                              "tokens_per_second": (eval_count / (eval_ms / 1000)
                                  if isinstance(eval_count, (int, float)) and eval_ms else None),
                              "model_memory_mb": None,
                              "experiment_fingerprint": hashlib.sha256(json.dumps({
                                  "model": model_id, "digest": model_state.get("digest"),
                                  "image": image_sha, "detection_ids": detection_ids,
                                  "bbox_by_id": bbox_by_id, "ocr": ocr_rows,
                                  "budget": budget, "prompt": prompt_fingerprint,
                                  "schema": schema_fingerprint,
                                  "normalization": OUTPUT_NORMALIZATION_VERSION,
                              }, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()}
                except Exception as exc:
                    result = {**common, "status": "FAILED", "wall_ms": round(
                        (time.perf_counter_ns() - start_ns) / 1_000_000, 3), "json_valid": False,
                        "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                        "batch_errors": batch_errors,
                        "raw_model_response": {"batches": raw_batches}}
            produced.append(result)
            all_rows.append(result)
            _save_runs(all_rows)
            progress.progress(index / len(selected_models), text=f"{index}/{len(selected_models)} · {result['status']}")
            status.write(f"{index}/{len(selected_models)} · {MODEL_CATALOG[model_key]['label']} · "
                         f"{result['status']} · {result.get('wall_ms') or 0:.0f}ms")
        status.update(label="모델 비교 완료", state="complete", expanded=False)
        st.session_state[run_key] = run_id
        st.success(f"{len(produced)}개 모델 결과를 실험 기록에 저장했습니다.")
        st.rerun()

    component_runs = [row for row in all_rows if row.get("project_id") == project.project_id
                      and row.get("photo_id") == photo_id and row.get("component_id") == component_id]
    run_ids = list(dict.fromkeys(row.get("run_id") for row in reversed(component_runs) if row.get("run_id")))
    if not run_ids:
        return
    last_run = st.session_state.get(run_key)
    run_index = run_ids.index(last_run) if last_run in run_ids else 0
    chosen_run = st.selectbox("이 전체 사진의 비교 기록", run_ids, index=run_index,
                              key=f"cc_archive2_history_board_v3_{project.project_id}_{photo_id}")
    run_rows = {row.get("model_key"): row for row in component_runs if row.get("run_id") == chosen_run}
    st.markdown("#### 모델별 결과")
    columns = st.columns(2)
    for index, model_key in enumerate(MODEL_CATALOG):
        row = run_rows.get(model_key)
        with columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"**{MODEL_CATALOG[model_key]['label']}**")
                if not row:
                    st.caption("이 실행에서 선택되지 않음")
                    continue
                if row.get("status") not in {"COMPLETE", "PARTIAL"}:
                    st.warning(row.get("error") or row.get("status"))
                    if row.get("status") == "UNAVAILABLE":
                        st.caption(f"설치 안내 · `ollama pull {MODEL_CATALOG[model_key]['ollama_id']}` 후 설치 상태를 확인하세요.")
                    continue
                prediction_table = [{
                    "박스 번호": model_detection_id_by_real_id.get(str(item.get("detection_id")), "—"),
                    "YOLO 원본 ID": item.get("detection_id"),
                    "YOLO 종류": item.get("detector_class"),
                    "모델 제안": item.get("proposed_class", UNKNOWN_CLASS),
                    "박스 판별": {"COMPONENT": "부품 후보 (AI)", "NOT_COMPONENT": "오탐 후보 (AI)",
                                  "UNCERTAIN": "실물 여부 미확정 (AI)", "NO_RESPONSE": "모델 무응답"}.get(
                                      item.get("object_status"), "실물 여부 불확실"),
                    "분류 상태": "미응답" if item.get("response_status") == "NO_RESPONSE" else
                                "종류 미확정" if item.get("proposed_class") == UNKNOWN_CLASS else "분류 제안",
                    "사용자 확인 정답": item.get("ground_truth") or "—",
                } for item in row.get("predictions", [])]
                if row.get("status") == "PARTIAL":
                    st.warning("일부 박스에 응답이 없습니다. 무응답은 unknown 분류로 간주하지 않았습니다.")
                st.caption("unknown + ‘부품 후보 (AI)’는 종류를 못 정한 박스, ‘오탐 후보 (AI)’는 실제 소자가 아닐 수 있다는 모델 제안입니다. ‘미확정’·‘무응답’은 판정 보류입니다. 모두 AI 제안이므로 확대 사진으로 사람이 확인해야 확정됩니다.")
                unanswered = sum(item.get("response_status") == "NO_RESPONSE"
                                 for item in row.get("predictions", []))
                if unanswered:
                    st.caption(f"응답 수신 {len(prediction_table) - unanswered}/{len(prediction_table)}개 · 무응답 {unanswered}개")
                if row.get("batch_errors"):
                    with st.expander("실패한 배치 원인", expanded=False):
                        for message in row["batch_errors"]:
                            st.error(message)
                st.markdown(f"### 전체 박스 분류 · {len(prediction_table)}개")
                if prediction_table:
                    st.dataframe(prediction_table, use_container_width=True, hide_index=True)
                    correct_n = sum(bool(item.get("ground_truth"))
                                    and item.get("ground_truth") == item.get("proposed_class")
                                    for item in row.get("predictions", []))
                    labeled_n = sum(bool(item.get("ground_truth")) for item in row.get("predictions", []))
                    if labeled_n:
                        st.caption(f"사용자 확정 정답 비교 · {correct_n}/{labeled_n} 일치")
                st.caption(f"전체 {row.get('wall_ms', 0):.0f}ms · Prefill {row.get('prompt_eval_ms') if row.get('prompt_eval_ms') is not None else '—'}ms · "
                           f"생성 {row.get('eval_ms') if row.get('eval_ms') is not None else '—'}ms · "
                           f"{row.get('tokens_per_second') if row.get('tokens_per_second') is not None else '—'} token/s · "
                           f"로드 {row.get('load_ms') if row.get('load_ms') is not None else '미보고'}ms")
                with st.expander("모델 원본 응답", expanded=False):
                    st.json(row.get("raw_model_response") or {})
                    if row.get("normalization_notes"):
                        st.caption("정규화 · " + "; ".join(row["normalization_notes"]))

    timing_rows = [{"모델": MODEL_CATALOG[key]["label"], "응답 ms": row["wall_ms"]}
                   for key, row in run_rows.items() if row.get("status") == "COMPLETE"
                   and isinstance(row.get("wall_ms"), (int, float))]
    if timing_rows:
        st.markdown("#### 응답 시간 비교")
        st.bar_chart(timing_rows, x="모델", y="응답 ms")
    if not any(row.get("ground_truth_by_id") for row in run_rows.values()):
        st.caption("사용자 확정 정답이 없거나 평가 대상이 아니므로 정확도는 표시하지 않습니다.")
    export_rows = [row for row in run_rows.values()]
    summary = _summarize(export_rows)
    json_bytes = json.dumps({"run_id": chosen_run, "summary": summary, "results": export_rows},
                            ensure_ascii=False, indent=2).encode("utf-8")
    csv_buffer = io.StringIO()
    csv_rows = []
    for row in export_rows:
        if isinstance(row.get("predictions"), list) and row["predictions"]:
            for prediction in row["predictions"]:
                if not isinstance(prediction, dict):
                    continue
                csv_row = {**row, **prediction,
                           "prediction": prediction.get("proposed_class"),
                           "prediction_subtype": prediction.get("proposed_subtype")}
                csv_row.pop("predictions", None)
                csv_rows.append(csv_row)
        else:
            csv_rows.append(row)
    fields = sorted({key for row in csv_rows for key in row})
    writer = csv.DictWriter(csv_buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(csv_rows)
    left, right = st.columns(2)
    left.download_button("이 비교 JSON 저장", json_bytes,
                         file_name=f"archive2-{chosen_run}-{component_id}.json",
                         mime="application/json", key=f"cc_archive2_json_v2_{chosen_run}_{component_id}")
    right.download_button("이 비교 CSV 저장", csv_buffer.getvalue().encode("utf-8-sig"),
                          file_name=f"archive2-{chosen_run}-{component_id}.csv", mime="text/csv",
                          key=f"cc_archive2_csv_v2_{chosen_run}_{component_id}")


def _ns_ms(value: Any) -> float | None:
    return float(value) / 1_000_000 if isinstance(value, (int, float)) and value >= 0 else None
