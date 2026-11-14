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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
OUTPUT_BUDGETS = (128, 192, 256)
PROMPT_VERSION = "selected-class-only-v1"
SCHEMA_VERSION = "class-only-review-schema-v1"
OUTPUT_NORMALIZATION_VERSION = "archive2-class-output-v2"


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
        labeled = [row for row in completed if row.get("ground_truth") in COMPONENT_CLASSES]
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
            "unknown_rate": (sum(row.get("prediction") == UNKNOWN_CLASS for row in completed) / len(completed)
                             if completed else None),
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
        CLASS_ONLY_REVIEW_SCHEMA, _CLASS_ONLY_SYSTEM, _review_images, review_components_with_qwen,
    )

    st.markdown("### Archive 2 · 소자별 이미지 AI 비교")
    st.caption("사진 → YOLO 박스 → 모델 선택 → 비교 실행. 결과는 실험 기록에만 저장되고 검수·진단 데이터는 바뀌지 않습니다.")
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
    by_id = {str(row["component_id"]): row for row in active}
    component_ids = list(by_id)
    component_id = st.selectbox(
        "소자 박스", component_ids, key=f"cc_archive2_component_v2_{project.project_id}_{photo_id}",
        format_func=lambda value: f"{value} · YOLO {by_id[value].get('detector_class') or by_id[value].get('class_name') or by_id[value].get('소자 종류') or 'unknown'}",
    )
    source = by_id[component_id]
    try:
        bbox = [float(source[name]) for name in ("x1", "y1", "x2", "y2")]
    except (KeyError, TypeError, ValueError):
        bbox = []
    if len(bbox) != 4:
        st.error("선택한 YOLO 박스의 좌표가 올바르지 않아 Crop을 만들 수 없습니다.")
        return

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
                                  f"{availability.get(value, {}).get('status', '확인 중')}",
    )
    with st.expander("고급 설정", expanded=False):
        crop_px = st.selectbox("Crop 최대 변 길이", CROP_SIZES, index=CROP_SIZES.index(512),
                               key=f"cc_archive2_crop_v2_{project.project_id}")
        padding = st.selectbox("주변 여백 비율", PADDING_RATIOS, index=PADDING_RATIOS.index(.25),
                               key=f"cc_archive2_padding_v2_{project.project_id}",
                               format_func=lambda value: f"{value:.0%}")
        budget = st.selectbox("출력 상한", (128, 192, 256), index=1,
                              key=f"cc_archive2_budget_v2_{project.project_id}")

    detection = {"component_id": component_id,
                 "class_name": str(source.get("detector_class") or source.get("class_name")
                                   or source.get("소자 종류") or UNKNOWN_CLASS),
                 "bbox_xyxy": bbox, "confidence": source.get("신뢰도", source.get("confidence")),
                 "ocr_candidates": source.get("ocr_candidates", []),
                 "ocr_class_hints": source.get("ocr_class_hints", []),
                 "reference_designators": source.get("reference_designators", []),
                 "body_marking": source.get("body_marking"),
                 "marking_candidates": source.get("marking_candidates", []),
                 "possible_part_number": source.get("possible_part_number"),
                 "user_component_knowledge": []}
    image_sha = hashlib.sha256(photo["data"]).hexdigest()
    prompt_fingerprint = hashlib.sha256(_CLASS_ONLY_SYSTEM.encode("utf-8")).hexdigest()
    schema_fingerprint = hashlib.sha256(json.dumps(
        CLASS_ONLY_REVIEW_SCHEMA, ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()
    preview_images, _size, crop_manifest, *_ = _review_images(
        image_bytes, [{"detection_id": component_id, **detection}], max_crop_count=1,
        component_crop_padding_ratio=float(padding), component_crop_size_px=int(crop_px),
        include_full_image=False,
    )
    original_class = str(source.get("detector_class") or source.get("class_name")
                         or source.get("소자 종류") or UNKNOWN_CLASS)
    confirmed_class = (str(source.get("소자 종류") or "")
                       if source.get("상태") == "확인" else "")
    top, bottom = st.columns([1, 2])
    top.image(preview_images[0], caption=f"선택한 소자 Crop · {crop_manifest[0]['bbox_xyxy']}",
              use_container_width=True)
    bottom.markdown(f"**소자 ID** · `{component_id}`")
    bottom.markdown(f"**YOLO 결과** · {original_class} · 신뢰도 {source.get('신뢰도', source.get('confidence', '미제공'))}")
    if confirmed_class:
        bottom.caption(f"사용자 확정 종류(모델 입력에서 제외) · {confirmed_class}")
    else:
        bottom.caption("사용자 확정 정답 없음")
    bottom.caption(f"동일 Crop · OCR 문맥 {len(ocr_rows)}개 · {image_size[0]}×{image_size[1]}px 원본")

    run_key = f"cc_archive2_last_run_v2_{project.project_id}_{photo_id}_{component_id}"
    all_rows = _load_runs()
    if st.button("선택 소자 모델 비교 실행", type="primary",
                  key=f"cc_archive2_run_v2_{project.project_id}_{photo_id}_{component_id}",
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
            common = {"run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                      "execution_mode": "REAL_MODEL", "backend": backend_code,
                      "model_key": model_key, "model_id": model_id, "runtime_id": model_id,
                      "ollama_model_id": model_id, "runtime_version": runtime_version,
                      "runtime_environment": runtime_environment, "project_id": project.project_id,
                      "board_id": project.board_id, "photo_id": photo_id, "component_id": component_id,
                      "image_sha256": image_sha, "bbox_xyxy": bbox, "detector_class_input": original_class,
                      "ground_truth": confirmed_class.casefold() if confirmed_class.casefold() in COMPONENT_CLASSES else None,
                      "crop_px": int(crop_px), "padding_ratio": float(padding),
                      "output_budget": int(budget), "repeat_index": 0,
                      "prompt_version": PROMPT_VERSION, "prompt_fingerprint": prompt_fingerprint,
                      "schema_version": SCHEMA_VERSION, "schema_fingerprint": schema_fingerprint,
                      "normalization_version": OUTPUT_NORMALIZATION_VERSION,
                      "model_digest": model_state.get("digest"), "json_valid": False,
                      "schema_valid": False, "prediction": None, "prediction_subtype": None,
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
                raw_response: dict[str, Any] = {}
                original_chat_json = run_client.chat_json
                original_request = run_client._request

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

                run_client._request = archive2_request

                def archive2_chat_json(*args: Any, **kwargs: Any) -> dict[str, Any]:
                    response = original_chat_json(*args, **kwargs)
                    normalized = _normalize_experiment_response(response, component_id)
                    raw_response.update(normalized.get("archive2_normalization", {}))
                    if raw_response.get("http_normalization_notes"):
                        raw_response["schema_valid_before_normalization"] = False
                    raw_response["notes"] = [
                        *raw_response.get("http_normalization_notes", []),
                        *normalized.get("archive2_normalization", {}).get("notes", []),
                    ]
                    raw_response.setdefault("content", normalized.get("archive2_raw_content"))
                    return normalized

                run_client.chat_json = archive2_chat_json
                try:
                    reviewed = review_components_with_qwen(
                        run_client, model_id, image_bytes, [detection], ocr_rows,
                        classification_only=True, include_board_context=False,
                        classification_crop_size_px=int(crop_px),
                        classification_padding_ratio=float(padding),
                        classification_output_budget=int(budget), classification_think=False,
                    )
                    wall_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                    prediction = reviewed["class_reviews"][0]
                    metrics = reviewed.get("ollama_metrics", {})
                    eval_ms = _ns_ms(metrics.get("eval_duration_ns"))
                    eval_count = metrics.get("eval_count")
                    result = {**common, "status": "COMPLETE", "json_valid": True,
                              "schema_valid": raw_response.get("schema_valid_before_normalization", False),
                              "normalization_notes": raw_response.get("notes", []),
                              "raw_model_response": raw_response.get("content"),
                              "prediction": prediction.get("proposed_class", UNKNOWN_CLASS),
                              "prediction_subtype": prediction.get("proposed_subtype"),
                              "confidence_uncalibrated": (None if "confidence missing or invalid" in raw_response.get("notes", [])
                                                           else prediction.get("confidence")),
                              "decision": raw_response.get("original_decision") or prediction.get("decision"),
                              "evidence": prediction.get("evidence"),
                              "uncertain": raw_response.get(
                                  "model_uncertain", prediction.get("uncertain",
                                                                     prediction.get("decision") == "uncertain")),
                              "wall_ms": round(wall_ms, 3),
                              "load_ms": _ns_ms(metrics.get("load_duration_ns")),
                              "prompt_eval_ms": _ns_ms(metrics.get("prompt_eval_duration_ns")),
                              "eval_ms": eval_ms, "prompt_eval_count": metrics.get("prompt_eval_count"),
                              "eval_count": eval_count,
                              "tokens_per_second": (eval_count / (eval_ms / 1000)
                                  if isinstance(eval_count, (int, float)) and eval_ms else None),
                              "model_memory_mb": None,
                              "experiment_fingerprint": hashlib.sha256(json.dumps({
                                  "model": model_id, "digest": model_state.get("digest"),
                                  "image": image_sha, "component": component_id, "bbox": bbox,
                                  "ocr": ocr_rows, "crop": crop_px, "padding": padding,
                                  "budget": budget, "prompt": prompt_fingerprint,
                                  "schema": schema_fingerprint,
                                  "normalization": OUTPUT_NORMALIZATION_VERSION,
                              }, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()}
                except Exception as exc:
                    result = {**common, "status": "FAILED", "wall_ms": round(
                        (time.perf_counter_ns() - start_ns) / 1_000_000, 3), "json_valid": False,
                        "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                        "normalization_notes": [*raw_response.get("http_normalization_notes", []),
                                                 *raw_response.get("notes", [])],
                        "raw_model_response": raw_response.get("content")}
                finally:
                    run_client.chat_json = original_chat_json
                    run_client._request = original_request
            produced.append(result)
            all_rows.append(result)
            _save_runs(all_rows)
            progress.progress(index / len(selected_models), text=f"{index}/{len(selected_models)} · {result['status']}")
            status.write(f"{index}/{len(selected_models)} · {MODEL_CATALOG[model_key]['label']} · "
                         f"{result['status']} · {result.get('wall_ms') or 0:.0f}ms")
        status.update(label="모델 비교 완료", state="complete", expanded=False)
        st.session_state[run_key] = run_id
        st.success(f"{len(produced)}개 모델 결과를 실험 기록에 저장했습니다.")

    component_runs = [row for row in all_rows if row.get("project_id") == project.project_id
                      and row.get("photo_id") == photo_id and row.get("component_id") == component_id]
    run_ids = list(dict.fromkeys(row.get("run_id") for row in reversed(component_runs) if row.get("run_id")))
    if not run_ids:
        return
    last_run = st.session_state.get(run_key)
    run_index = run_ids.index(last_run) if last_run in run_ids else 0
    chosen_run = st.selectbox("이 소자의 비교 기록", run_ids, index=run_index,
                              key=f"cc_archive2_history_v2_{project.project_id}_{photo_id}_{component_id}")
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
                if row.get("status") != "COMPLETE":
                    st.warning(row.get("error") or row.get("status"))
                    if row.get("status") == "UNAVAILABLE":
                        st.caption(f"설치 안내 · `ollama pull {MODEL_CATALOG[model_key]['ollama_id']}` 후 설치 상태를 확인하세요.")
                    continue
                st.markdown(f"### 분류 제안 · {row.get('prediction') or UNKNOWN_CLASS}")
                if row.get("prediction_subtype"):
                    st.caption(f"세부 유형 · {row['prediction_subtype']}")
                st.write(row.get("evidence") or "판단 근거 없음")
                st.caption("불확실" if row.get("uncertain") else "제안 · 사용자 확인 필요")
                if row.get("confidence_uncalibrated") is not None:
                    st.caption(f"비보정 점수 · {row['confidence_uncalibrated']}")
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
    if not any(row.get("ground_truth") for row in run_rows.values()):
        st.caption("사용자 확정 정답이 없거나 평가 대상이 아니므로 정확도는 표시하지 않습니다.")
    export_rows = [row for row in run_rows.values()]
    summary = _summarize(export_rows)
    json_bytes = json.dumps({"run_id": chosen_run, "summary": summary, "results": export_rows},
                            ensure_ascii=False, indent=2).encode("utf-8")
    csv_buffer = io.StringIO()
    fields = sorted({key for row in export_rows for key in row})
    writer = csv.DictWriter(csv_buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(export_rows)
    left, right = st.columns(2)
    left.download_button("이 비교 JSON 저장", json_bytes,
                         file_name=f"archive2-{chosen_run}-{component_id}.json",
                         mime="application/json", key=f"cc_archive2_json_v2_{chosen_run}_{component_id}")
    right.download_button("이 비교 CSV 저장", csv_buffer.getvalue().encode("utf-8-sig"),
                          file_name=f"archive2-{chosen_run}-{component_id}.csv", mime="text/csv",
                          key=f"cc_archive2_csv_v2_{chosen_run}_{component_id}")


def _ns_ms(value: Any) -> float | None:
    return float(value) / 1_000_000 if isinstance(value, (int, float)) and value >= 0 else None
