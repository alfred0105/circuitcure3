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
from itertools import product
from pathlib import Path
from typing import Any

from circuitcare.component_taxonomy import COMPONENT_CLASSES, UNKNOWN_CLASS, normalize_qwen_class
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

    proposed_raw = row.get("proposed_class", row.get("proposed_base_class"))
    proposed = normalize_qwen_class(proposed_raw) if proposed_raw is not None else UNKNOWN_CLASS
    if proposed_raw is None:
        notes.append("proposed_class missing; recorded as unknown")
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
    decision = row.get("decision")
    if decision not in {"suggest", "uncertain"}:
        decision = "uncertain"
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
    normalized = {"detection_id": detection_id, "decision": decision,
                  "proposed_class": proposed if proposed in {*COMPONENT_CLASSES, UNKNOWN_CLASS,
                                                                "other_component", "board_feature",
                                                                "not_component"} else UNKNOWN_CLASS,
                  "confidence": confidence if confidence is not None else 0.0,
                  "evidence": evidence[:500]}
    content = dict(content) if isinstance(content, dict) else {}
    content["class_reviews"] = [normalized]
    response["content"] = content
    response["archive2_normalization"] = {"schema_valid_before_normalization": schema_valid,
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
    """Run sequential, isolated experiments using cached detections and OCR."""
    from circuitcare.dashboard.views.unified_workspace import _cached_photo_pixels
    from circuitcare.vision.component_recognition_panel import review_aux_keys, review_keys, review_scope
    from circuitcare.vision.ai_review_service import (
        CLASS_ONLY_REVIEW_SCHEMA, _CLASS_ONLY_SYSTEM, review_components_with_qwen,
    )

    st.markdown("### Archive 2 · 경량 이미지 모델 비교")
    st.caption("실험은 저장된 YOLO/OCR 결과를 재사용합니다. 모델 제안은 원본 검수·Registry·진단 기록에 적용되지 않습니다.")
    photos = [row for row in photos if isinstance(row, dict) and row.get("image_id") and isinstance(row.get("data"), bytes)]
    if not photos:
        st.info("PCB 사진이 없습니다.")
        return
    photo_ids = [str(row["image_id"]) for row in photos]
    pkey = f"cc_archive2_photo_{project.project_id}"
    photo_id = st.selectbox("실험 사진", photo_ids, key=pkey,
                            format_func=lambda value: f"사진 {photo_ids.index(value) + 1} · {value}")
    photo = next(row for row in photos if str(row["image_id"]) == photo_id)
    try:
        _image, _source_size, image_size, image_bytes = _cached_photo_pixels(photo["data"])
    except Exception as exc:
        st.error(f"사진을 준비하지 못했습니다 · {exc}")
        return
    scope = review_scope(image_bytes, project.project_id, photo_id)
    prompt_fingerprint = hashlib.sha256(_CLASS_ONLY_SYSTEM.encode("utf-8")).hexdigest()
    schema_fingerprint = hashlib.sha256(json.dumps(
        CLASS_ONLY_REVIEW_SCHEMA, ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()
    _result_key, rows_key, _error_key = review_keys(scope)
    rows = st.session_state.get(rows_key, [])
    if not isinstance(rows, list) or not rows:
        st.warning("저장된 YOLO 탐지 결과가 없습니다. 먼저 PCB 검수 화면에서 사진을 탐지하세요. 여기서는 YOLO를 다시 실행하지 않습니다.")
        return
    ocr_key, _ = review_aux_keys(scope)
    ocr_state = st.session_state.get(ocr_key, {})
    ocr_rows = ocr_state.get("records", []) if isinstance(ocr_state, dict) else []
    active = [row for row in rows if isinstance(row, dict) and row.get("상태") != "제외"]
    confirmed = [row for row in active if row.get("상태") == "확인"]
    st.caption(f"같은 탐지 {len(active)}개 · 사용자 확인 정답 {len(confirmed)}개 · OCR 캐시 {len(ocr_rows)}개 · "
               f"사진 {image_size[0]}×{image_size[1]} px")
    if not ocr_rows:
        st.info("저장된 OCR 결과가 없습니다. 현재 결과 그대로 진행하거나 원본 검수 화면에서 OCR을 실행하세요.")
    by_id = {str(row.get("component_id")): row for row in active if row.get("component_id")}
    ids = list(by_id)
    default_ids = [str(row["component_id"]) for row in confirmed[:10] if row.get("component_id")]
    selected_ids = st.multiselect("비교할 박스", ids, default=default_ids,
                                 key=f"cc_archive2_boxes_{project.project_id}_{photo_id}")
    backend = st.selectbox("백엔드", ["Ollama", "MLX 현재 서버"],
                           key=f"cc_archive2_backend_{project.project_id}")
    backend_code = "MLX" if backend.startswith("MLX") else "Ollama"
    client = OllamaClient(timeout_seconds=4, num_predict=256, keep_alive="10m", auto_start=False,
                          backend="mlx" if backend_code == "MLX" else "ollama")
    availability_key = f"cc_archive2_model_status_{backend_code}"
    availability = st.session_state.get(availability_key, {})
    if st.button("설치·이미지 입력 지원 확인", key=f"cc_archive2_probe_{backend_code}"):
        availability = _probe_models(client, backend_code)
        st.session_state[availability_key] = availability
    status_rows = [{"모델": spec["label"], "식별자": spec["mlx_id"] if backend_code == "MLX" else spec["ollama_id"],
                    "상태": availability.get(key, {}).get("status", "확인 전"),
                    "사유": availability.get(key, {}).get("reason", "모델 확인을 실행하세요.")}
                   for key, spec in MODEL_CATALOG.items()]
    st.dataframe(status_rows, use_container_width=True, hide_index=True)
    ready = [key for key in MODEL_CATALOG if availability.get(key, {}).get("status") == "READY"]
    selected_models = st.multiselect("실행 모델", list(MODEL_CATALOG), default=ready,
                                     key=f"cc_archive2_models_{backend_code}",
                                     format_func=lambda value: MODEL_CATALOG[value]["label"])
    crop_sizes = st.multiselect("Crop 최대 변 길이", CROP_SIZES, default=[512],
                                key=f"cc_archive2_crop_{project.project_id}")
    paddings = st.multiselect("주변 여백 비율", PADDING_RATIOS, default=[0.25],
                              key=f"cc_archive2_padding_{project.project_id}",
                              format_func=lambda value: f"{value:.0%}")
    budgets = st.multiselect("출력 토큰 상한", OUTPUT_BUDGETS, default=[128],
                             key=f"cc_archive2_budget_{project.project_id}")
    repeats = st.number_input("소자별 반복 횟수", min_value=1, max_value=5, value=2, step=1,
                              key=f"cc_archive2_repeats_{project.project_id}")
    call_count = len(selected_ids) * len(selected_models) * len(crop_sizes) * len(paddings) * len(budgets) * int(repeats)
    st.caption(f"예상 {call_count}회 호출 · 순차 실행 · 모델별 입력 Crop/OCR/JSON Schema 공유")
    st.caption("전체 PCB는 모델에 보내지 않습니다. 전달되는 것은 선택 박스 Crop과 해당 영역 OCR 문맥뿐이며, 사용자 정답은 프롬프트에서 제외됩니다.")
    run_button = st.button("모델 비교 실행", type="primary", key=f"cc_archive2_run_{project.project_id}",
                           disabled=(not call_count or any(availability.get(key, {}).get("status") != "READY"
                                                          for key in selected_models)))
    all_rows = _load_runs()
    if run_button:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        confirmed_labels = {str(row.get("component_id")): str(row.get("소자 종류") or "unknown").casefold()
                            for row in confirmed}
        image_sha = hashlib.sha256(photo["data"]).hexdigest()
        ds_fingerprint = hashlib.sha256(json.dumps({
            "image_sha": image_sha,
            "boxes": [{"id": key, "bbox": [by_id[key].get(k) for k in ("x1", "y1", "x2", "y2")],
                       "label": confirmed_labels.get(key)} for key in selected_ids],
            "ocr": ocr_rows,
        }, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
        progress = st.progress(0, text="실험 시작")
        status = st.status("모델 비교 중", expanded=True)
        run_client = OllamaClient(timeout_seconds=600, num_predict=256, keep_alive="10m", auto_start=False,
                                  backend="mlx" if backend_code == "MLX" else "ollama")
        try:
            runtime_version = run_client._request("GET", "/api/version").get("version")
        except Exception:
            runtime_version = None
        runtime_environment = {"python_version": sys.version.split()[0], "platform": platform.platform(),
                               "machine": platform.machine(),
                               "code_revision": os.environ.get("CIRCUITCARE_GIT_SHA")}
        produced = []
        count, total = 0, max(call_count, 1)
        for model_key, crop_px, padding, budget in product(selected_models, crop_sizes, paddings, budgets):
            spec = MODEL_CATALOG[model_key]
            runtime_id = availability[model_key].get("runtime_id") or (
                "qwen3.5:9b" if backend_code == "MLX" else spec["ollama_id"])
            think_supported = bool(availability[model_key].get("think_supported"))
            for component_id in selected_ids:
                source = by_id[component_id]
                try:
                    bbox = [float(source[name]) for name in ("x1", "y1", "x2", "y2")]
                except (KeyError, TypeError, ValueError):
                    bbox = []
                detected_type = str(source.get("detector_class") or source.get("class_name") or "unknown")
                ground_truth = confirmed_labels.get(component_id)
                if ground_truth not in COMPONENT_CLASSES:
                    ground_truth = None
                small_smd = bool(bbox and max(bbox[2]-bbox[0], bbox[3]-bbox[1]) <= 64
                                 and ground_truth in {"resistor", "capacitor", "diode", "led", "inductor", "fuse", "transistor"})
                # Keep detector and OCR evidence, but never pass the confirmed answer or confirmed part knowledge.
                detection = {"component_id": component_id, "class_name": detected_type, "bbox_xyxy": bbox,
                             "confidence": source.get("신뢰도", source.get("confidence")),
                             "ocr_candidates": source.get("ocr_candidates", []),
                             "ocr_class_hints": source.get("ocr_class_hints", []),
                             "reference_designators": source.get("reference_designators", []),
                             "body_marking": source.get("body_marking"),
                             "marking_candidates": source.get("marking_candidates", []),
                             "possible_part_number": source.get("possible_part_number"),
                             "user_component_knowledge": []}
                experiment_fingerprint = hashlib.sha256(json.dumps({
                    "backend": backend_code, "model_id": runtime_id,
                    "model_digest": availability[model_key].get("digest"),
                    "image_sha256": image_sha, "component_id": component_id,
                    "bbox": bbox, "ocr": ocr_rows,
                    "prompt_version": PROMPT_VERSION, "prompt_fingerprint": prompt_fingerprint,
                    "schema_version": SCHEMA_VERSION, "schema_fingerprint": schema_fingerprint,
                    "crop_px": int(crop_px), "padding_ratio": float(padding),
                    "output_budget": int(budget), "think": False if think_supported else None,
                }, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
                for repeat_index in range(int(repeats)):
                    count += 1
                    start_ns = time.perf_counter_ns()
                    original_chat_json = run_client.chat_json
                    response_adaptation: dict[str, Any] = {}
                    def archive2_chat_json(*args: Any, **kwargs: Any) -> dict[str, Any]:
                        response = original_chat_json(*args, **kwargs)
                        normalized = _normalize_experiment_response(response, component_id)
                        response_adaptation.update(normalized.get("archive2_normalization", {}))
                        response_adaptation["raw_model_response"] = normalized.get("archive2_raw_content")
                        return normalized
                    run_client.chat_json = archive2_chat_json
                    try:
                        reviewed = review_components_with_qwen(
                            run_client, runtime_id, image_bytes, [detection], ocr_rows,
                            classification_only=True, include_board_context=False,
                            classification_crop_size_px=int(crop_px),
                            classification_padding_ratio=float(padding),
                            classification_output_budget=int(budget),
                            classification_think=False if think_supported else None,
                        )
                        wall_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                        prediction = reviewed["class_reviews"][0]
                        metrics = reviewed.get("ollama_metrics", {})
                        load_ms = _ns_ms(metrics.get("load_duration_ns"))
                        prefill_ms = _ns_ms(metrics.get("prompt_eval_duration_ns"))
                        decode_ms = _ns_ms(metrics.get("eval_duration_ns"))
                        eval_count = metrics.get("eval_count")
                        model_memory_mb = None
                        try:
                            ps = run_client._request("GET", "/api/ps")
                            target_id = spec["ollama_id"] if backend_code == "Ollama" else runtime_id
                            process = next((row for row in ps.get("models", [])
                                            if row.get("name") in {runtime_id, target_id}), None)
                            if process and isinstance(process.get("size"), (int, float)):
                                model_memory_mb = process["size"] / (1024 * 1024)
                        except Exception:
                            pass
                        result = {"run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                  "execution_mode": "REAL_MODEL", "backend": backend_code,
                                  "model_key": model_key, "model_id": runtime_id,
                                  "ollama_model_id": spec["ollama_id"],
                                  "runtime_id": runtime_id, "model_digest": availability[model_key].get("digest"),
                                  "runtime_version": runtime_version, "runtime_environment": runtime_environment,
                                  "experiment_fingerprint": experiment_fingerprint,
                                  "prompt_version": PROMPT_VERSION, "prompt_fingerprint": prompt_fingerprint,
                                  "schema_version": SCHEMA_VERSION, "schema_fingerprint": schema_fingerprint,
                                  "project_id": project.project_id, "board_id": project.board_id,
                                  "photo_id": photo_id, "image_sha256": image_sha,
                                  "dataset_fingerprint": ds_fingerprint, "component_id": component_id,
                                  "ground_truth": ground_truth,
                                  "ground_truth_subtype": source.get("confirmed_subtype") if ground_truth else None,
                                  "small_smd": small_smd, "crop_px": int(crop_px),
                                  "padding_ratio": float(padding), "output_budget": int(budget),
                                  "repeat_index": repeat_index, "wall_ms": round(wall_ms, 3),
                                  "load_ms": load_ms, "prompt_eval_ms": prefill_ms, "eval_ms": decode_ms,
                                  "prompt_eval_count": metrics.get("prompt_eval_count"), "eval_count": eval_count,
                                  "tokens_per_second": (eval_count / (decode_ms / 1000)
                                      if isinstance(eval_count, (int, float)) and decode_ms else None),
                                  "model_memory_mb": model_memory_mb,
                                  "first_run_label": "FIRST" if repeat_index == 0 else "WARM",
                                  "first_is_confirmed_cold": bool(repeat_index == 0 and load_ms and load_ms > 0),
                                  "think_parameter": "false" if think_supported else "omitted (model did not declare thinking)",
                                  "json_valid": True, "schema_valid": response_adaptation.get(
                                      "schema_valid_before_normalization", False),
                                  "normalization_notes": response_adaptation.get("notes", []),
                                  "raw_model_response": response_adaptation.get("raw_model_response"),
                                  "status": "COMPLETE",
                                  "prediction": prediction.get("proposed_class", UNKNOWN_CLASS),
                                  "prediction_subtype": prediction.get("proposed_subtype"),
                                  "confidence_uncalibrated": prediction.get("confidence"),
                                  "decision": prediction.get("decision"), "evidence": prediction.get("evidence"),
                                  "uncertain": prediction.get("decision") == "uncertain"}
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {str(exc)[:400]}"
                        unavailable = any(token in message.casefold() for token in
                                          ("unsupported", "not supported", "unknown model", "architecture", "projector"))
                        result = {"run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                  "execution_mode": "REAL_MODEL", "backend": backend_code,
                                  "model_key": model_key, "model_id": runtime_id,
                                  "ollama_model_id": spec["ollama_id"],
                                  "runtime_id": runtime_id, "project_id": project.project_id,
                                  "runtime_version": runtime_version, "runtime_environment": runtime_environment,
                                  "experiment_fingerprint": experiment_fingerprint,
                                  "prompt_version": PROMPT_VERSION, "prompt_fingerprint": prompt_fingerprint,
                                  "schema_version": SCHEMA_VERSION, "schema_fingerprint": schema_fingerprint,
                                  "board_id": project.board_id, "photo_id": photo_id,
                                  "image_sha256": image_sha, "dataset_fingerprint": ds_fingerprint,
                                  "component_id": component_id, "ground_truth": ground_truth,
                                  "crop_px": int(crop_px), "padding_ratio": float(padding),
                                  "output_budget": int(budget), "repeat_index": repeat_index,
                                  "wall_ms": round((time.perf_counter_ns() - start_ns) / 1_000_000, 3),
                                  "status": "UNAVAILABLE" if unavailable else "FAILED",
                                  "error": message, "json_valid": False,
                                  "normalization_notes": response_adaptation.get("notes", [])}
                        if unavailable:
                            availability[model_key] = {**availability[model_key],
                                "status": "UNAVAILABLE", "reason": message}
                            st.session_state[availability_key] = availability
                    finally:
                        run_client.chat_json = original_chat_json
                    produced.append(result)
                    all_rows.append(result)
                    _save_runs(all_rows)
                    progress.progress(count / total, text=f"{count}/{total} 호출 · {result['status']}")
                    status.write(f"{count}/{total} · {MODEL_CATALOG[model_key]['label']} · {component_id} · "
                                 f"{result['status']} · {result.get('wall_ms', 0):.0f} ms")
        status.update(label=f"비교 완료 · {len(produced)}개 기록", state="complete", expanded=False)
        st.session_state[f"cc_archive2_last_run_{project.project_id}"] = run_id
        st.success("실험 결과를 원본 검수·측정 데이터와 분리해 저장했습니다.")

    if not all_rows:
        return
    st.markdown("#### 결과")
    run_ids = list(dict.fromkeys(row.get("run_id") for row in all_rows if row.get("run_id")))
    last_run = st.session_state.get(f"cc_archive2_last_run_{project.project_id}")
    default_index = run_ids.index(last_run) if last_run in run_ids else len(run_ids) - 1
    chosen = st.selectbox("실험 실행", run_ids, index=default_index,
                          key=f"cc_archive2_show_run_{project.project_id}")
    run_rows = [row for row in all_rows if row.get("run_id") == chosen]
    summary = _summarize(run_rows)
    display = [{**row,
                "Accuracy": f"{row['accuracy']:.1%}" if row.get("accuracy") is not None else "NOT_EVALUATED",
                "Macro F1": f"{row['macro_f1']:.3f}" if row.get("macro_f1") is not None else "NOT_EVALUATED",
                "정답 수": row["sample_n"],
                "JSON 유효율": f"{row['json_valid_rate']:.1%}",
                "원본 스키마 일치율": f"{row['schema_valid_rate']:.1%}",
                "Unknown": f"{row['unknown_rate']:.1%}"}
               for row in summary]
    st.dataframe(display, use_container_width=True, hide_index=True)
    if not any(row.get("sample_n") for row in summary):
        st.info("이 실행에 사용자 확정 18종 정답이 없어 정확도·Macro F1은 NOT_EVALUATED입니다.")
    st.dataframe([{"모델": MODEL_CATALOG.get(row.get("model_key"), {}).get("label", row.get("model_key")),
                   "소자 ID": row.get("component_id"), "정답": row.get("ground_truth") or "미확정",
                   "예측": row.get("prediction", row.get("status")), "세부 유형": row.get("prediction_subtype"),
                   "근거": row.get("evidence"), "불확실": row.get("uncertain"),
                   "JSON 유효": row.get("json_valid"), "시간 ms": row.get("wall_ms"), "오류": row.get("error")}
                  for row in run_rows], use_container_width=True, hide_index=True)
    json_bytes = json.dumps({"run_id": chosen, "summary": summary, "results": run_rows},
                            ensure_ascii=False, indent=2).encode("utf-8")
    buffer = io.StringIO()
    fields = sorted({key for row in run_rows for key in row})
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(run_rows)
    left, right = st.columns(2)
    left.download_button("JSON 내보내기", json_bytes, file_name=f"archive2-{chosen}.json",
                         mime="application/json", key=f"cc_archive2_json_{chosen}")
    right.download_button("CSV 내보내기", buffer.getvalue().encode("utf-8-sig"),
                          file_name=f"archive2-{chosen}.csv", mime="text/csv",
                          key=f"cc_archive2_csv_{chosen}")
    eligible = [row for row in summary if row.get("sample_n", 0) >= 20
                and row.get("class_n", 0) >= 5 and row.get("accuracy") is not None]
    if eligible:
        best = max(eligible, key=lambda row: (row.get("macro_f1") or 0,
                                              -(row.get("mean_ms") or float("inf"))))
        st.info(f"확정 정답 20개 이상·5개 종류 이상 기준 우선 후보 · {best['model']} · Macro F1 {best['macro_f1']:.3f} · "
                f"평균 {best['mean_ms']:.0f} ms")
    else:
        st.caption("추천 모델은 설정별 사용자 확정 정답 20개·5개 종류 이상이 쌓일 때까지 표시하지 않습니다.")


def _ns_ms(value: Any) -> float | None:
    return float(value) / 1_000_000 if isinstance(value, (int, float)) and value >= 0 else None
