"""Independent real-PCB case registry, production-engine harness, and metrics.

Gold answers live under ``reference`` and are only read by ``score_run``;
``run_case`` receives the case's ``inference_input`` object only.
"""

from __future__ import annotations

import csv
from contextlib import nullcontext
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from circuitcare.frontend.board_component_registry import (
    build_board_component_registry, diagnosis_components,
)
from circuitcare.component_taxonomy import normalize_component_class
from circuitcare.symptom_diagnosis import load_knowledge
from circuitcare.symptom_diagnosis.diagnosis_state import build_overlay_state
from circuitcare.symptom_diagnosis.engine import diagnose


CASE_KINDS = {"REAL_CONFIRMED", "REAL_HEALTHY", "REAL_UNKNOWN", "REAL_MULTIPLE_FAULTS",
              "REAL_NON_COMPONENT", "SYNTHETIC", "SIMULATED", "MOCK"}
GOLD_STATUSES = {"CONFIRMED", "UNCONFIRMED", "UNKNOWN"}


def validate_case(case: dict[str, Any]) -> None:
    if not isinstance(case, dict) or not str(case.get("case_id") or "").strip():
        raise ValueError("PCB validation case requires case_id.")
    if case.get("case_kind") not in CASE_KINDS:
        raise ValueError("case_kind must explicitly distinguish real, synthetic, simulated, or mock data.")
    inference = case.get("inference_input")
    reference = case.get("reference")
    if not isinstance(inference, dict) or not isinstance(reference, dict):
        raise ValueError("inference_input and reference sections are required.")
    forbidden = {"ground_truth", "fault_type", "fault_parameter", "injected_fault", "reference"}
    if forbidden.intersection(inference):
        raise ValueError("Gold-answer fields must not appear in inference_input.")
    if not case.get("board_id"):
        raise ValueError("board_id is required.")
    if reference.get("adjudication_status", "UNKNOWN") not in GOLD_STATUSES:
        raise ValueError("Invalid gold adjudication_status.")
    if reference.get("failure_category") not in {None, "DETECTION_ERROR", "COMPONENT_IDENTITY_ERROR",
            "REGISTRY_ERROR", "SYMPTOM_INTERPRETATION_ERROR", "RULE_INFERENCE_ERROR",
            "MEASUREMENT_VALIDATION_ERROR", "UI_STATE_ERROR"}:
        raise ValueError("Unknown failure_category.")
    if case["case_kind"].startswith("REAL_"):
        if not case.get("photo_paths"):
            raise ValueError("Real cases require source photo paths.")
        if reference.get("adjudication_status") == "CONFIRMED" and not reference.get("evidence_sources"):
            raise ValueError("Confirmed real fault labels require evidence_sources.")


def register_cases(source: str | Path, destination: str | Path) -> Path:
    """Validate and copy case JSON, or import CSV rows with JSON in case_json."""
    source, destination = Path(source), Path(destination)
    if source.suffix.casefold() == ".csv":
        cases = []
        with source.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                payload = json.loads(row.get("case_json") or "{}")
                cases.append(payload)
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        cases = payload.get("cases", [payload])
    ids: set[str] = set()
    for case in cases:
        validate_case(case)
        if case["case_id"] in ids:
            raise ValueError(f"Duplicate PCB case ID: {case['case_id']}")
        ids.add(case["case_id"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"schema_version": 1, "cases": cases}, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    return destination


def run_case(case: dict[str, Any], *, detector: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
             reviewer: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
             llm: Callable[[dict[str, Any]], Any] | None = None) -> dict[str, Any]:
    """Run injected model stages and the production Registry/diagnose/evaluator path."""
    validate_case(case)
    started = time.perf_counter()
    inputs = case["inference_input"]
    stages: list[dict[str, Any]] = []

    def stage(name: str, operation: Callable[[], Any], *, input_summary: dict[str, Any] | None = None) -> Any:
        begin = time.perf_counter()
        try:
            value = operation()
            stages.append({"stage": name, "status": "OK", "duration_ms": round((time.perf_counter() - begin) * 1000, 3),
                           "input_summary": input_summary or {}, "output_summary": _summarize(value)})
            return value
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            stages.append({"stage": name, "status": "ERROR", "duration_ms": round((time.perf_counter() - begin) * 1000, 3),
                           "input_summary": input_summary or {}, "error": error})
            return {"_stage_error": error}

    detection = stage("yolo_detection", lambda: detector(inputs) if detector else {
        "status": "NOT_RUN", "detections": inputs.get("detections", [])}, input_summary={
            "mode": "REAL_MODEL" if detector else "MOCK_OR_CACHED", "photo_id": inputs.get("photo_id"),
            "photo_path_present": bool(inputs.get("photo_path"))})
    if detection.get("_stage_error"):
        return {"run_id": str(uuid.uuid4()), "case_id": case["case_id"], "case_kind": case["case_kind"],
                "status": "PIPELINE_FAILED", "failure_category": "DETECTION_ERROR",
                "error": detection["_stage_error"], "stage_trace": stages,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3), "metrics": "NOT_EVALUATED"}
    reviewed = stage("component_review", lambda: reviewer(inputs, detection) if reviewer else {
        "components": inputs.get("components", detection.get("detections", [])),
        "status": "MOCK_OR_PRE_REVIEWED"}, input_summary={"detection_count": len(detection.get("detections", [])),
            "mode": "REAL_OCR_QWEN" if reviewer else "MOCK_OR_PRE_REVIEWED"})
    if reviewed.get("_stage_error"):
        return {"run_id": str(uuid.uuid4()), "case_id": case["case_id"], "case_kind": case["case_kind"],
                "status": "PIPELINE_FAILED", "failure_category": "COMPONENT_IDENTITY_ERROR",
                "error": reviewed["_stage_error"], "stage_trace": stages,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3), "metrics": "NOT_EVALUATED"}
    snapshots = inputs.get("photo_snapshots", {})
    if not snapshots and inputs.get("photo_id") and reviewed.get("components"):
        snapshots = {str(inputs["photo_id"]): {
            "version": str(inputs.get("component_map_version", "validation-photo-v1")),
            "image_fingerprint": str(inputs.get("image_fingerprint") or detection.get("image_fingerprint") or ""),
            "review_version": str(inputs.get("review_version", "validation-review-v1")),
            "registry_components": _canonical_components(reviewed["components"]),
        }}
    if snapshots:
        registry = stage("board_registry", lambda: build_board_component_registry(
            {}, project_id=str(inputs.get("project_id", "validation")),
            board_id=str(case["board_id"]), photo_snapshots=snapshots),
            input_summary={"photo_count": len(snapshots), "component_count": sum(len(item.get("registry_components", item.get("components", []))) for item in snapshots.values())})
        components = [item.__dict__ for item in diagnosis_components(registry)]
        confirmed = [row.component_id for row in diagnosis_components(registry) if row.presence_status == "CONFIRMED"]
    else:
        components = _canonical_components(reviewed.get("components", []))
        registry = {"components": [], "photos": []}
        confirmed = [str(item.get("component_id")) for item in components
                     if item.get("presence_status") == "CONFIRMED" or item.get("identification_status") == "USER_CONFIRMED"]
    diagnosis_inputs = {key: inputs.get(key) for key in (
        "symptoms", "symptom_text", "observations", "inventory_complete", "measurements",
        "current_context", "silkscreen_labels")}
    diagnosis_inputs.update(components=components, confirmed_component_ids=confirmed)
    result = stage("production_rule_diagnosis", lambda: diagnose(**diagnosis_inputs), input_summary={
        "symptoms": diagnosis_inputs.get("symptoms"), "component_count": len(components),
        "confirmed_component_count": len(confirmed), "measurement_count": len(diagnosis_inputs.get("measurements") or [])})
    if result.get("_stage_error"):
        return {"run_id": str(uuid.uuid4()), "case_id": case["case_id"], "case_kind": case["case_kind"],
                "status": "PIPELINE_FAILED", "error": result["_stage_error"], "stage_trace": stages,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3), "metrics": "NOT_EVALUATED"}
    llm_result = None
    if llm is not None:
        llm_result = stage("llm_assist", lambda: llm(result))
        if llm_result.get("_stage_error"):
            # Rule-based diagnosis remains the valid production output.
            llm_result = {"status": "FAILED_RULES_RETAINED", "error": llm_result["_stage_error"]}
    measurement_comparison = None
    if "measurements_before" in inputs:
        before_inputs = dict(diagnosis_inputs)
        before_inputs["measurements"] = inputs.get("measurements_before") or []
        before_result = stage("pre_measurement_rule_diagnosis", lambda: diagnose(**before_inputs), input_summary={
            "measurement_count": len(before_inputs["measurements"]), "component_count": len(components)})
        if not before_result.get("_stage_error"):
            before_rank = [item.get("fault_id") for item in before_result.get("hypotheses", [])]
            after_rank = [item.get("fault_id") for item in result.get("hypotheses", [])]
            measurement_comparison = {"before_rank": before_rank, "after_rank": after_rank,
                                      "rank_changed": before_rank != after_rank}
    overlays = stage("current_result_overlay", lambda: build_overlay_state(
        project_id=str(inputs.get("project_id", "validation")), board_id=str(case["board_id"]),
        component_map_version=str(registry.get("version", "validation")), result=result),
        input_summary={"diagnosis_status": result.get("status"), "candidate_count": len(result.get("component_candidates", []))})
    return {
        "run_id": str(uuid.uuid4()), "case_id": case["case_id"], "case_kind": case["case_kind"],
        "detections": detection.get("detections", []),
        "reviewed_components": reviewed.get("components", []),
        "ocr_evidence": (reviewed.get("ocr_result") or {}).get("records", []),
        "qwen_review": reviewed.get("qwen_review"),
        "registry_component_count": len(registry.get("components", [])),
        "status": result.get("status"), "top_candidates": result.get("component_candidates", [])[:3],
        "hypotheses": result.get("hypotheses", [])[:3], "next_tests": result.get("next_tests", []),
        "measurement_evidence": result.get("measurement_evidence", []), "llm": llm_result,
        "measurement_comparison": measurement_comparison,
        "overlay": overlays, "stage_trace": stages,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "metrics": "NOT_EVALUATED",
    }


def _summarize(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": type(value).__name__}
    summary: dict[str, Any] = {}
    for key in ("status", "verification_status", "component_count", "image_width", "image_height"):
        if key in value and isinstance(value[key], (str, int, float, bool, type(None))):
            summary[key] = value[key]
    for key in ("detections", "components", "class_reviews", "duplicate_reviews", "group_reviews",
                "missed_component_candidates", "hypotheses", "component_candidates", "next_tests", "measurement_evidence"):
        if isinstance(value.get(key), (list, tuple)):
            summary[f"{key}_count"] = len(value[key])
    if "error" in value:
        summary["error"] = str(value["error"])
    return summary


def production_yolo_detector(inference_input: dict[str, Any]) -> dict[str, Any]:
    """Invoke the app's cached first-pass YOLO path for real-weight validation runs."""
    import io
    from PIL import Image
    from circuitcare.vision.component_recognition_panel import ensure_component_detection

    photo_path = inference_input.get("photo_path")
    if not photo_path:
        raise ValueError("inference_input.photo_path is required for --real-yolo.")
    image_bytes = Path(photo_path).read_bytes()
    import hashlib
    with Image.open(io.BytesIO(image_bytes)) as image:
        size = image.size

    state: dict[str, Any] = {}
    app_api = type("ValidationStreamlitAdapter", (), {
        "session_state": state, "spinner": lambda self, _message: nullcontext(),
    })()
    result, error = ensure_component_detection(
        app_api, image_bytes=image_bytes,
        project_id=str(inference_input.get("project_id", "validation")),
        photo_id=str(inference_input.get("photo_id", Path(photo_path).stem)),
        review_image_size=size,
    )
    if error or result is None:
        raise RuntimeError(error or "Production YOLO returned no result.")
    scope = next((key.removeprefix("cc_component_recognition_") for key in state
                  if key.startswith("cc_component_recognition_")), None)
    return {**result, "detections": state.get(f"cc_component_review_rows_{scope}", result.get("detections", [])),
            "review_scope": scope, "_app_state": state, "_image_bytes": image_bytes,
            "image_fingerprint": hashlib.sha256(image_bytes).hexdigest()}


def production_qwen_reviewer(inference_input: dict[str, Any], detection: dict[str, Any]) -> dict[str, Any]:
    """Use the app's real RapidOCR + Qwen review workflow after production YOLO."""
    from contextlib import nullcontext
    from circuitcare.vision.component_recognition_panel import _run_ai_review_workflow, review_aux_keys, review_keys

    scope = detection.get("review_scope")
    state = detection.get("_app_state")
    image_bytes = detection.get("_image_bytes")
    if not scope or not isinstance(state, dict) or not image_bytes:
        raise ValueError("Production YOLO review session is incomplete.")
    records_key = review_keys(scope)[1]
    ocr_key, qwen_key = review_aux_keys(scope)
    api = type("ValidationStreamlitAdapter", (), {
        "session_state": state, "spinner": lambda self, _message: nullcontext(),
        "warning": lambda self, _message: None,
    })()
    _run_ai_review_workflow(api, image_bytes=image_bytes, scope=scope,
        rows=state.get(records_key, []), records_key=records_key, ocr_key=ocr_key, qwen_key=qwen_key)
    return {"components": state.get(records_key, []), "qwen_review": state.get(qwen_key),
            "ocr_result": state.get(ocr_key), "status": "PRODUCTION_OCR_QWEN"}


def _canonical_components(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = row.get("component_type", row.get("class_name", row.get("소자 종류", "unknown")))
        status = row.get("identification_status", row.get("상태"))
        confirmed = status in {"USER_CONFIRMED", "확인"} or row.get("presence_status") == "CONFIRMED"
        box = row.get("bbox_xyxy") or [row.get(key) for key in ("x1", "y1", "x2", "y2")]
        result.append({**row, "component_id": str(row.get("component_id") or ""),
            "component_type": normalize_component_class(str(label)),
            "bbox_xyxy": box, "identification_status": "USER_CONFIRMED" if confirmed else "UNCONFIRMED",
            "presence_status": "CONFIRMED" if confirmed else "UNCONFIRMED",
            "component_type_status": "USER_CONFIRMED" if confirmed else "UNCONFIRMED"})
    return result


def score_run(case: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Score only adjudicated labels; unresolved/multi-fault labels are explicit."""
    validate_case(case)
    truth = case["reference"]
    detection = _score_detection(truth, run)
    if truth.get("adjudication_status") != "CONFIRMED":
        return {"case_id": case["case_id"], "status": "NOT_EVALUATED", "reason": "gold truth not confirmed",
                "detection": detection}
    roots = truth.get("root_cause_ids", [])
    if not roots:
        if case.get("case_kind") == "REAL_HEALTHY":
            predicted_fault = bool(run.get("hypotheses") or run.get("top_candidates"))
            return {"case_id": case["case_id"], "status": "SCORED", "healthy_case": True,
                    "healthy_correct": not predicted_fault, "false_fault_candidate": predicted_fault,
                    "abstained": str(run.get("status", "")).startswith("ABSTAIN"),
                    "detection": detection, "failure_category": truth.get("failure_category", "UNCLASSIFIED")}
        return {"case_id": case["case_id"], "status": "NOT_EVALUATED", "reason": "no confirmed root cause",
                "detection": detection}
    ranked = [str(row.get("fault_id") or row.get("fault_hypothesis") or "")
              for row in [*run.get("hypotheses", []), *run.get("top_candidates", [])]]
    ranked = list(dict.fromkeys(value for value in ranked if value))
    return {"case_id": case["case_id"], "status": "SCORED", "failure_category": truth.get("failure_category", "UNCLASSIFIED"),
            "root_cause_count": len(roots),
            "top1_hit": bool(ranked and ranked[0] in roots), "top3_hit": bool(set(ranked[:3]) & set(roots)),
            "abstained": str(run.get("status", "")).startswith("ABSTAIN"),
            "unsupported_specific_attribution": bool(run.get("top_candidates") and not run.get("measurement_evidence")),
            "useful_test_recommended": (bool(set(truth.get("useful_test_ids", [])) &
                {str(row.get("test_id")) for row in run.get("next_tests", [])})
                if truth.get("recommendation_labels_status") == "CONFIRMED" else None),
            "verdict_matches": _score_verdicts(truth, run),
            "rank_improvement": _rank_improvement(roots, run.get("measurement_comparison")),
            "detection": detection}


def _score_verdicts(reference: dict[str, Any], run: dict[str, Any]) -> list[bool] | None:
    if reference.get("measurement_verdict_labels_status") != "CONFIRMED":
        return None
    actual = {str(row.get("measurement_id")): str(row.get("verdict"))
              for row in run.get("measurement_evidence", [])}
    expected = {str(row.get("measurement_id")): str(row.get("expected_verdict"))
                for row in reference.get("measurement_verdict_labels", [])}
    return [actual.get(key) == value for key, value in expected.items()]


def _rank_improvement(roots: list[str], comparison: Any) -> int | None:
    if not isinstance(comparison, dict) or not roots:
        return None
    def best_rank(rows: list[str]) -> int | None:
        positions = [rows.index(root) for root in roots if root in rows]
        return min(positions) if positions else None
    before, after = best_rank(comparison.get("before_rank", [])), best_rank(comparison.get("after_rank", []))
    if before is None or after is None:
        return None
    return 1 if after < before else (-1 if after > before else 0)


def _score_detection(reference: dict[str, Any], run: dict[str, Any], *, iou_threshold: float = 0.5) -> dict[str, Any]:
    if reference.get("gold_components_status") != "CONFIRMED":
        return {"status": "NOT_EVALUATED", "tp": 0, "fp": 0, "fn": 0}
    gold = reference.get("gold_components", [])
    predicted = run.get("detections", [])
    candidates = []
    for pi, prediction in enumerate(predicted):
        box = prediction.get("bbox_xyxy") or [prediction.get(key) for key in ("x1", "y1", "x2", "y2")]
        for gi, label in enumerate(gold):
            target = label.get("bbox_xyxy")
            if not box or not target or len(box) != 4 or len(target) != 4:
                continue
            iou = _bbox_iou(box, target)
            if iou >= iou_threshold:
                candidates.append((iou, pi, gi,
                    normalize_component_class(str(prediction.get("class_name", prediction.get("소자 종류", "unknown"))))
                    == normalize_component_class(str(label.get("component_type", "unknown")))))
    candidates.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    tp = 0
    for _iou, pi, gi, class_match in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        if class_match:
            tp += 1
    return {"status": "EVALUATED", "tp": tp, "fp": len(predicted) - tp,
            "fn": len(gold) - tp, "iou_threshold": iou_threshold,
            "predicted_count": len(predicted), "gold_count": len(gold)}


def _bbox_iou(left: Any, right: Any) -> float:
    try:
        ax1, ay1, ax2, ay2 = map(float, left)
        bx1, by1, bx2, by2 = map(float, right)
    except (TypeError, ValueError):
        return 0.0
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def evaluate_runs(cases: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Return metrics with explicit denominators; missing/unknown labels stay unscored."""
    by_id = {run.get("case_id"): run for run in runs}
    scores = [score_run(case, by_id[case["case_id"]]) for case in cases if case["case_id"] in by_id]
    scored = [item for item in scores if item["status"] == "SCORED"]
    root_scored = [item for item in scored if not item.get("healthy_case")]
    abstention = [item for item in scores if item["case_id"] in by_id]
    detection_scored = [item["detection"] for item in scores if item.get("detection", {}).get("status") == "EVALUATED"]
    recommendation_rows = [item["useful_test_recommended"] for item in scored if item.get("useful_test_recommended") is not None]
    verdict_rows = [value for item in scored for value in (item.get("verdict_matches") or [])]
    ranking_rows = [item["rank_improvement"] for item in scored if item.get("rank_improvement") is not None]
    attribution_rows = [item for item in root_scored if "unsupported_specific_attribution" in item]
    tp = sum(item["tp"] for item in detection_scored)
    fp = sum(item["fp"] for item in detection_scored)
    fn = sum(item["fn"] for item in detection_scored)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    def ratio(key: str) -> dict[str, Any]:
        n = len(root_scored)
        hits = sum(bool(row.get(key)) for row in root_scored)
        value = hits / n if n else None
        # Wilson 95% interval; shown with denominator, including at small n.
        if n:
            from math import sqrt
            z = 1.959963984540054
            denom = 1 + z * z / n
            center = (value + z * z / (2 * n)) / denom
            margin = z * sqrt(value * (1 - value) / n + z * z / (4 * n * n)) / denom
            interval = [max(0.0, center - margin), min(1.0, center + margin)]
        else:
            interval = None
        return {"value": value, "n": n, "correct": hits, "confidence_interval_95_wilson": interval,
                "status": "EVALUATED" if n else "NOT_EVALUATED"}
    category_counts: dict[str, int] = {}
    for case in cases:
        category = case.get("reference", {}).get("failure_category")
        if category:
            category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "top1_accuracy": ratio("top1_hit"), "top3_hit_rate": ratio("top3_hit"),
        "healthy_board_specificity": {"value": (sum(bool(row.get("healthy_correct")) for row in scored if row.get("healthy_case")) /
                sum(bool(row.get("healthy_case")) for row in scored)) if any(row.get("healthy_case") for row in scored) else None,
            "n": sum(bool(row.get("healthy_case")) for row in scored),
            "status": "EVALUATED" if any(row.get("healthy_case") for row in scored) else "NOT_EVALUATED"},
        "diagnostic_abstention_rate": {
            "value": sum(bool(row.get("abstained")) for row in abstention) / len(abstention) if abstention else None,
            "n": len(abstention), "status": "EVALUATED" if abstention else "NOT_EVALUATED"},
        "unsupported_specific_attribution_rate": {"status": "EVALUATED" if attribution_rows else "NOT_EVALUATED",
            "n": len(attribution_rows),
            "value": sum(bool(item.get("unsupported_specific_attribution")) for item in attribution_rows) / len(attribution_rows) if attribution_rows else None},
        "detection_precision_recall_f1": {"status": "EVALUATED" if detection_scored else "NOT_EVALUATED",
            "n_cases": len(detection_scored), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "reason": None if detection_scored else "no independently adjudicated detection boxes"},
        "measurement_recommendation_utility": {"status": "EVALUATED" if recommendation_rows else "NOT_EVALUATED",
            "n": len(recommendation_rows), "useful_recommendation_rate": sum(recommendation_rows) / len(recommendation_rows) if recommendation_rows else None,
            "reason": None if recommendation_rows else "no independently verified case-level useful-test labels"},
        "verdict_agreement": {"status": "EVALUATED" if verdict_rows else "NOT_EVALUATED", "n": len(verdict_rows),
            "agreement": sum(verdict_rows) / len(verdict_rows) if verdict_rows else None,
            "reason": None if verdict_rows else "no adjudicated per-measurement verdict labels"},
        "measurement_rank_improvement": {"status": "EVALUATED" if ranking_rows else "NOT_EVALUATED", "n": len(ranking_rows),
            "improved": sum(value > 0 for value in ranking_rows), "unchanged": sum(value == 0 for value in ranking_rows),
            "worsened": sum(value < 0 for value in ranking_rows),
            "reason": None if ranking_rows else "no paired adjudicated before/after measurements"},
        "model_cost": {"status": "NOT_EVALUATED", "n": 0,
            "reason": "provider billing metadata unavailable"},
        "cases": scores,
        "failure_categories": category_counts,
        "improvement_priorities": sorted(category_counts, key=lambda key: (-category_counts[key], key)),
    }


def compare_before_after_measurement(case: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Run identical production logic without and with recorded measurements."""
    from copy import deepcopy
    baseline_case = deepcopy(case)
    baseline_case["inference_input"]["measurements"] = []
    before = run_case(baseline_case, **kwargs)
    after = run_case(case, **kwargs)
    before_rank = [row.get("fault_id") for row in before.get("hypotheses", [])]
    after_rank = [row.get("fault_id") for row in after.get("hypotheses", [])]
    return {"before": before, "after": after, "before_rank": before_rank, "after_rank": after_rank,
            "rank_changed": before_rank != after_rank}


def save_run(path: str | Path, run: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite validation run: {target}")
    target.write_text(json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target
