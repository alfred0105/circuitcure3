from copy import deepcopy

import pytest

from circuitcare.benchmark.pcb_validation import (
    evaluate_runs, run_case, score_run, validate_case,
)
from circuitcare.frontend.board_component_registry import (
    build_board_component_registry, current_source_observation_fingerprint,
    resolve_measurement_board_component,
)
from circuitcare.frontend.measurement_models import create_measurement_record
from circuitcare.dashboard.views.measurement_panel import _measurement_verification_gaps
from circuitcare.symptom_diagnosis.measurement_evaluator import evaluate_component_measurements
from circuitcare.symptom_diagnosis.measurement_validation import measurement_verification_gaps


def _profile():
    return {"profiles": {"resistor": {"tests": {"resistance": {
        "measurement_type": "resistance", "target_keywords": ["A-B"],
        "condition_keywords": ["power off"], "fault_relations": {
            "R_FAIL": {"TEST_NORMAL": "supports", "TEST_ABNORMAL": "refutes"}}}}}}}


def _measurement(value=1.5):
    return {"measurement_id": f"m-{value}", "component_id": "bc-r1", "board_component_id": "bc-r1",
            "identity_resolution_status": "USER_RECONFIRMED_VERIFIED", "component_identity_status": "USER_CONFIRMED",
            "fault_hypothesis": "R_FAIL", "test_id": "resistance", "measurement_target": "A-B",
            "measurement_location": "A-B", "operating_conditions": "power off", "reference_conditions": "power off",
            "location_verified": True, "source": "user_measurement", "verification_status": "USER_REPORTED",
            "uncertainty": 0.01, "uncertainty_source": "meter specification", "reference_min": 1,
            "reference_max": 2, "reference_source": "datasheet", "reference_source_id": "ds-r1",
            "reference_verification_status": "DATASHEET_VERIFIED", "component_type": "resistor",
            "measurement_type": "resistance", "value": value}


def test_ui_and_engine_share_measurement_requirements_without_node_mapping():
    row = create_measurement_record(project_id="p", board_id="b", component_id="c",
        board_component_id="bc-c", component_identity_status="USER_CONFIRMED", measurement_type="voltage",
        value="2", unit="V", source="user_measurement")
    gaps_ui = _measurement_verification_gaps(row)
    gaps_engine = measurement_verification_gaps(row, profiles=_profile())
    assert gaps_ui == gaps_engine
    assert "전기적 측정 지점 연결 미확인" not in gaps_ui
    assert row.mapping_status == "NODE_MAPPING_UNAVAILABLE"


def test_conflicting_valid_measurements_are_conflicted_but_missing_requirements_abstain():
    readings = [_measurement(1.5), _measurement(9.0)]
    result = evaluate_component_measurements(readings, profiles=_profile())
    assert next(iter(result.values()))["status"] == "CONFLICTED"
    invalid = _measurement(9.0)
    invalid["location_verified"] = False
    outcome = evaluate_component_measurements([invalid], profiles=_profile())
    assert next(iter(outcome.values()))["status"] == "INCONCLUSIVE"
    assert "측정 지점 확인 전" in next(iter(outcome.values()))["reason"]


def test_observation_fingerprint_blocks_same_photo_reused_id_after_redetection():
    base = {"version": "v1", "image_fingerprint": "photo-sha", "review_version": "review1",
            "registry_components": [{"component_id": "R1", "component_type": "resistor",
              "bbox_xyxy": [1, 2, 20, 30], "identification_status": "USER_CONFIRMED",
              "presence_status": "CONFIRMED", "component_type_status": "USER_CONFIRMED"}]}
    registry1 = build_board_component_registry({}, project_id="p", board_id="b", photo_snapshots={"photo": base})
    fp1 = current_source_observation_fingerprint(registry1, photo_id="photo", component_id="R1")
    current, status = resolve_measurement_board_component(registry1, photo_id="photo", component_id="R1",
        photo_fingerprint="photo-sha", observation_fingerprint=fp1)
    assert current and status == "VERIFIED_SOURCE_IDENTITY"
    changed = deepcopy(base)
    changed["version"] = "v2"
    changed["registry_components"][0]["bbox_xyxy"] = [30, 40, 50, 60]
    registry2 = build_board_component_registry({}, project_id="p", board_id="b", photo_snapshots={"photo": changed})
    fp2 = current_source_observation_fingerprint(registry2, photo_id="photo", component_id="R1")
    assert fp2 != fp1
    current, status = resolve_measurement_board_component(registry2, photo_id="photo", component_id="R1",
        photo_fingerprint="photo-sha", observation_fingerprint=fp1)
    assert current is None and status == "RECONFIRMATION_REQUIRED"


def _case():
    return {"case_id": "fixture-mock-1", "board_id": "board-1", "case_kind": "MOCK",
            "photo_paths": [], "inference_input": {"project_id": "project-1", "symptoms": ["NO_POWER"],
                "components": [], "measurements": []},
            "reference": {"adjudication_status": "UNKNOWN", "root_cause_ids": [], "evidence_sources": []}}


def test_harness_uses_production_diagnose_and_separates_truth():
    case = _case()
    validate_case(case)
    run = run_case(case)
    assert run["status"].startswith(("CANDIDATES", "ABSTAIN"))
    assert "reference" not in run
    assert score_run(case, run)["status"] == "NOT_EVALUATED"
    metrics = evaluate_runs([case], [run])
    assert metrics["top1_accuracy"]["status"] == "NOT_EVALUATED"
    leaked = deepcopy(case)
    leaked["inference_input"]["fault_type"] = "DO_NOT_LEAK"
    with pytest.raises(ValueError, match="Gold-answer"):
        validate_case(leaked)


def test_mock_model_stages_are_separate_and_llm_failure_keeps_rule_diagnosis():
    case = _case()
    run = run_case(case, detector=lambda _inputs: {"detections": [], "status": "MOCK"},
        reviewer=lambda _inputs, result: {"components": [], "status": "MOCK_REVIEW"},
        llm=lambda _result: (_ for _ in ()).throw(RuntimeError("offline")))
    assert run["status"] != "PIPELINE_FAILED"
    assert run["llm"]["status"] == "FAILED_RULES_RETAINED"
    assert [step["stage"] for step in run["stage_trace"]][:2] == ["yolo_detection", "component_review"]


def test_top3_scoring_and_detection_metrics_require_adjudicated_labels():
    case = _case()
    case["reference"] = {"adjudication_status": "CONFIRMED", "root_cause_ids": ["F1", "F2"],
        "gold_components_status": "CONFIRMED", "gold_components": [{"bbox_xyxy": [0, 0, 10, 10],
        "component_type": "resistor"}], "evidence_sources": ["repair_record"]}
    run = {"case_id": case["case_id"], "status": "CANDIDATES_AVAILABLE",
        "hypotheses": [{"fault_id": "F3"}, {"fault_id": "F2"}], "top_candidates": [],
        "detections": [{"bbox_xyxy": [0, 0, 10, 10], "class_name": "resistor"}],
        "measurement_evidence": []}
    score = score_run(case, run)
    assert score["top1_hit"] is False and score["top3_hit"] is True
    metrics = evaluate_runs([case], [run])
    assert metrics["detection_precision_recall_f1"]["f1"] == 1
    assert metrics["top3_hit_rate"]["n"] == 1
    assert metrics["top1_accuracy"]["confidence_interval_95_wilson"] is not None
