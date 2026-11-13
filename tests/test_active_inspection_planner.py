from circuitcare.symptom_diagnosis.active_inspection_planner import plan_active_inspections
from circuitcare.symptom_diagnosis import diagnose


def _knowledge():
    return {
        "fault_modes": {"fault_modes": {
            "F1": {"recommended_tests": ["ASK_SPLIT"], "supporting_evidence": {"OBS_A": 1}},
            "F2": {"recommended_tests": ["ASK_SPLIT"], "opposing_evidence": {"OBS_A": 1}},
        }},
        "test_catalog": {"tests": {
            "ASK_SPLIT": {
                "name": "분리 질문", "type": "question", "risk_level": "none",
                "results": {
                    "yes": {"observations": {"OBS_A": True}},
                    "no": {"observations": {"OBS_A": False}},
                },
            },
            "VISUAL_SAFE": {
                "name": "전원 차단 후 외관 확인", "type": "visual_inspection",
                "risk_level": "low", "target_scope": "board",
                "requires_power_off": True, "requires_stored_energy_safe": True,
                "results": {},
            },
        }},
        "component_test_profiles": {"profiles": {}},
    }


def _plan(**overrides):
    payload = {
        "hypotheses": [{"fault_id": "F1", "score": 2}, {"fault_id": "F2", "score": 1}],
        "component_candidates": [],
        "functional_block_groups": [],
        "evidence_gaps": [],
        "measurements": [],
        "observations": {},
        "inspection_history": [],
        "knowledge": _knowledge(),
    }
    payload.update(overrides)
    return plan_active_inspections(**payload)


def test_planner_ranks_deterministic_question_by_hypothesis_separation_without_probability():
    result = _plan()
    assert result["status"] == "READY"
    item = result["candidates"][0]
    assert item["test_id"] == "ASK_SPLIT"
    assert item["separated_hypothesis_pairs"] == [["F1", "F2"]]
    assert item["score_is_probability"] is False
    assert "information_gain" not in item


def test_measurement_separation_requires_declared_relations_for_both_hypotheses():
    knowledge = _knowledge()
    knowledge["test_catalog"]["tests"]["OHMS"] = {
        "name": "저항 측정", "type": "measurement", "risk_level": "unknown", "results": {},
    }
    knowledge["fault_modes"]["fault_modes"]["F1"]["recommended_tests"] = ["OHMS"]
    knowledge["fault_modes"]["fault_modes"]["F2"]["recommended_tests"] = ["OHMS"]
    knowledge["component_test_profiles"]["profiles"] = {"resistor": {"tests": {"OHMS": {
        "fault_relations": {
            "F1": {"TEST_NORMAL": "supports", "TEST_ABNORMAL": "refutes"},
            "F2": {"TEST_NORMAL": "refutes", "TEST_ABNORMAL": "supports"},
        },
    }}}}
    candidates = [
        {"component_id": "R1", "component_type": "resistor", "fault_hypothesis": "F1"},
        {"component_id": "R1", "component_type": "resistor", "fault_hypothesis": "F2"},
    ]
    plan = _plan(knowledge=knowledge, component_candidates=candidates)
    measurement = next(item for item in plan["blocked"] if item["test_id"] == "OHMS")
    assert measurement["separated_hypothesis_pairs"] == [["F1", "F2"]]
    knowledge["component_test_profiles"]["profiles"]["resistor"]["tests"]["OHMS"]["fault_relations"].pop("F2")
    plan_without_second_relation = _plan(knowledge=knowledge, component_candidates=candidates)
    measurement = next(item for item in plan_without_second_relation["blocked"] if item["test_id"] == "OHMS")
    assert measurement["separated_hypothesis_pairs"] == []


def test_completed_inspection_is_not_recommended_again_without_retest():
    result = _plan(inspection_history=[{
        "board_component_id": None, "fault_id": "F1", "related_fault_ids": ["F1", "F2"],
        "test_id": "ASK_SPLIT", "result": "예", "verification_status": "INCONCLUSIVE",
    }])
    assert not result["candidates"]


def test_risky_inspection_is_blocked_until_power_and_stored_energy_are_confirmed():
    knowledge = _knowledge()
    knowledge["fault_modes"]["fault_modes"]["F1"]["recommended_tests"].append("VISUAL_SAFE")
    result = _plan(knowledge=knowledge)
    blocked = next(row for row in result["blocked"] if row["test_id"] == "VISUAL_SAFE")
    assert blocked["blocked_until_safe"] is True
    assert blocked["status"] == "NEEDS_SAFETY_OR_TEST_DETAILS"
    assert "잔류 에너지 안전 확인" in blocked["prerequisites"]


def test_question_answer_history_does_not_recommend_the_same_completed_check():
    result = _plan(inspection_history=[{
        "target_id": None, "related_fault_ids": ["F1", "F2"],
        "test_id": "ASK_SPLIT", "result": "예", "verification_status": "COMPLETE",
    }])
    assert not result["candidates"]


def test_missing_hypotheses_abstain_from_planning():
    result = _plan(hypotheses=[])
    assert result["status"] == "NO_HYPOTHESES"
    assert result["candidates"] == []


def test_production_diagnose_returns_inspection_plan_alongside_existing_rule_candidates():
    result = diagnose(symptoms=["NO_POWER"], observations={"POWER_SOURCE_VERIFIED": False})
    assert "active_inspection_plan" in result
    assert result["active_inspection_plan"]["score_is_probability"] is False
    assert result["hypotheses"]
