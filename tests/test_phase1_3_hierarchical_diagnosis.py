from circuitcare.symptom_diagnosis import diagnose
from circuitcare.symptom_diagnosis.evidence_ledger import build_evidence_ledger
from circuitcare.symptom_diagnosis.functional_blocks import build_functional_block_groups


def test_role_groups_keep_members_but_never_claim_electrical_connectivity():
    components = [
        {"component_id": "U1", "component_type": "ic", "bbox_xyxy": [1, 1, 5, 5]},
        {"component_id": "C1", "component_type": "capacitor", "bbox_xyxy": [8, 1, 10, 3]},
    ]
    roles = [
        {"component_id": "U1", "candidate_blocks": ["power", "control"],
         "role_evidence": {"power": ["OCR identity"]}},
        {"component_id": "C1", "candidate_blocks": ["power"],
         "role_evidence": {"power": ["nearby silkscreen"]}},
    ]
    groups = build_functional_block_groups(components, roles)
    power = next(row for row in groups if row["role"] == "power")
    assert power["component_ids"] == ["C1", "U1"]
    assert power["verification_status"] == "INFERRED"
    assert power["connectivity_status"] == "NOT_INFERRED"
    assert power["block_id"].startswith("block-")


def test_evidence_ledger_preserves_reported_inferred_ai_and_conflicting_sources():
    rows = build_evidence_ledger(
        observations=[{"id": "o1", "code": "NO_POWER", "value": True,
                      "status": "observed", "source": "user_report"}],
        conflicts=[{"component_id": "R1", "code": "HOT", "values": [True, False]}],
        functional_blocks=[{"block_id": "b1", "role": "power", "component_ids": ["R1"]}],
        ai_suggestions=[{"component_id": "R1", "suspected_fault_id": "RESISTOR_FAILURE",
                         "source": "qwen"}],
        source_version="v1",
    )
    statuses = {row["kind"]: row["status"] for row in rows}
    assert statuses["observation"] == "USER_REPORTED"
    assert statuses["functional_role"] == "INFERRED"
    assert statuses["ai_suggestion"] == "AI_SUGGESTED"
    assert statuses["conflict"] == "CONFLICTED"
    assert all(row["source_version"] == "v1" for row in rows)


def test_same_type_parts_remain_as_an_unresolved_candidate_group():
    parts = [
        {"component_id": "C1", "component_type": "capacitor", "component_type_status": "USER_CONFIRMED"},
        {"component_id": "C2", "component_type": "capacitor", "component_type_status": "USER_CONFIRMED"},
    ]
    result = diagnose(symptoms=["LOCALIZED_HEATING"], components=parts,
                      confirmed_component_ids={"C1", "C2"})
    candidates = [row for row in result["all_component_candidates"]
                  if row.get("fault_hypothesis") == "CAPACITOR_SHORT"]
    assert {row["component_id"] for row in candidates} == {"C1", "C2"}
    assert {row["candidate_group_status"] for row in candidates} == {"UNRESOLVED_AMBIGUITY"}
    assert len({row["candidate_group_id"] for row in candidates}) == 1
    assert set(result["component_candidate_groups"][0]["component_ids"]) == {"C1", "C2"}


def test_hierarchical_hypotheses_and_evidence_gaps_do_not_claim_fault_probability():
    resistor = {"component_id": "R7", "component_type": "resistor",
                "component_type_status": "USER_CONFIRMED"}
    result = diagnose(symptoms=["LOCALIZED_HEATING"], components=[resistor],
                      confirmed_component_ids={"R7"})
    hierarchy = result["hierarchical_hypotheses"]
    assert hierarchy["components"]
    assert hierarchy["system"]
    assert all(row.get("score_is_probability") is False for row in hierarchy["components"])
    gap = next(row for row in result["evidence_gaps"] if row["fault_id"] == "RESISTOR_FAILURE")
    assert any(item.get("test_id") == "isolated_resistance" for item in gap["missing_evidence"])


def test_confirmed_components_and_unconfirmed_detections_stay_separate_in_ledger():
    result = diagnose(symptoms=["NO_POWER"], components=[
        {"component_id": "J1", "component_type": "connector", "component_type_status": "USER_CONFIRMED"},
        {"component_id": "C9", "component_type": "capacitor", "component_type_status": "UNCONFIRMED"},
    ], confirmed_component_ids={"J1", "C9"})
    identities = [row for row in result["evidence_ledger"] if row["kind"] == "component_identity"]
    by_id = {row["component_id"]: row["status"] for row in identities}
    assert by_id == {"J1": "VERIFIED", "C9": "UNKNOWN"}


def test_unconfirmed_parts_can_hint_blocks_but_never_become_component_fault_candidates():
    result = diagnose(symptoms=["RELAY_NO_ACTUATION"], components=[
        {"component_id": "K1", "component_type": "relay", "component_type_status": "USER_CONFIRMED",
         "presence_status": "CONFIRMED"},
        {"component_id": "Q1", "component_type": "transistor", "component_type_status": "UNCONFIRMED",
         "presence_status": "UNCONFIRMED"},
    ], confirmed_component_ids={"K1"})
    driver = next(row for row in result["functional_block_groups"] if row["role"] == "driver")
    assert {"K1", "Q1"} & set(driver["component_ids"])
    q1_member = next(row for row in driver["members"] if row["component_id"] == "Q1")
    assert q1_member["evidence_status"] == "UNKNOWN"
    assert all(row["component_id"] != "Q1" for row in result["all_component_candidates"])

