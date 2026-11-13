from circuitcare.symptom_diagnosis import diagnose
from circuitcare.symptom_diagnosis.evidence_ledger import build_evidence_ledger
from circuitcare.symptom_diagnosis.functional_blocks import build_functional_block_groups


def test_role_groups_keep_separate_instances_without_positions_and_never_claim_connectivity():
    components = [
        {"component_id": "U1", "component_type": "ic"},
        {"component_id": "C1", "component_type": "capacitor"},
    ]
    roles = [
        {"component_id": "U1", "candidate_blocks": ["power", "control"],
         "role_evidence": {"power": ["OCR identity"]}},
        {"component_id": "C1", "candidate_blocks": ["power"],
         "role_evidence": {"power": ["nearby silkscreen"]}},
    ]
    groups = build_functional_block_groups(components, roles)
    power = [row for row in groups if row["role"] == "power"]
    assert {tuple(row["component_ids"]) for row in power} == {("U1",), ("C1",)}
    assert all(row["verification_status"] == "INFERRED" for row in power)
    assert all(row["connectivity_status"] == "NOT_INFERRED" for row in power)
    assert all(row["partition_status"] == "UNRESOLVED_NO_POSITION" for row in power)
    assert all(row["block_id"].startswith("block-") for row in power)


def test_position_clusters_same_role_blocks_but_keeps_role_verification_separate():
    components = [
        {"component_id": "J1", "board_component_id": "J1", "component_type": "connector",
         "component_type_status": "USER_CONFIRMED", "presence_status": "CONFIRMED",
         "bbox_xyxy": [0, 0, 10, 10]},
        {"component_id": "C1", "board_component_id": "C1", "component_type": "capacitor",
         "component_type_status": "USER_CONFIRMED", "presence_status": "CONFIRMED",
         "bbox_xyxy": [12, 0, 20, 10]},
        {"component_id": "J2", "board_component_id": "J2", "component_type": "connector",
         "component_type_status": "USER_CONFIRMED", "presence_status": "CONFIRMED",
         "bbox_xyxy": [800, 700, 810, 710]},
    ]
    roles = [
        {"component_id": "J1", "candidate_blocks": ["power"], "role_evidence": {"power": ["J1"]}},
        {"component_id": "C1", "candidate_blocks": ["power"], "role_evidence": {"power": ["C1"]}},
        {"component_id": "J2", "candidate_blocks": ["power"], "role_evidence": {"power": ["J2"]}},
    ]
    groups = build_functional_block_groups(components, roles)
    power = [row for row in groups if row["role"] == "power"]
    assert {tuple(row["component_ids"]) for row in power} == {("C1", "J1"), ("J2",)}
    assert all(row["verification_status"] == "INFERRED" for row in power)
    assert all(member["identity_status"] == "VERIFIED" and member["role_status"] == "INFERRED"
               for row in power for member in row["members"])


def test_evidence_ledger_preserves_reported_inferred_ai_and_conflicting_sources():
    rows = build_evidence_ledger(
        observations=[{"id": "o1", "code": "NO_POWER", "value": True,
                      "status": "observed", "source": "user_report"}],
        conflicts=[{"component_id": "R1", "code": "HOT", "values": [True, False]}],
        functional_blocks=[{"block_id": "b1", "role": "power", "component_ids": ["R1"]}],
        functional_roles=[{"component_id": "R1", "candidate_blocks": ["power"],
                           "role_evidence": {"power": ["kind and OCR hint"]}}],
        ai_suggestions=[{"component_id": "R1", "suspected_fault_id": "RESISTOR_FAILURE",
                         "source": "qwen"}],
        source_version="v1",
    )
    statuses = {row["kind"]: row["status"] for row in rows}
    assert statuses["observation"] == "USER_REPORTED"
    assert statuses["component_functional_role"] == "INFERRED"
    assert statuses["functional_block_membership"] == "INFERRED"
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
    driver_groups = [row for row in result["functional_block_groups"] if row["role"] == "driver"]
    assert {"K1", "Q1"} <= {component_id for row in driver_groups for component_id in row["component_ids"]}
    assert len(driver_groups) >= 2  # no positions: do not merge unrelated instances by role alone
    driver = next(row for row in driver_groups if "Q1" in row["component_ids"])
    q1_member = next(row for row in driver["members"] if row["component_id"] == "Q1")
    assert q1_member["evidence_status"] == "UNKNOWN"
    assert all(row["component_id"] != "Q1" for row in result["all_component_candidates"])

