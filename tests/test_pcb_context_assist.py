from circuitcare.frontend.models import ImageRecord
from circuitcare.symptom_diagnosis import load_knowledge
from circuitcare.symptom_diagnosis.llm_assist import (
    select_follow_up_question, suggest_pcb_fault_candidates,
)
from circuitcare.symptom_diagnosis.candidate_fusion import fuse_inspection_candidates
from circuitcare.symptom_diagnosis.pcb_context import build_pcb_context
from circuitcare.dashboard.views.symptom_diagnosis_panel import (
    _combined_observations, _registry_valid_suspects, _save_follow_up_answer,
)


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def chat_json(self, model, messages, *, output_schema):
        self.calls.append((model, messages, output_schema))
        return {"content": self.content, "model": model}


def test_context_preserves_ids_photo_frames_and_unverified_suspicions():
    rows = [
        {"component_id": "board-c1", "board_component_id": "board-c1", "photo_id": "front",
         "photo_local_component_id": "local-1", "coordinate_frame": "source_image_px",
         "bbox_xyxy": [10, 20, 30, 40], "component_type": "capacitor", "ocr_marking": "104",
         "presence_status": "CONFIRMED"},
        {"component_id": "board-c2", "photo_local_component_id": "local-1", "bbox_xyxy": [1, 2, 3, 4],
         "component_type": "resistor", "presence_status": "CONFIRMED"},
    ]
    context = build_pcb_context(
        components=rows, photo_records=[ImageRecord("front", image_dimensions=(100, 200)),
                                       ImageRecord("back", image_dimensions=(100, 200))],
        diagnosis={"functional_blocks": [], "component_candidates": [], "symptom_codes": ["NO_POWER"]},
        suspected_components=[{"board_component_id": "board-c1", "suspected_fault_mode": "CAPACITOR_SHORT",
                              "evidence_status": "UNVERIFIED"}],
        board_registry={"components": [{"board_component_id": "board-c1", "source_components": [{
            "component_id": "local-1", "photo_id": "front", "detection_id": "det-1",
            "coordinate_frame": "SOURCE_PHOTO_PIXEL", "bbox_xyxy": [10, 20, 30, 40],
            "component_type": "capacitor", "review_status": "CONFIRMED", "ocr_marking": "104",
            "source_observation_fingerprint": "obs-sha256",
        }]}]},
    )
    first, second = context["focus_components"]
    assert first["board_component_id"] == "board-c1"
    assert first["photo_local_component_id"] == "local-1"
    assert first["normalized_geometry"]["center_x"] == 0.2
    assert first["ocr_marking"] == "104"
    assert first["connectivity_status"] == "UNKNOWN"
    assert first["user_suspicion"]["evidence_status"] == "UNVERIFIED"
    assert first["source_observations"][0]["photo_local_component_id"] == "local-1"
    assert first["source_observations"][0]["normalized_geometry"]["center_x"] == 0.2
    assert first["source_observations"][0]["source_observation_fingerprint"] == "obs-sha256"
    assert second["photo_id"] is None and second["normalized_geometry"] is None
    assert context["symptom_codes"] == ["NO_POWER"]


def test_large_context_limits_focus_and_records_omissions():
    rows = [{"component_id": f"c{i}", "component_type": "resistor", "bbox_xyxy": [i, 1, i+1, 2],
             "presence_status": "CONFIRMED"} for i in range(180)]
    context = build_pcb_context(components=rows, photo_records=[], diagnosis={}, token_budget=600,
                                max_focus_components=10)
    assert len(context["focus_components"]) <= 10
    assert context["omitted_component_count"] >= 170
    assert context["summary"]["component_count"] == 180
    assert context["context_token_estimate"] <= context["token_budget"]


def test_qwen_candidate_schema_and_result_are_allowlisted_and_unverified():
    knowledge = load_knowledge()
    fault_id = next(key for key, row in knowledge["fault_modes"]["fault_modes"].items()
                    if "capacitor" in row.get("candidate_component_types", []))
    response = {"proposals": [{"component_id": "c1", "suspected_fault_id": fault_id,
        "functional_block": "UNKNOWN", "reason": "검토 후보", "supporting_evidence_ids": ["NO_POWER"],
        "opposing_evidence_ids": [], "missing_information": [], "recommended_next_action": "전원 차단 후 육안 확인",
        "uncertainty_status": "UNVERIFIED"}]}
    client = FakeClient(response)
    context = {"focus_components": [{"component_id": "c1", "type": "capacitor", "review_status": "CONFIRMED",
                                     "functional_roles": []}], "observations": {"NO_POWER": True}}
    result = suggest_pcb_fault_candidates(client, "qwen", pcb_context=context, diagnosis={}, knowledge=knowledge)
    assert result["proposals"][0]["component_id"] == "c1"
    assert result["proposals"][0]["evidence_status"] == "UNVERIFIED"
    assert result["proposals"][0]["rank_status"] == "NOT_MERGED"
    item_schema = client.calls[0][2]["properties"]["proposals"]["items"]["properties"]
    assert item_schema["component_id"]["enum"] == ["c1"]
    assert "FAULT_SUPPORTED" not in item_schema["uncertainty_status"]["enum"]


def test_qwen_rejects_wrong_fault_component_pair_and_unconfirmed_component():
    knowledge = load_knowledge()
    resistor_fault = next(key for key, row in knowledge["fault_modes"]["fault_modes"].items()
                          if "resistor" in row.get("candidate_component_types", []))
    response = {"proposals": [{"component_id": "c1", "suspected_fault_id": resistor_fault,
        "functional_block": "UNKNOWN", "reason": "", "supporting_evidence_ids": [],
        "opposing_evidence_ids": [], "missing_information": [], "recommended_next_action": "",
        "uncertainty_status": "UNVERIFIED"}]}
    for kind, review in [("capacitor", "CONFIRMED"), ("resistor", "UNCONFIRMED")]:
        context = {"focus_components": [{"component_id": "c1", "type": kind, "review_status": review,
                                         "functional_roles": []}], "observations": {}}
        result = suggest_pcb_fault_candidates(FakeClient(response), "qwen", pcb_context=context,
                                              diagnosis={}, knowledge=knowledge)
        assert result["proposals"] == []


def test_qwen_can_propose_unreviewed_detection_as_review_task_only():
    knowledge = load_knowledge()
    capacitor_fault = next(key for key, row in knowledge["fault_modes"]["fault_modes"].items()
                           if "capacitor" in row.get("candidate_component_types", []))
    response = {"proposals": [{"component_id": "board-c1", "suspected_fault_id": capacitor_fault,
        "functional_block": "UNKNOWN", "reason": "육안 재검수 필요", "supporting_evidence_ids": [],
        "opposing_evidence_ids": [], "missing_information": ["소자 종류 확인"],
        "recommended_next_action": "박스 종류 확인", "uncertainty_status": "UNVERIFIED"}]}
    context = {"focus_components": [{"component_id": "board-c1", "type": "unknown",
        "candidate_component_types": ["capacitor"], "review_status": "DETECTED_UNVERIFIED",
        "component_type_status": "UNCONFIRMED"}], "observations": {}}
    result = suggest_pcb_fault_candidates(FakeClient(response), "qwen", pcb_context=context,
                                          diagnosis={}, knowledge=knowledge)
    assert len(result["proposals"]) == 1
    assert result["proposals"][0]["verification_status"] == "NEEDS_COMPONENT_REVIEW"
    assert result["proposals"][0]["evidence_status"] == "UNVERIFIED"


def test_candidate_fusion_merges_pair_without_upgrading_qwen_verdict_or_score():
    rule = {"component_id": "bc-1", "fault_hypothesis": "CAP_OPEN", "score": 1.5,
            "candidate_rank": 4, "visual_status": "SUSPECTED", "measurement_verdict": "non_diagnostic",
            "evidence_ids": ["OBS_A"], "fault_title": "커패시터 개방"}
    proposal = {"component_id": "bc-1", "suspected_fault_id": "CAP_OPEN",
                "component_type": "capacitor", "reason": "검토 이유", "supporting_evidence_ids": ["OBS_A"],
                "opposing_evidence_ids": [], "recommended_next_action": "소자 확인",
                "component_review_status": "USER_CONFIRMED"}
    qwen_only = {**proposal, "component_id": "bc-2", "suspected_fault_id": "CAP_SHORT",
                 "component_review_status": "DETECTED_UNVERIFIED"}
    result = fuse_inspection_candidates(
        rule_candidates=[rule], qwen_proposals=[proposal, qwen_only],
        components=[{"component_id": "bc-1", "component_type": "capacitor", "presence_status": "CONFIRMED"},
                    {"component_id": "bc-2", "component_type": "unknown", "presence_status": "UNCONFIRMED"}],
        suspected_components=[{"board_component_id": "bc-1", "suspected_fault_mode": "CAP_OPEN",
                               "evidence_status": "UNVERIFIED"}],
        allowed_fault_ids={"CAP_OPEN", "CAP_SHORT"},
    )
    assert len([row for row in result["candidates"] if row["component_id"] == "bc-1"]) == 1
    merged = result["candidates"][0]
    assert set(merged["candidate_source"]) == {"RULE_BASELINE", "QWEN_UNVERIFIED", "USER_SUSPICION"}
    assert merged["score"] == 1.5
    assert merged["visual_status"] == "SUSPECTED"
    unreviewed = next(row for row in result["candidates"] if row["component_id"] == "bc-2")
    assert unreviewed["verification_status"] == "NEEDS_COMPONENT_REVIEW"
    assert unreviewed["score"] is None


def test_question_answers_accumulate_unknown_is_not_evidence_and_cancel_is_local():
    state = {"manual_observations": {"VISIBLE_SOLDER_DAMAGE": True}, "answers": {}}
    q_a = {"test_id": "A", "type": "question", "question": "A?", "results": {
        "yes": {"observations": {"POWER_SOURCE_VERIFIED": True}},
        "no": {"observations": {"POWER_SOURCE_VERIFIED": False}},
    }}
    q_b = {"test_id": "B", "type": "question", "question": "B?", "results": {
        "yes": {"observations": {"BUZZER_VERIFIED_FAILED": True}},
        "no": {"observations": {"BUZZER_VERIFIED_FAILED": False}},
    }}
    _save_follow_up_answer(state, q_a, "예")
    _save_follow_up_answer(state, q_b, "아니오")
    rows = _combined_observations(state)
    assert {row["code"] for row in rows} >= {"POWER_SOURCE_VERIFIED", "BUZZER_VERIFIED_FAILED",
                                            "VISIBLE_SOLDER_DAMAGE"}
    _save_follow_up_answer(state, q_a, "모르겠음")
    rows = _combined_observations(state)
    assert not any(row.get("test_id") == "A" for row in rows)
    assert any(row.get("test_id") == "B" and row["value"] is False for row in rows)
    _save_follow_up_answer(state, q_a, "예")
    _save_follow_up_answer(state, q_a, "답변 취소")
    rows = _combined_observations(state)
    assert not any(row.get("test_id") == "A" for row in rows)
    assert any(row.get("test_id") == "B" for row in rows)
    assert any(row.get("source") == "user_report" and row["code"] == "VISIBLE_SOLDER_DAMAGE" for row in rows)


def test_spatial_relations_are_photo_local_and_never_claim_connectivity():
    rows = [
        {"component_id": "front-a", "photo_id": "front", "coordinate_frame": "source_image_px",
         "bbox_xyxy": [0, 0, 10, 10], "component_type": "resistor", "presence_status": "CONFIRMED"},
        {"component_id": "front-b", "photo_id": "front", "coordinate_frame": "source_image_px",
         "bbox_xyxy": [20, 0, 30, 10], "component_type": "capacitor", "presence_status": "CONFIRMED"},
        {"component_id": "back-c", "photo_id": "back", "coordinate_frame": "source_image_px",
         "bbox_xyxy": [0, 0, 10, 10], "component_type": "diode", "presence_status": "CONFIRMED"},
    ]
    context = build_pcb_context(components=rows,
        photo_records=[ImageRecord("front", image_dimensions=(100, 100)),
                       ImageRecord("back", image_dimensions=(100, 100))], diagnosis={})
    relations = context["focus_components"][0]["nearby_components"]
    assert [row["component_id"] for row in relations] == ["front-b"]
    assert relations[0]["relation_status"] == "SPATIAL_ONLY"
    assert context["connectivity_status"] == "UNKNOWN"


def test_registry_replacement_or_reobservation_invalidates_user_suspect_for_analysis():
    active = {"component_id": "R1", "board_component_id": "bc-1", "photo_id": "front",
              "photo_sha256": "old-photo", "selection_version": "old-observation",
              "evidence_status": "UNVERIFIED"}
    registry = {"photos": [{"photo_id": "front", "image_fingerprint": "new-photo"}],
                "components": [{"board_component_id": "bc-1", "presence_status": "CONFIRMED",
                                "source_components": [{"photo_id": "front", "component_id": "R1",
                                    "source_observation_fingerprint": "new-observation",
                                    "review_status": "USER_CONFIRMED"}]}]}
    current, stale = _registry_valid_suspects([active], registry)
    assert current == [] and stale == [active]


def test_follow_up_chooses_one_allowed_question_and_skips_answered():
    q1 = {"type": "question", "test_id": "ASK_POWER_SOURCE", "question": "전원 입력이 정상인가요?"}
    q2 = {"type": "question", "test_id": "ASK_OTHER_OUTPUTS", "question": "다른 출력도 동작하지 않나요?"}
    client = FakeClient({"test_id": q2["test_id"], "reason": "증상 범위 구분"})
    result = select_follow_up_question(client, "qwen", pcb_context={}, diagnosis={"next_tests": [q1, q2]},
                                       answered_test_ids=[])
    assert result["test_id"] == q2["test_id"]
    assert result["question_source"] == "QWEN_SELECTED_EXISTING_SAFE_QUESTION"
    assert select_follow_up_question(client, "qwen", pcb_context={}, diagnosis={"next_tests": [q1]},
                                     answered_test_ids=[q1["test_id"]]) is None
