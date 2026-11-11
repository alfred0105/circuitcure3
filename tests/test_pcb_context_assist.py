from circuitcare.frontend.models import ImageRecord
from circuitcare.symptom_diagnosis import load_knowledge
from circuitcare.symptom_diagnosis.llm_assist import (
    select_follow_up_question, suggest_pcb_fault_candidates,
)
from circuitcare.symptom_diagnosis.pcb_context import build_pcb_context


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
        diagnosis={"functional_blocks": [], "component_candidates": []},
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


def test_large_context_limits_focus_and_records_omissions():
    rows = [{"component_id": f"c{i}", "component_type": "resistor", "bbox_xyxy": [i, 1, i+1, 2],
             "presence_status": "CONFIRMED"} for i in range(180)]
    context = build_pcb_context(components=rows, photo_records=[], diagnosis={}, token_budget=600,
                                max_focus_components=10)
    assert len(context["focus_components"]) <= 10
    assert context["omitted_component_count"] >= 170
    assert context["summary"]["component_count"] == 180


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
