from __future__ import annotations

import base64
import io
import json

from PIL import Image

from circuitcare.local_llm import OllamaClient
from circuitcare.vision.ai_review_service import review_components_with_qwen
from circuitcare.vision.box_reconciliation import associate_ocr_with_components
from circuitcare.vision.box_reconciliation import (
    board_visible_review_rows, boxes_overlap_candidate, deduplicate_component_boxes, preserve_review_rows,
    same_component_box_pair,
)
from circuitcare.vision.component_recognition_panel import (
    INFERENCE_CONFIG, _review_fingerprint, duplicate_exclusion_enabled, invalidate_qwen_review,
    review_aux_keys,
)


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(output, format="PNG")
    return output.getvalue()


def test_ollama_chat_json_attaches_base64_image_to_user_message(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(client, "model_info", lambda _name: {"name": "qwen3.5:9b"})
    calls = []

    def request(method, path, body=None):
        calls.append((method, path, body))
        return {"model": "qwen3.5:9b", "message": {"content": "{\"ok\":true}"}, "done": True}

    monkeypatch.setattr(client, "_request", request)
    image_bytes = _png()
    result = client.chat_json("qwen3.5:9b", [{"role": "user", "content": "Review this."}], images=[image_bytes])

    user_message = calls[0][2]["messages"][0]
    assert user_message["images"] == [base64.b64encode(image_bytes).decode("ascii")]
    assert result["content"] == {"ok": True}


def test_qwen_review_validates_detection_ids_and_unverified_boxes():
    class FakeClient:
        def __init__(self):
            self.images = None
            self.schema = None

        def chat_json(self, _model, _messages, *, output_schema, images):
            self.images = images
            self.schema = output_schema
            return {"model": "qwen3.5:9b", "content": {
                "class_reviews": [
                    {"detection_id": "r1", "decision": "suggest", "proposed_class": "capacitor",
                     "confidence": .76, "evidence": "Two terminals and nearby C12 marking."},
                    {"detection_id": "missing-id", "decision": "suggest", "proposed_class": "IC",
                     "confidence": .99, "evidence": "Unknown target."},
                ],
                "duplicate_reviews": [
                    {"detection_ids": ["r1", "r2"], "decision": "same_component", "confidence": .8,
                     "evidence": "Boxes overlap the same body."},
                    {"detection_ids": ["r1", "not-real"], "decision": "same_component", "confidence": .8,
                     "evidence": "Invalid id."},
                ],
                "missed_component_candidates": [
                    {"image_index": 0, "bbox_norm_999": [600, 400, 800, 700], "proposed_class": "capacitor",
                     "confidence": .66, "evidence": "Visible unboxed body."},
                    {"image_index": 0, "bbox_norm_999": [300, 300, 300, 400], "proposed_class": "resistor",
                     "confidence": .9, "evidence": "Degenerate box."},
                ],
                "notes": ["Review each suggestion manually."],
            }}

    client = FakeClient()
    result = review_components_with_qwen(
        client, "qwen3.5:9b", _png(), [
            {"component_id": "r1", "class_name": "resistor", "confidence": .42, "bbox_xyxy": [5, 5, 35, 25]},
            {"component_id": "r2", "class_name": "resistor", "confidence": .31, "bbox_xyxy": [8, 7, 37, 27]},
        ], [{"raw_text": "C12", "bbox_xyxy": [38, 5, 48, 12], "confidence": .9,
             "reference_candidates": ["C12"]}],
    )

    assert len(client.images) == 4  # annotated photo, two detection crops, and a focused duplicate-pair crop
    assert client.schema is not None
    assert len(result["class_reviews"]) == 1
    assert len(result["duplicate_reviews"]) == 1
    assert len(result["missed_component_candidates"]) == 1
    assert result["verification_status"] == "AI_SUGGESTIONS_ONLY"
    candidate = result["missed_component_candidates"][0]
    assert candidate["review_status"] == "검수 필요"
    assert candidate["bbox_xyxy"][0] > 0


def test_silkscreen_reference_is_attached_as_class_hint_not_applied_as_fact():
    rows = associate_ocr_with_components(
        [{"component_id": "part-1", "class_name": "resistor", "bbox_xyxy": [0, 0, 20, 20]}],
        [{"text": "C12", "raw_text": "C12", "bbox_xyxy": [22, 4, 34, 11],
          "confidence": .93, "reference_candidates": ["C12"]}],
    )
    assert rows[0]["ocr_candidates"] == ["C12"]
    assert rows[0]["ocr_class_hints"] == ["capacitor"]
    assert rows[0]["reference_designators"] == ["C12"]
    assert rows[0]["class_name"] == "resistor"
    assert rows[0]["ocr_association_status"] == "ASSOCIATED"


def test_secondary_tile_size_is_supported_by_locator():
    assert INFERENCE_CONFIG["secondary_tile"] == 640


def test_containment_is_a_review_signal_but_small_true_component_is_not_suppressed():
    assert boxes_overlap_candidate([45, 45, 55, 55], [0, 0, 100, 100])
    assert not same_component_box_pair([45, 45, 55, 55], [0, 0, 100, 100])


def test_location_ensemble_suppresses_same_body_with_different_box_sizes():
    kept, suppressed = deduplicate_component_boxes([
        {"source": "parsa", "score": .78, "box": [10, 10, 110, 110]},
        {"source": "pcb_ai_doctor_v12n", "score": .72, "box": [20, 20, 100, 100]},
    ])
    assert len(kept) == 1
    assert len(suppressed) == 1
    assert kept[0]["source"] == "parsa"


def test_location_ensemble_keeps_partially_overlapping_distinct_boxes():
    kept, suppressed = deduplicate_component_boxes([
        {"source": "parsa", "score": .81, "box": [0, 0, 60, 60]},
        {"source": "pcb_ai_doctor_v12n", "score": .75, "box": [40, 0, 100, 60]},
    ])
    assert len(kept) == 2
    assert suppressed == []


def test_user_edited_yolo_box_and_class_survive_rerun():
    rows = preserve_review_rows([
        {"component_id": "yolo-1", "소자 종류": "capacitor", "상태": "검수 필요",
         "x1": 12, "y1": 12, "x2": 48, "y2": 48, "user_edited": True},
    ], [{"class_name": "resistor", "confidence": .9, "bbox_xyxy": [10, 10, 50, 50]}])
    assert rows[0]["component_id"] == "yolo-1"
    assert rows[0]["소자 종류"] == "capacitor"
    assert rows[0]["x1"] == 12


def test_qwen_still_checks_full_photo_when_yolo_has_no_boxes():
    class EmptyClient:
        def __init__(self):
            self.called = False

        def chat_json(self, _model, messages, *, output_schema, images):
            self.called = True
            payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
            assert payload["detector_proposals"] == []
            return {"model": "qwen3.5:9b", "content": {
                "class_reviews": [], "duplicate_reviews": [],
                "missed_component_candidates": [{
                    "image_index": 0, "bbox_norm_999": [200, 200, 600, 600],
                    "proposed_class": "IC", "confidence": .7, "evidence": "Visible IC body.",
                }], "notes": [],
            }}

    client = EmptyClient()
    result = review_components_with_qwen(client, "qwen3.5:9b", _png(), [], [])
    assert client.called
    assert len(result["missed_component_candidates"]) == 1


def test_qwen_receives_overlapping_native_resolution_tiles_and_maps_tile_box():
    class TileClient:
        def chat_json(self, _model, _messages, *, output_schema, images):
            assert len(images) > 1
            return {"model": "qwen3.5:9b", "content": {
                "class_reviews": [], "duplicate_reviews": [],
                "missed_component_candidates": [{
                    "image_index": 1, "bbox_norm_999": [100, 100, 300, 300],
                    "proposed_class": "capacitor", "confidence": .7, "evidence": "Small part in tile.",
                }], "notes": [],
            }}

    output = io.BytesIO()
    Image.new("RGB", (1800, 700), "white").save(output, format="PNG")
    result = review_components_with_qwen(TileClient(), "qwen3.5:9b", output.getvalue(), [], [])
    assert result["review_tiles"] == 2
    candidate = result["missed_component_candidates"][0]["bbox_xyxy"]
    assert candidate[0] > 0 and candidate[2] > candidate[0]


def test_cached_qwen_review_is_invalidated_when_source_box_changes():
    rows = [{"component_id": "r1", "소자 종류": "resistor", "상태": "검수 필요",
             "x1": 1, "y1": 1, "x2": 10, "y2": 10}]
    before = _review_fingerprint(rows, None)
    rows[0]["x1"] = 3
    assert _review_fingerprint(rows, None) != before
    _ocr_key, qwen_key = review_aux_keys("scope")
    session_state = {qwen_key: {"class_reviews": ["stale"]}}
    invalidate_qwen_review(session_state, "scope")
    assert qwen_key not in session_state


def test_qwen_duplicate_exclusion_is_only_enabled_for_confident_same_component():
    pair = ([10, 10, 110, 110], [20, 20, 100, 100])
    far_pair = ([0, 0, 20, 20], [500, 500, 520, 520])
    assert duplicate_exclusion_enabled({"decision": "same_component", "confidence": .7}, *pair)
    assert not duplicate_exclusion_enabled({"decision": "same_component", "confidence": .99}, *far_pair)
    assert not duplicate_exclusion_enabled({"decision": "different_components", "confidence": .99}, *pair)
    assert not duplicate_exclusion_enabled({"decision": "same_component", "confidence": .5}, *pair)


def test_qwen_duplicate_claim_for_far_apart_objects_is_rejected():
    class FarPairClient:
        def chat_json(self, _model, _messages, *, output_schema, images):
            return {"model": "qwen3.5:9b", "content": {
                "class_reviews": [],
                "duplicate_reviews": [{"detection_ids": ["r1", "r20"], "decision": "same_component",
                                       "confidence": .99, "evidence": "False visual claim."}],
                "missed_component_candidates": [], "notes": [],
            }}

    detections = [
        {"component_id": "r1", "class_name": "resistor", "confidence": .8, "bbox_xyxy": [10, 10, 40, 30]},
        {"component_id": "r20", "class_name": "resistor", "confidence": .8, "bbox_xyxy": [700, 500, 730, 520]},
    ]
    photo = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(photo, format="PNG")
    result = review_components_with_qwen(FarPairClient(), "qwen3.5:9b", photo.getvalue(), detections, [])
    assert result["duplicate_reviews"] == []


def test_user_excluded_duplicate_stays_hidden_after_new_detection_matches():
    old_rows = [
        {"component_id": "active", "소자 종류": "IC", "상태": "검수 필요",
         "x1": 10, "y1": 10, "x2": 110, "y2": 110},
        {"component_id": "duplicate", "소자 종류": "IC", "상태": "제외",
         "x1": 20, "y1": 20, "x2": 100, "y2": 100, "duplicate_of": "active"},
    ]
    rows = preserve_review_rows(old_rows, [{"class_name": "unknown", "confidence": .9,
                                           "bbox_xyxy": [12, 12, 108, 108]}])
    assert {row["component_id"] for row in rows} == {"active", "duplicate"}
    assert next(row for row in rows if row["component_id"] == "duplicate")["상태"] == "제외"
    assert [row["component_id"] for row in board_visible_review_rows(rows)] == ["active"]
