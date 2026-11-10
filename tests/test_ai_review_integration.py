from __future__ import annotations

import base64
import io

from PIL import Image

from circuitcare.local_llm import OllamaClient
from circuitcare.vision.ai_review_service import review_components_with_qwen
from circuitcare.vision.box_reconciliation import associate_ocr_with_components


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
                    {"bbox_norm_999": [600, 400, 800, 700], "proposed_class": "capacitor",
                     "confidence": .66, "evidence": "Visible unboxed body."},
                    {"bbox_norm_999": [300, 300, 300, 400], "proposed_class": "resistor",
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

    assert len(client.images) == 3  # annotated photo and one crop for each uncertain proposal
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
    assert rows[0]["class_name"] == "resistor"
    assert rows[0]["ocr_association_status"] == "ASSOCIATED"
