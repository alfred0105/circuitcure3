import json

from circuitcare.component_taxonomy import COMPONENT_CLASSES, QWEN_CLASSIFICATION_CLASSES, UNKNOWN_CLASS
from circuitcare.vision.ai_review_service import _review_images
from circuitcare.vision.model_comparison import (
    MODEL_CATALOG, _normalize_experiment_chat_http_response,
    _normalize_experiment_response, _probe_models, _summarize,
)
from PIL import Image
import io


class FakeModelClient:
    def __init__(self, models, metadata=None):
        self.models = models
        self.metadata = metadata or {}

    def list_models(self):
        return self.models

    def show_model(self, name):
        return self.metadata.get(name, {"capabilities": ["vision"]})


def test_probe_reports_uninstalled_or_nonvision_models_unavailable():
    qwen = MODEL_CATALOG["qwen35_9b"]["ollama_id"]
    lfm = MODEL_CATALOG["lfm3_vl_2b"]["ollama_id"]
    client = FakeModelClient(
        [{"name": qwen, "digest": "qwen-sha"}, {"name": lfm}],
        {qwen: {"capabilities": ["vision", "thinking"]}, lfm: {"capabilities": ["completion"]}},
    )
    result = _probe_models(client, "Ollama")
    assert result["qwen35_9b"]["status"] == "READY"
    assert result["qwen35_9b"]["think_supported"] is True
    assert result["lfm3_vl_2b"]["status"] == "UNAVAILABLE"
    assert result["qwen35_4b"]["status"] == "UNAVAILABLE"


def test_probe_reports_all_unavailable_when_service_is_unreachable():
    class Offline:
        def list_models(self):
            raise OSError("connection refused")

    result = _probe_models(Offline(), "Ollama")
    assert {row["status"] for row in result.values()} == {"UNAVAILABLE"}
    assert "연결 실패" in result["qwen35_9b"]["reason"]


def test_mlx_availability_does_not_claim_nonconfigured_models():
    result = _probe_models(FakeModelClient([{"name": "qwen3.5:9b"}]), "MLX")
    assert result["qwen35_9b"]["status"] == "READY"
    assert all(result[key]["status"] == "UNAVAILABLE"
               for key in ("qwen35_4b", "lfm3_vl_2b", "minicpm_v47_1b"))


def test_summary_with_no_confirmed_labels_does_not_invent_accuracy():
    rows = [{"status": "COMPLETE", "backend": "Ollama", "model_key": "qwen35_9b",
             "crop_px": 512, "padding_ratio": .25, "output_budget": 128,
             "prediction": UNKNOWN_CLASS, "ground_truth": None, "json_valid": True,
             "wall_ms": 100.0, "repeat_index": 0}]
    summary = _summarize(rows)[0]
    assert summary["accuracy"] is None
    assert summary["macro_f1"] is None
    assert summary["sample_n"] == 0
    assert summary["unknown_rate"] == 1


def test_archive2_normalizes_evidence_alias_and_missing_confidence_without_losing_raw_fields():
    response = {"content": {"class_reviews": [{
        "detection_id": "U1", "decision": "uncertain", "proposed_class": "unknown",
        "evidence_sentence": "Visual marking is not readable.",
    }]}}
    normalized = _normalize_experiment_response(response, "U1")
    assert normalized["content"]["class_reviews"][0]["evidence"] == "Visual marking is not readable."
    assert normalized["content"]["class_reviews"][0]["confidence"] == 0.0
    assert normalized["archive2_normalization"]["schema_valid_before_normalization"] is False
    assert normalized["archive2_raw_content"]["class_reviews"][0]["evidence_sentence"] == "Visual marking is not readable."


def test_archive2_normalizes_envelopes_and_percent_confidence_conservatively():
    response = {"content": {"selected_detections": [{
        "detection_id": "R1", "decision": "suggest", "proposed_class": "resistor",
        "confidence": 82, "evidence": "Visible resistor body.",
    }]}}
    normalized = _normalize_experiment_response(response, "R1")
    row = normalized["content"]["class_reviews"][0]
    assert row["proposed_class"] == "resistor"
    assert row["confidence"] == .82
    assert normalized["archive2_normalization"]["schema_valid_before_normalization"] is False
    assert "confidence percentage converted to 0..1" in normalized["archive2_normalization"]["notes"]


def test_archive2_missing_classification_becomes_unknown_not_detector_echo():
    response = {"content": {"selected_detections": [{
        "detection_id": "C4", "detector_class": "capacitor", "nearby_ocr_text_candidates": [],
    }]}}
    normalized = _normalize_experiment_response(response, "C4")
    row = normalized["content"]["class_reviews"][0]
    assert row["proposed_class"] == UNKNOWN_CLASS
    assert row["decision"] == "uncertain"
    assert normalized["archive2_normalization"]["schema_valid_before_normalization"] is False


def test_archive2_preserves_explicit_class_when_model_uses_confirmed_decision_alias():
    response = {"content": {"class_reviews": [{
        "detection_id": "U1", "proposed_class": "ic", "decision": "confirmed",
        "evidence_sentence": "Four-sided leads and visible package marking indicate an IC.",
    }]}}
    normalized = _normalize_experiment_response(response, "U1")
    row = normalized["content"]["class_reviews"][0]
    assert row["proposed_class"] == "ic"
    assert row["decision"] == "suggest"
    assert normalized["archive2_normalization"]["model_uncertain"] is False


def test_archive2_preserves_class_alias_and_explicit_uncertainty_separately():
    response = {"content": {"class_reviews": [{
        "detection_id": "U1", "component_type": "IC", "confidence": .8,
        "decision": "uncertain", "evidence": "Square body with pins on four sides.",
    }]}}
    normalized = _normalize_experiment_response(response, "U1")
    row = normalized["content"]["class_reviews"][0]
    assert row["proposed_class"] == "ic"
    assert row["decision"] == "suggest"
    assert normalized["archive2_normalization"]["model_uncertain"] is True
    assert "component class alias normalized" in normalized["archive2_normalization"]["notes"]


def test_archive2_all_models_are_presented_through_one_classification_field_schema():
    raw_outputs = [
        {"class_reviews": [{"detection_id": "U1", "proposed_class": "ic",
                            "decision": "confirmed", "evidence_sentence": "Four-sided IC package."}]},
        {"class_reviews": [{"detection_id": "U1", "component_type": "IC",
                            "confidence": .95, "evidence": "Square body and metallic pins."}]},
        {"class_reviews": [{"detection_id": "U1", "proposed_class": "ic", "decision": "uncertain",
                            "confidence": 9, "evidence": "Visible IC marking."}]},
    ]
    results = [_normalize_experiment_response({"content": raw}, "U1")["content"]["class_reviews"][0]
               for raw in raw_outputs]
    expected_fields = {"detection_id", "proposed_class", "proposed_subtype", "decision",
                       "confidence", "evidence", "uncertain"}
    assert all(set(row) == expected_fields for row in results)
    assert [row["proposed_class"] for row in results] == ["ic", "ic", "ic"]
    assert [row["uncertain"] for row in results] == [False, False, True]


def test_archive2_proposals_are_normalized_to_the_fixed_18_class_taxonomy():
    assert len(COMPONENT_CLASSES) == 18
    assert set(QWEN_CLASSIFICATION_CLASSES) == {*COMPONENT_CLASSES, UNKNOWN_CLASS}
    allowed = set(QWEN_CLASSIFICATION_CLASSES)
    for class_name in COMPONENT_CLASSES:
        normalized = _normalize_experiment_response({"content": {"class_reviews": [{
            "detection_id": "U1", "proposed_class": class_name, "decision": "suggest",
            "confidence": .8, "evidence": "Visual class proposal.",
        }]}}, "U1")
        assert normalized["content"]["class_reviews"][0]["proposed_class"] in allowed
        assert normalized["content"]["class_reviews"][0]["proposed_class"] == class_name

    for unsupported in ("other_component", "board_feature", "not_component", "fictional_part"):
        normalized = _normalize_experiment_response({"content": {"class_reviews": [{
            "detection_id": "U1", "proposed_class": unsupported, "decision": "suggest",
            "confidence": .8, "evidence": "Unsupported label.",
        }]}}, "U1")
        row = normalized["content"]["class_reviews"][0]
        assert row["proposed_class"] == UNKNOWN_CLASS
        assert row["decision"] == "uncertain"
        assert row["uncertain"] is True


def test_archive2_wraps_qwen_root_array_before_shared_json_parser():
    raw_rows = [{"detection_id": "U1", "component_type": "IC", "confidence": 0.91,
                 "decision": "suggest", "evidence_sentence": "Square package with pins."}]
    ollama_response = {"message": {"content": "```json\n" + json.dumps(raw_rows) + "\n```"}, "done": True}

    adapted, original, notes = _normalize_experiment_chat_http_response(ollama_response)
    assert original == raw_rows
    assert notes == ["root JSON array wrapped as class_reviews"]

    parsed_for_shared_path = json.loads(adapted["message"]["content"])
    normalized = _normalize_experiment_response({"content": parsed_for_shared_path}, "U1")
    row = normalized["content"]["class_reviews"][0]
    assert row["proposed_class"] == "ic"
    assert row["decision"] == "suggest"
    assert row["evidence"] == "Square package with pins."


def test_archive2_does_not_rewrite_object_root_response():
    response = {"message": {"content": '{"class_reviews": []}'}}
    adapted, original, notes = _normalize_experiment_chat_http_response(response)
    assert adapted is response
    assert original == {"class_reviews": []}
    assert notes == []


def test_summary_uses_only_confirmed_18_class_labels_and_reports_slices():
    assert len(COMPONENT_CLASSES) == 18
    rows = [{"status": "COMPLETE", "backend": "Ollama", "model_key": "qwen35_9b",
             "crop_px": 512, "padding_ratio": .25, "output_budget": 128,
             "prediction": "resistor", "prediction_subtype": "smd", "ground_truth": "resistor", "ground_truth_subtype": "smd",
             "small_smd": True, "json_valid": True, "wall_ms": 100.0, "repeat_index": 0},
            {"status": "COMPLETE", "backend": "Ollama", "model_key": "qwen35_9b",
             "crop_px": 512, "padding_ratio": .25, "output_budget": 128,
             "prediction": "capacitor", "ground_truth": "resistor", "json_valid": False,
             "wall_ms": 200.0, "repeat_index": 1},
            {"status": "COMPLETE", "backend": "Ollama", "model_key": "qwen35_9b",
             "crop_px": 512, "padding_ratio": .25, "output_budget": 128,
             "prediction": "capacitor", "ground_truth": "unconfirmed", "json_valid": True,
             "wall_ms": 300.0, "repeat_index": 0},
            {"status": "FAILED", "backend": "Ollama", "model_key": "qwen35_9b",
             "crop_px": 512, "padding_ratio": .25, "output_budget": 128,
             "prediction": None, "ground_truth": "resistor", "json_valid": False,
             "wall_ms": 400.0, "repeat_index": 0}]
    summary = _summarize(rows)[0]
    assert summary["sample_n"] == 2
    assert summary["accuracy"] == .5
    assert summary["subtype_accuracy"] == 1
    assert summary["small_smd_accuracy"] == 1
    assert summary["json_valid_rate"] == 2 / 4
    assert summary["mean_ms"] == 250


def test_classification_crop_size_and_padding_are_applied_to_shared_review_path():
    image = io.BytesIO()
    Image.new("RGB", (1200, 800), "white").save(image, format="PNG")
    crop_images, _size, manifest, *_ = _review_images(
        image.getvalue(), [{"detection_id": "R1", "class_name": "resistor", "bbox_xyxy": [400, 300, 700, 500]}],
        max_crop_count=1, component_crop_padding_ratio=.25,
        component_crop_size_px=384, include_full_image=False,
    )
    with Image.open(io.BytesIO(crop_images[0])) as crop:
        assert max(crop.size) <= 384
    assert len(crop_images) == 1
    assert manifest[0]["kind"] == "detection_crop"
