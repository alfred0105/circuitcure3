import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_component_domain", ROOT / "scripts" / "evaluate_component_domain.py")
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


def test_detector_failure_is_not_counted_as_zero_false_positives(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    row_root = tmp_path / "data/component_training/raw"
    row_root.mkdir(parents=True)
    image_path = row_root / "bread.jpg"
    Image.new("RGB", (16, 16), "white").save(image_path)
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"images": [{"image_id": "bread-1", "image_path": "raw/bread.jpg",
        "image_type": "BREADBOARD", "hard_negative_eligible": True, "board_group_id": "board-1"}]}))
    monkeypatch.setattr(evaluate, "detect_components", lambda *a, **k: {
        "status": "MODEL_UNAVAILABLE", "detections": []})

    result = evaluate.evaluate_negatives(Path("model.onnx"), review, confidence=0.25, nms_iou=0.45)
    metrics = result["BREADBOARD"]
    assert metrics["reviewed_image_count"] == 1
    assert metrics["evaluated_image_count"] == 0
    assert metrics["inference_failure_count"] == 1
    assert metrics["image_level_false_positive_rate"] is None
    assert metrics["mean_false_positives_per_image"] is None
    assert metrics["status"] == "NO_SUCCESSFUL_INFERENCES"
    assert metrics["per_image"][0]["status"] == "INFERENCE_FAILED"


def test_only_successful_images_enter_negative_metric_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    row_root = tmp_path / "data/component_training/raw"
    row_root.mkdir(parents=True)
    for name in ("good.jpg", "bad.jpg"):
        Image.new("RGB", (16, 16), "white").save(row_root / name)
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"images": [
        {"image_id": "good", "image_path": "raw/good.jpg", "image_type": "BREADBOARD",
         "hard_negative_eligible": True},
        {"image_id": "bad", "image_path": "raw/bad.jpg", "image_type": "BREADBOARD",
         "hard_negative_eligible": True}]}))

    def detector(image, model, **kwargs):
        if image.getpixel((0, 0)) == (255, 255, 255):
            return {"status": "INFERENCE_COMPLETE", "model": {"status": "READY"}, "detections": []}
        return {"status": "MODEL_UNAVAILABLE", "detections": []}

    # Make one image distinguishable so only its inference can fail.
    Image.new("RGB", (16, 16), "black").save(row_root / "bad.jpg")
    monkeypatch.setattr(evaluate, "detect_components", detector)
    result = evaluate.evaluate_negatives(Path("model.onnx"), review, confidence=0.25, nms_iou=0.45)
    metrics = result["BREADBOARD"]
    assert metrics["reviewed_image_count"] == 2
    assert metrics["evaluated_image_count"] == 1
    assert metrics["inference_failure_count"] == 1
    assert metrics["image_level_false_positive_rate"] == 0
    assert metrics["mean_false_positives_per_image"] == 0
    assert metrics["status"] == "PARTIAL_INFERENCE_FAILURE"


@pytest.mark.parametrize(
    ("reviewed", "evaluated", "failed", "status"),
    [
        (0, 0, 0, "NO_REVIEWED_IMAGES"),
        (2, 2, 0, "MEASURED_REVIEWED_EMPTY_IMAGES"),
        (2, 1, 1, "PARTIAL_INFERENCE_FAILURE"),
        (2, 0, 2, "NO_SUCCESSFUL_INFERENCES"),
    ],
)
def test_main_reports_reviewed_evaluated_failed_negative_counts(
        tmp_path, monkeypatch, capsys, reviewed, evaluated, failed, status):
    """Run the whole CLI entry point; no stale image_count access is allowed."""
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fixture")
    review = tmp_path / "review.json"
    review.write_text('{"images": []}')
    manifest = tmp_path / "evaluation.json"
    manifest.write_text('{"images": []}')
    report_file = tmp_path / "combined.json"

    def fake_eval_run(command, **kwargs):
        assert command[-2] == "--output"
        Path(command[-1]).write_text(json.dumps({
            "evaluation_status": "SOURCE_LABEL_COMPARISON",
            "dataset": {"image_count": 1, "input_fingerprints": {"img": {"board_id": "board-1"}}},
            "metrics": {}, "steady_per_image_latency_ms": 10.0,
            "peak_process_rss_bytes": 1024,
        }))
        return None

    monkeypatch.setattr(evaluate.subprocess, "run", fake_eval_run)
    monkeypatch.setattr(
        evaluate, "evaluate_negatives",
        lambda *_args, **_kwargs: {
            domain: {
                "reviewed_image_count": reviewed if domain == "BREADBOARD" else 0,
                "evaluated_image_count": evaluated if domain == "BREADBOARD" else 0,
                "inference_failure_count": failed if domain == "BREADBOARD" else 0,
                "image_level_false_positive_rate": None,
                "mean_false_positives_per_image": None,
                "total_false_positives": 0,
                "status": status if domain == "BREADBOARD" else "NO_REVIEWED_IMAGES",
                "per_image": [],
            }
            for domain in ("BARE_PCB", "BREADBOARD")
        },
    )
    monkeypatch.setattr(evaluate.sys, "argv", [
        "evaluate_component_domain.py", "--model", str(model),
        "--review-manifest", str(review), "--evaluation-manifest", str(manifest),
        "--output", str(report_file),
    ])

    assert evaluate.main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["negative_counts"]["BREADBOARD"] == {
        "reviewed": reviewed, "evaluated": evaluated, "failed": failed,
    }
    assert summary["negative_counts"]["BARE_PCB"] == {
        "reviewed": 0, "evaluated": 0, "failed": 0,
    }
    report = json.loads(report_file.read_text())
    assert report["reviewed_empty_domain_metrics"]["BREADBOARD"]["status"] == status
