import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_external_component_candidate",
    ROOT / "scripts" / "prepare_external_component_candidate.py")
candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate)


def make_source(root: Path, *, objects=None, include_annotation=True):
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(json.dumps({"classes": [{"title": "Resistor"}]}))
    for split in ("train", "validation", "test"):
        (root / split / "img").mkdir(parents=True)
        (root / split / "ann").mkdir()
    image = root / "train" / "img" / "board.jpg"
    Image.new("RGB", (20, 10), "white").save(image)
    if include_annotation:
        data = {"size": {"height": 10, "width": 20}, "objects": objects if objects is not None else [{
            "classTitle": "Resistor", "geometryType": "rectangle",
            "points": {"exterior": [[2, 1], [10, 9]]}}]}
        (root / "train" / "ann" / "board.jpg.json").write_text(json.dumps(data))


def test_candidate_converter_preserves_provenance_and_marks_blockers(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    make_source(source)
    result = candidate.convert(source, output)
    manifest = json.loads((output / "candidate_manifest.json").read_text())
    label = next((output / "labels" / "train").glob("*.txt")).read_text().strip()
    assert result["images"] == 1 and result["annotations"] == 1
    assert label.startswith("0 ")
    assert manifest["training_ready"] is False
    assert manifest["images"][0]["board_review_status"] == "BOARD_ID_UNRESOLVED"
    assert (output / "images" / "train" / "board.jpg").is_symlink()


def test_candidate_converter_rejects_invalid_or_duplicate_boxes(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    obj = {"classTitle": "Resistor", "geometryType": "rectangle",
           "points": {"exterior": [[2, 1], [10, 9]]}}
    make_source(source, objects=[obj, obj])
    with pytest.raises(ValueError, match="duplicate box"):
        candidate.convert(source, output)


def test_candidate_converter_rejects_missing_annotation(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    make_source(source, include_annotation=False)
    with pytest.raises(ValueError, match="missing annotation"):
        candidate.convert(source, output)


def test_empty_annotations_are_recorded_but_not_exported(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    make_source(source, objects=[])
    result = candidate.convert(source, output)
    manifest = json.loads((output / "candidate_manifest.json").read_text())
    assert result["empty_unreviewed"] == 1
    assert manifest["images"][0]["annotation_status"] == "EMPTY_UNREVIEWED"
    assert manifest["images"][0]["included_in_loader_smoke_export"] is False
    assert not any(path.is_file() for path in (output / "images").rglob("*"))
