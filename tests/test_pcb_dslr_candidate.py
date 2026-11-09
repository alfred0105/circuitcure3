import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image
from circuitcare.vision.detection_evaluation import validate_annotation_manifest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_pcb_dslr_candidate", ROOT / "scripts" / "prepare_pcb_dslr_candidate.py")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def make_source(root: Path, board_count=20):
    for board in range(1, board_count + 1):
        directory = root / f"pcb{board}"
        directory.mkdir(parents=True)
        image = Image.new("RGB", (20, 10))
        image.putdata([((board * 17 + x * 13 + y * 7) % 256,
                        (board * 31 + x * 3 + y * 19) % 256,
                        (board * 47 + x * 23 + y * 5) % 256)
                       for y in range(10) for x in range(20)])
        image.save(directory / "rec1.jpg")
        Image.new("L", (20, 10), 255).save(directory / "rec1-mask.png")
        (directory / "rec1-annot.txt").write_text("10 5 4 4 0 CHIP\n")


def test_pcb_dslr_candidate_exports_ic_and_splits_by_board(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    make_source(source)
    result = converter.convert(source, output)
    assert result["image_count"] == 20
    assert result["board_count"] == 20
    assert result["split_boards"] == {"train": 14, "validation": 3, "test": 3}
    assert result["annotations"] == 20
    assert result["board_split_leakage"] == "PASS"
    assert (output / "data.yaml").is_file()
    assert sum(1 for p in (output / "labels").rglob("*.txt") if p.read_text().strip()) == 20
    manifest = json.loads((output / "candidate_manifest.json").read_text())
    assert manifest["training_ready"] is False
    assert all(row["training_include"] is False for row in manifest["images"])
    checked = validate_annotation_manifest(manifest, class_names=["integrated_circuit"])
    assert checked["status"] == "VALID"
    assert checked["board_split"]["status"] == "VALID"
    assert checked["scoreable_status"] == "BLOCKED_NO_REVIEWED_GROUND_TRUTH"


def test_pcb_dslr_candidate_rejects_bad_box_coordinates(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    make_source(source)
    (source / "pcb1" / "rec1-annot.txt").write_text("40 5 4 4 0 CHIP\n")
    with pytest.raises(ValueError, match="out-of-bounds rotated box"):
        converter.convert(source, output)
