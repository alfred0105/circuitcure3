import json
import importlib.util
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_kaggle_component_c1", ROOT / "scripts" / "analyze_kaggle_component_c1.py")
c1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c1)


def _row(tmp_path, name, stem, split, box=True):
    path = tmp_path / f"{name}.jpg"
    Image.new("RGB", (80, 60), (20, 120, 60)).save(path)
    annotations = ([{"class": "Resistor", "bbox_xyxy": [20, 15, 40, 35]}] if box else [])
    return {"image_id": name, "image_path": str(path), "image_sha256": name,
            "image_size_px": [80, 60], "source_stem_group": stem,
            "source_split": split, "annotation_count": len(annotations),
            "annotations": annotations}


def test_c1_audit_keeps_identity_and_annotation_approval_pending(tmp_path):
    rows = [_row(tmp_path, "a", "frame-a", "train"),
            _row(tmp_path, "b", "frame-b", "test", box=False)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"images": rows}))
    output, assets = tmp_path / "reports", tmp_path / "sheets"

    report = c1.analyze(manifest, output, assets)

    assert report["source_image_count"] == 2
    assert report["filename_stem_group_count"] == 2
    assert report["empty_annotation_review"]["human_confirmed_count"] == 0
    assert report["annotation_recovery"]["source_labels_edited"] == 0
    assert all(group["board_id"] == "BOARD_ID_UNRESOLVED"
               for group in report["filename_group_summary"])
    queue = (output / "pcb-component-detection-c1-empty-review-queue.csv").read_text()
    assert "EMPTY_UNREVIEWED" in queue
    assert "PARTIAL_ANNOTATION_CANDIDATE" in queue
    assert (assets / "empty-annotation-review" / "sheet-001.jpg").is_file()


def test_phash_identical_images_have_equal_hash(tmp_path):
    import numpy as np

    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[8:24, 10:22] = [30, 180, 90]
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    Image.fromarray(pixels).save(first)
    Image.fromarray(pixels.copy()).save(second)
    assert c1.phash(first) == c1.phash(second)
