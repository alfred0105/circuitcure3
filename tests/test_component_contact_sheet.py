import importlib.util
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "create_component_contact_sheet", ROOT / "scripts" / "create_component_contact_sheet.py")
sheet_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sheet_module)


def test_contact_sheet_pages_and_review_index(tmp_path):
    image = tmp_path / "board.jpg"
    Image.new("RGB", (80, 60), "green").save(image)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"images": [
        {"image_id": "a", "image_path": str(image), "image_type": "POPULATED_PCB",
         "annotation_count": 1, "review_status": "AUTO_CLASSIFIED"},
        {"image_id": "b", "image_path": str(image), "image_type": "UNKNOWN",
         "annotation_count": 0, "review_status": "DOMAIN_REVIEW_REQUIRED"}]}))
    result = sheet_module.create(manifest, tmp_path / "sheets", page_size=1)
    assert result["page_count"] == 2
    assert (tmp_path / "sheets" / "contact-sheet-001.jpg").is_file()
    assert (tmp_path / "sheets" / "contact-sheet-002.jpg").is_file()
    assert len((tmp_path / "sheets" / "review-index.csv").read_text().splitlines()) == 3
