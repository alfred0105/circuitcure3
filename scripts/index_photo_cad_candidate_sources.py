#!/usr/bin/env python3
"""Index locally fetched photo/CAD sources without promoting uncertain pairs to labels."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/component_training/raw/external/pcb_photo_cad_pairs"
OUT = ROOT / "data/component_training/processed/pcb_photo_cad_pairs/source-index"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def commit(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def list_rel(repo: Path, patterns: tuple[str, ...]) -> list[str]:
    items = set()
    for pattern in patterns:
        items.update(p for p in repo.glob(pattern) if p.is_file())
    return sorted(p.relative_to(repo).as_posix() for p in items)


SOURCES = {
    "storm-tracker-hardware": {
        "repo": "storm-tracker-hardware", "source_url": "https://github.com/solderable/storm-tracker-hardware",
        "photo_patterns": ("docs/images/*.jpg",),
        "cad_patterns": ("hardware/*.kicad_pcb", "hardware/*.kicad_sch", "bom/*.csv", "manufacturing/rev-a/assembly/*.csv"),
        "match_rule": "All 3 source photos are explicitly documented as fabricated Rev A hardware and linked to this same Rev A design.",
        "pair_status": "SOURCE_DECLARED_REV_MATCH",
        "board_candidates": ["hardware/storm_tracker.kicad_pcb"],
        "circuit_candidates": ["hardware/storm_tracker.kicad_sch"],
    },
    "handy": {
        "repo": "handy", "source_url": "https://github.com/lakshveerrao/handy",
        "photo_patterns": ("hardware/pcb/photos/*",),
        "cad_patterns": ("hardware/pcb/kicad/*.kicad_pcb", "hardware/pcb/kicad/*.kicad_sch", "hardware/pcb/gerbers/*"),
        "match_rule": "README labels the two files as PCB front/back and states the included Gerbers are the actual fixed project board.",
        "pair_status": "SOURCE_DOCUMENTED_SINGLE_BOARD_MATCH",
        "board_candidates": ["hardware/pcb/kicad/Robotic Hand.kicad_pcb"],
        "circuit_candidates": [],
    },
    "FEMTO": {
        "repo": "FEMTO", "source_url": "https://github.com/mfolejewski/FEMTO",
        "photo_patterns": ("Photos/*",),
        "cad_patterns": ("source_files/Kicad/**/*.kicad_pcb", "source_files/Kicad/**/*.kicad_sch", "source_files/Altium/**/*.PcbDoc", "source_files/Altium/**/*.SchDoc", "manufacturing_files/**/*.csv"),
        "match_rule": "Filename-based Module/Tester/Frame family candidate; revision-level photo-to-CAD match not asserted.",
        "pair_status": "CANDIDATE_FAMILY_MATCH_REVISION_UNVERIFIED",
        "board_candidates": ["source_files/Kicad/Module/Femto_v1.kicad_pcb", "source_files/Kicad/Tester/Femto_tester_v1.kicad_pcb", "source_files/Kicad/Frame/Femto_frame.kicad_pcb"],
        "circuit_candidates": ["source_files/Kicad/Module/Femto_v1.kicad_sch", "source_files/Kicad/Tester/Femto_tester_v1.kicad_sch"],
    },
    "PicoBerry": {
        "repo": "PicoBerry", "source_url": "https://github.com/mfolejewski/PicoBerry",
        "photo_patterns": ("photos/*",),
        "cad_patterns": ("source files/KiCad/*.kicad_pcb", "source files/KiCad/*.kicad_sch", "source files/KiCad/*.SchDoc", "manufacturing files/*.csv"),
        "match_rule": "Project photos linked to the V1.0 design family by repository; per-photo side and revision not confirmed.",
        "pair_status": "CANDIDATE_PROJECT_MATCH_REVISION_UNVERIFIED",
        "board_candidates": ["source files/KiCad/PicoBerry_V1.0.kicad_pcb"],
        "circuit_candidates": ["source files/KiCad/PicoBerry_V1.0.kicad_sch"],
    },
    "pixie-device": {
        "repo": "pixie-device", "source_url": "https://github.com/firefly/pixie-device",
        "photo_patterns": ("panel/output/board-front.jpg", "panel/output/board-back.jpg"),
        "cad_patterns": ("*.kicad_pcb", "*.kicad_sch", "output/rev-*/production/*.csv"),
        "match_rule": "Board-front/back photo filenames exclude render-* images, but several revisions exist and the photo revision is not pinned.",
        "pair_status": "CANDIDATE_PHOTO_CAD_REVISION_AMBIGUOUS",
        "board_candidates": ["output/rev-2", "output/rev-4", "output/rev-5", "output/rev-6", "pixie-device.kicad_pcb"],
        "circuit_candidates": ["pixie-device.kicad_sch", "output.kicad_sch", "power.kicad_sch", "mcu.kicad_sch"],
    },
    "omi-consumer": {
        "repo": "omi", "source_url": "https://github.com/BasedHardware/omi",
        "photo_patterns": ("omi/hardware/consumer/assembly/photos/*",),
        "cad_patterns": ("omi/hardware/consumer/electrical/**/*.zip", "omi/hardware/consumer/bom/*.csv"),
        "match_rule": "Assembly photos belong to the Consumer product; the design has separate mainboard, charger, and FPC boards, so per-image board mapping remains ambiguous.",
        "pair_status": "CANDIDATE_MULTI_BOARD_PRODUCT_MATCH",
        "board_candidates": ["omi/hardware/consumer/electrical/mainboard", "omi/hardware/consumer/electrical/charger-board", "omi/hardware/consumer/electrical/fpc-board"],
        "circuit_candidates": ["Altium CAD archives referenced by the Consumer docs; archive contents not extracted in this index"],
    },
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    projects = []
    image_count = 0
    for project_id, spec in SOURCES.items():
        repo = RAW / spec["repo"]
        if not (repo / ".git").exists():
            projects.append({"project_id": project_id, "status": "SOURCE_NOT_CLONED"})
            continue
        images = []
        for relative in list_rel(repo, spec["photo_patterns"]):
            path = repo / relative
            try:
                with Image.open(path) as image:
                    dims = list(image.size)
            except Exception:
                dims = None
            is_render = any(token in relative.lower() for token in ("render", "3d_view", "2d_view", "exploded", "oshw_facts"))
            images.append({
                "source_path": f"data/component_training/raw/external/pcb_photo_cad_pairs/{spec['repo']}/{relative}",
                "source_relative_path": relative,
                "sha256": sha(path), "width_height_px": dims,
                "auto_image_role": "RENDER_OR_DIAGRAM_EXCLUDE_FROM_CAMERA_TRAINING" if is_render else "PHOTO_CANDIDATE_REQUIRES_AUTOMATED_REGISTRATION",
                "board_candidates": spec["board_candidates"],
                "circuit_candidates": spec["circuit_candidates"],
                "pair_status": spec["pair_status"],
            })
        image_count += len(images)
        projects.append({
            "project_id": project_id,
            "source_url": spec["source_url"],
            "source_commit": commit(repo),
            "source_license_clearance": "USER_CONFIRMED_INTERNAL_USE",
            "pair_status": spec["pair_status"],
            "matching_rule": spec["match_rule"],
            "candidate_board_and_circuit_files": spec["board_candidates"] + spec["circuit_candidates"],
            "available_cad_files": list_rel(repo, spec["cad_patterns"]),
            "photos": images,
            "manual_image_review": False,
            "image_space_labels": "NOT_CREATED_UNTIL_AUTOMATED_REGISTRATION_IS_RELIABLE",
        })
    result = {
        "schema_version": "circuitcare.photo-cad-candidate-index.v1",
        "created_local_date": "2026-11-07",
        "status": "SOURCE_INDEXED; ONLY_STORM_AND_HANDY_HAVE_SOURCE_DOCUMENTED_SINGLE_BOARD_MATCHES",
        "internal_training_use": "USER_CONFIRMED_FOR_SELECTED_SOURCES",
        "photo_count": image_count,
        "projects": projects,
        "notes": [
            "This index links source photos with source CAD candidates; it does not assert pixel-level component labels.",
            "Same-board/revision declarations are only marked where the repository documents them.",
            "Render/diagram-like files are flagged from filenames and paths, without visual review.",
            "Keep every image from one board/revision in a single split group.",
        ],
    }
    target = OUT / "photo_cad_candidate_index.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "project_count": len(projects), "photo_count": image_count, "pair_statuses": {x["project_id"]: x.get("pair_status", x.get("status")) for x in projects}}, indent=2))


if __name__ == "__main__":
    main()
