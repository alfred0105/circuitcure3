#!/usr/bin/env python3
"""Package the source-documented Handy PCB photo set with its single KiCad board."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from curate_storm_tracker_cad_pair import registration_candidates, sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/component_training/raw/external/pcb_photo_cad_pairs/handy"
DEFAULT_OUTPUT = ROOT / "data/component_training/processed/pcb_photo_cad_pairs/handy-robotic-hand"
PHOTO_NAMES = ("handy-pcb-front.jpg", "handy-pcb-back.jpg")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--kicad-cli", type=Path, default=Path(shutil.which("kicad-cli") or "/Applications/KiCad.app/Contents/MacOS/kicad-cli"))
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    board = source / "hardware/pcb/kicad/Robotic Hand.kicad_pcb"
    photo_dir = source / "hardware/pcb/photos"
    if not board.is_file() or any(not (photo_dir / n).is_file() for n in PHOTO_NAMES):
        raise FileNotFoundError("Handy board/photos did not match expected source paths")
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("photos", "cad/render"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    for name in PHOTO_NAMES:
        shutil.copy2(photo_dir / name, output / "photos" / name)
    shutil.copy2(board, output / "cad/Robotic Hand.kicad_pcb")
    pcb = output / "cad/Robotic Hand.kicad_pcb"
    for side, layers in (("top", "F.Cu,F.Mask,F.Silkscreen,F.Fab,Edge.Cuts"), ("bottom", "B.Cu,B.Mask,B.Silkscreen,B.Fab,Edge.Cuts")):
        subprocess.run([str(args.kicad_cli), "pcb", "export", "svg", "--mode-single", "--layers", layers, "--fit-page-to-board", "--exclude-drawing-sheet", "--output", str(output / f"cad/render/{side}.svg"), str(pcb)], check=True, capture_output=True, text=True)
    cli_version = subprocess.check_output([str(args.kicad_cli), "version"], text=True).strip()
    try:
        import cairosvg
        for side in ("top", "bottom"):
            cairosvg.svg2png(url=str(output / f"cad/render/{side}.svg"), write_to=str(output / f"cad/render/{side}.png"), output_width=1000)
    except ImportError:
        pass
    pos = output / "cad/robotic-hand-positions.csv"
    subprocess.run([str(args.kicad_cli), "pcb", "export", "pos", "--format", "csv", "--units", "mm", "--side", "both", "--output", str(pos), str(pcb)], check=True, capture_output=True, text=True)
    ipc = output / "cad/robotic-hand-ipc2581.xml"
    subprocess.run([str(args.kicad_cli), "pcb", "export", "ipc2581", "--output", str(ipc), str(pcb)], check=True, capture_output=True, text=True)
    if (output / "cad/render/top.png").is_file():
        diagnostics = registration_candidates(output, PHOTO_NAMES)
    else:
        diagnostics = {"status": "NOT_RUN_RENDER_MISSING", "images": {}}
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    images = []
    for name in PHOTO_NAMES:
        photo = output / "photos" / name
        with __import__("PIL.Image", fromlist=["Image"]).open(photo) as img:
            dimensions = list(img.size)
        diag = diagnostics["images"].get(name, {})
        inliers = max((x.get("ransac_inliers", 0) for x in diag.values() if isinstance(x, dict)), default=0)
        images.append({
            "image_id": f"handy-robotic-hand:{Path(name).stem}",
            "photo_path": str(photo.relative_to(output)),
            "photo_sha256": sha256(photo),
            "width_height_px": dimensions,
            "photo_side": "front" if "front" in name else "back",
            "board_id": "handy-robotic-hand-single-fixed-board",
            "revision_match_status": "SOURCE_DOCUMENTED_FIXED_BOARD_PHOTO_SET",
            "cad_path": "cad/Robotic Hand.kicad_pcb",
            "schematic": "NOT_FOUND_IN_REPOSITORY; board net connectivity retained in IPC-2581 export",
            "positions_path": "cad/robotic-hand-positions.csv",
            "registration_status": "DIAGNOSTIC_ONLY_LOW_SUPPORT" if inliers < 20 else "CANDIDATE_REQUIRES_VALIDATION",
            "registration_diagnostics": diag,
            "photo_bbox_status": "NOT_PROJECTED",
            "training_eligibility": "INTERNAL_USE_CLEARED_BY_USER; IMAGE_LABELS_PENDING_REGISTRATION",
        })
    manifest = {
        "dataset_id": "handy-robotic-hand-photo-cad-candidate-v1",
        "status": "BOARD_MATCHED_PHOTO_CAD_CANDIDATE_NOT_IMAGE_LABEL_READY",
        "source_repository": "https://github.com/lakshveerrao/handy",
        "source_commit": commit,
        "source_evidence": "Repository documents front/back PCB photos and identifies the included Gerber set as the actual fixed board used for the project.",
        "license": "MIT (source repository LICENSE); user confirms internal-use selection",
        "internal_training_use": "USER_CONFIRMED",
        "kicad_cli_version": cli_version,
        "board_sha256": sha256(pcb),
        "ipc2581_sha256": sha256(ipc),
        "positions_sha256": sha256(pos),
        "manual_image_review": False,
        "registration_diagnostics": diagnostics,
        "notes": ["No schematic file was found in the source repository.", "IPC-2581 and pick-and-place coordinates are generated from the KiCad PCB.", "CAD coordinates have not been projected into image-space boxes."],
        "images": images,
    }
    (output / "pair_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "images": len(images), "source_commit": commit, "registration": diagnostics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
