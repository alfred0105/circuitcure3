#!/usr/bin/env python3
"""Extract the Omi Consumer multi-board KiCad sources and export CAD connectivity."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/component_training/raw/external/pcb_photo_cad_pairs/omi"
DEFAULT_OUTPUT = ROOT / "data/component_training/processed/pcb_photo_cad_pairs/omi-consumer"
PHOTO_DIR = Path("omi/hardware/consumer/assembly/photos")
PHOTO_NAMES = (
    "components-disassembled.jpg",
    "materials-labelled.jpg",
    "materials-labelled-exploded-view.jpg",
    "outer-aluminium-covers.jpg",
)
ARCHIVES = {
    "mainboard": "omi/hardware/consumer/electrical/mainboard/altium/omi2-mainboard-v1.2-altium.zip",
    "charger": "omi/hardware/consumer/electrical/charger-board/altium/omi2-charger-v1.0-altium.zip",
    "fpc": "omi/hardware/consumer/electrical/fpc-board/altium/omi2-fpc-v1.0-altium.zip",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_kicad(archive: Path, target: Path) -> list[Path]:
    extracted = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            p = Path(member.filename)
            if "__MACOSX" in p.parts or member.is_dir():
                continue
            if p.suffix.lower() not in {".kicad_pcb", ".kicad_sch", ".kicad_pro", ".kicad_prl", ".kicad_sym"}:
                continue
            rel = Path(*p.parts[1:]) if len(p.parts) > 1 else p
            if not rel.name:
                continue
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(member))
            extracted.append(dest)
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--kicad-cli", type=Path, default=Path(shutil.which("kicad-cli") or "/Applications/KiCad.app/Contents/MacOS/kicad-cli"))
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    cad_root = output / "cad"
    cad_root.mkdir(parents=True, exist_ok=True)
    boards = {}
    for board_id, archive_rel in ARCHIVES.items():
        archive = source / archive_rel
        target = cad_root / board_id
        extracted = extract_kicad(archive, target)
        pcbs = [p for p in extracted if p.suffix == ".kicad_pcb"]
        schs = [p for p in extracted if p.suffix == ".kicad_sch"]
        if len(pcbs) != 1 or not schs:
            raise RuntimeError(f"expected one PCB and at least one schematic in {archive}: pcbs={pcbs}, schematics={schs}")
        pcb = pcbs[0]
        # The top-level schematic shares the board/project name when present.
        top_sch = next((p for p in schs if p.stem.lower() in {pcb.stem.lower(), "omi"}), schs[0])
        ipc = target / f"{board_id}.ipc2581.xml"
        pos = target / f"{board_id}-positions.csv"
        subprocess.run([str(args.kicad_cli), "pcb", "export", "ipc2581", "--output", str(ipc), str(pcb)], check=True, capture_output=True, text=True)
        subprocess.run([str(args.kicad_cli), "pcb", "export", "pos", "--format", "csv", "--units", "mm", "--side", "both", "--output", str(pos), str(pcb)], check=True, capture_output=True, text=True)
        netlist = target / f"{board_id}.net.xml"
        netlist_result = subprocess.run([str(args.kicad_cli), "sch", "export", "netlist", "--format", "kicadxml", "--output", str(netlist), str(top_sch)], capture_output=True, text=True)
        netlist_status = "EXPORTED" if netlist_result.returncode == 0 else "FAILED"
        board = {
            "board_id": board_id,
            "source_archive": archive_rel,
            "source_archive_sha256": digest(archive),
            "pcb": str(pcb.relative_to(output)),
            "pcb_sha256": digest(pcb),
            "schematic_top": str(top_sch.relative_to(output)),
            "schematic_sha256": digest(top_sch),
            "all_schematic_sheets": [str(p.relative_to(output)) for p in schs],
            "ipc2581": str(ipc.relative_to(output)),
            "ipc2581_sha256": digest(ipc),
            "positions": str(pos.relative_to(output)),
            "positions_sha256": digest(pos),
            "schematic_netlist_status": netlist_status,
            "schematic_netlist": str(netlist.relative_to(output)) if netlist_status == "EXPORTED" else None,
            "schematic_netlist_sha256": digest(netlist) if netlist_status == "EXPORTED" else None,
            "netlist_error_tail": netlist_result.stderr[-1000:] if netlist_status != "EXPORTED" else None,
        }
        boards[board_id] = board

    photo_root = source / PHOTO_DIR
    photo_dir = output / "photos"
    photo_dir.mkdir(exist_ok=True)
    images = []
    for name in PHOTO_NAMES:
        src = photo_root / name
        dst = photo_dir / name
        shutil.copy2(src, dst)
        with Image.open(dst) as im:
            dimensions = list(im.size)
        images.append({
            "image_id": f"omi-consumer:{Path(name).stem}",
            "photo_path": str(dst.relative_to(output)),
            "source_relative_path": str(src.relative_to(source)),
            "photo_sha256": digest(dst),
            "width_height_px": dimensions,
            "product_group": "omi-consumer",
            "candidate_boards": list(boards),
            "board_identity_status": "PRODUCT_GROUP_MATCH_MULTIPLE_BOARD_TARGETS",
            "per_board_photo_registration": "NOT_PERFORMED",
            "photo_role": "ASSEMBLY_PHOTO_CANDIDATE; no image review",
            "internal_training_use": "USER_CONFIRMED",
        })
    try:
        license_text = (source / "LICENSE").read_text(encoding="utf-8", errors="replace")
    except OSError:
        license_text = ""
    license_summary = "MIT" if "MIT License" in license_text else "USER_CONFIRMED_INTERNAL_USE; source license not parsed"
    repo_commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "dataset_id": "omi-consumer-photo-multiboard-cad-candidate-v1",
        "status": "PRODUCT_AND_CAD_LINKED_PER_IMAGE_BOARD_MAPPING_PENDING",
        "source_repository": "https://github.com/BasedHardware/omi",
        "source_commit": repo_commit,
        "source_license": license_summary,
        "internal_training_use": "USER_CONFIRMED",
        "kicad_cli_version": subprocess.check_output([str(args.kicad_cli), "version"], text=True).strip(),
        "boards": boards,
        "images": images,
        "manual_image_review": False,
        "image_space_component_boxes": "NOT_GENERATED; photos may show multiple boards, enclosures, or other assembly stages",
        "split_group": "omi-consumer-product-family",
        "notes": [
            "The same product contains three different electrical boards; do not assign each assembly image to one PCB without automatic evidence.",
            "KiCad design files were extracted from the repository's Altium-named ZIP archives.",
            "Each board has a KiCad PCB, schematic source, IPC-2581 export, and pick-and-place coordinates.",
        ],
    }
    (output / "pair_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "boards": list(boards), "images": len(images), "netlists": {k: v["schematic_netlist_status"] for k, v in boards.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
