#!/usr/bin/env python3
"""Create a provenance-first local photo/CAD candidate package for Storm Tracker Rev A."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/component_training/raw/external/pcb_photo_cad_pairs/storm-tracker-hardware"
DEFAULT_OUTPUT = ROOT / "data/component_training/processed/pcb_photo_cad_pairs/storm-tracker-rev-a"
PHOTO_NAMES = (
    "storm-tracker-board-back.jpg",
    "storm-tracker-board-in-hand.jpg",
    "storm-tracker-hero.jpg",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_cad_exports(output: Path, kicad_cli: Path) -> dict[str, object]:
    pcb = output / "cad/storm_tracker.kicad_pcb"
    schematic = output / "cad/storm_tracker.kicad_sch"
    render_dir = output / "cad/render"
    render_dir.mkdir(parents=True, exist_ok=True)
    for side, layers in (
        ("top", "F.Cu,F.Mask,F.Silkscreen,F.Fab,Edge.Cuts"),
        ("bottom", "B.Cu,B.Mask,B.Silkscreen,B.Fab,Edge.Cuts"),
    ):
        svg = render_dir / f"{side}.svg"
        subprocess.run([
            str(kicad_cli), "pcb", "export", "svg", "--mode-single", "--layers", layers,
            "--fit-page-to-board", "--exclude-drawing-sheet", "--output", str(svg), str(pcb),
        ], check=True, capture_output=True, text=True)
    netlist = output / "cad/storm_tracker.net.xml"
    subprocess.run([
        str(kicad_cli), "sch", "export", "netlist", "--format", "kicadxml",
        "--output", str(netlist), str(schematic),
    ], check=True, capture_output=True, text=True)
    result: dict[str, object] = {
        "kicad_cli_version": subprocess.check_output([str(kicad_cli), "version"], text=True).strip(),
        "schematic_netlist": str(netlist.relative_to(output)),
        "schematic_netlist_sha256": sha256(netlist),
        "renders": {},
    }
    try:
        import cairosvg
        for side in ("top", "bottom"):
            svg = render_dir / f"{side}.svg"
            png = render_dir / f"{side}.png"
            cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=1000)
            result["renders"][side] = {
                "svg": str(svg.relative_to(output)), "svg_sha256": sha256(svg),
                "png": str(png.relative_to(output)), "png_sha256": sha256(png),
            }
    except ImportError:
        result["renders"] = {"status": "SVG_CREATED; PNG_RENDERER_UNAVAILABLE"}
    return result


def registration_candidates(output: Path, photo_names: tuple[str, ...]) -> dict[str, object]:
    """Compute diagnostic ORB/RANSAC candidates; never promotes weak fits to labels."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {"status": "NOT_RUN_DEPENDENCY_UNAVAILABLE", "images": {}}
    orb = cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8, edgeThreshold=15, fastThreshold=8)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    rows: dict[str, object] = {}
    for photo_name in photo_names:
        photo_path = output / "photos" / photo_name
        image = cv2.imread(str(photo_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            rows[photo_name] = {"status": "IMAGE_READ_FAILED"}
            continue
        factor = 1000 / max(image.shape)
        image = cv2.resize(image, (round(image.shape[1] * factor), round(image.shape[0] * factor)))
        photo_kp, photo_desc = orb.detectAndCompute(image, None)
        side_results = {}
        for side in ("top", "bottom"):
            template = cv2.imread(str(output / f"cad/render/{side}.png"), cv2.IMREAD_GRAYSCALE)
            if template is None:
                side_results[side] = {"status": "TEMPLATE_NOT_RENDERED"}
                continue
            template_kp, template_desc = orb.detectAndCompute(template, None)
            if template_desc is None or photo_desc is None:
                side_results[side] = {"status": "NO_DESCRIPTORS"}
                continue
            pairs = matcher.knnMatch(template_desc, photo_desc, k=2)
            good = [a for a, b in pairs if a.distance < 0.72 * b.distance]
            inliers = 0
            median_error = None
            if len(good) >= 4:
                src_points = cv2.KeyPoint_convert([template_kp[m.queryIdx] for m in good])
                dst_points = cv2.KeyPoint_convert([photo_kp[m.trainIdx] for m in good])
                homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
                if homography is not None and mask is not None:
                    inlier_mask = mask.ravel().astype(bool)
                    inliers = int(inlier_mask.sum())
                    projected = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
                    errors = np.linalg.norm(projected - dst_points, axis=1)
                    median_error = float(np.median(errors[inlier_mask])) if inliers else None
            side_results[side] = {
                "good_feature_matches": len(good), "ransac_inliers": inliers,
                "median_reprojection_error_px": median_error,
                "status": "DIAGNOSTIC_ONLY_LOW_SUPPORT" if inliers < 20 else "CANDIDATE_REQUIRES_VALIDATION",
            }
        rows[photo_name] = side_results
    return {"status": "DIAGNOSTIC_ONLY_NO_MANUAL_REVIEW", "images": rows}


def collect(source: Path, output: Path, kicad_cli: Path | None = None) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    if not (source / ".git").exists():
        raise FileNotFoundError(f"expected a cloned source repository at {source}")
    if output == source or source in output.parents:
        raise ValueError("output must not be inside the source repository")

    image_source = source / "docs/images"
    board_source = source / "hardware/storm_tracker.kicad_pcb"
    schematic_source = source / "hardware/storm_tracker.kicad_sch"
    bom_source = source / "bom/storm-tracker-full.csv"
    placement_source = source / "manufacturing/rev-a/assembly/storm_tracker-all-pos.csv"
    required = [board_source, schematic_source, bom_source, placement_source]
    required += [image_source / name for name in PHOTO_NAMES]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("required source files missing: " + ", ".join(missing))

    output.mkdir(parents=True, exist_ok=True)
    for folder in ("photos", "cad", "source_data"):
        (output / folder).mkdir(exist_ok=True)

    copied: dict[str, Path] = {}
    for filename in PHOTO_NAMES:
        dst = output / "photos" / filename
        shutil.copy2(image_source / filename, dst)
        copied[f"photo:{filename}"] = dst
    for key, source_path, name in (
        ("pcb", board_source, board_source.name),
        ("schematic", schematic_source, schematic_source.name),
        ("bom", bom_source, bom_source.name),
        ("placement", placement_source, placement_source.name),
    ):
        dst = output / ("cad" if key in {"pcb", "schematic"} else "source_data") / name
        shutil.copy2(source_path, dst)
        copied[key] = dst

    if kicad_cli is None:
        candidate = shutil.which("kicad-cli") or "/Applications/KiCad.app/Contents/MacOS/kicad-cli"
        kicad_cli = Path(candidate)
    cad_exports = None
    if kicad_cli.is_file():
        cad_exports = build_cad_exports(output, kicad_cli)
    registration = registration_candidates(output, PHOTO_NAMES) if cad_exports and (output / "cad/render/top.png").exists() else {
        "status": "NOT_RUN_RENDER_MISSING", "images": {}
    }

    bom_rows = read_csv(copied["bom"])
    bom_by_ref: dict[str, dict[str, str]] = {}
    for item in bom_rows:
        refs = [ref.strip() for ref in item.get("References", "").split(",")]
        for ref in refs:
            if ref and ref != "PCB1":
                bom_by_ref[ref] = item

    positions = read_csv(copied["placement"])
    components: list[dict[str, object]] = []
    for row in positions:
        ref = row.get("Ref", "").strip()
        if not ref or ref.startswith("REF"):
            continue
        item = bom_by_ref.get(ref, {})
        populate = item.get("Populate", "UNKNOWN") or "UNKNOWN"
        components.append({
            "board_id": "storm-tracker-rev-a",
            "reference": ref,
            "value": row.get("Val", ""),
            "footprint": row.get("Package", ""),
            "bom_category": item.get("Category", "UNMAPPED"),
            "bom_population": populate,
            "x_mm": row.get("PosX", ""),
            "y_mm": row.get("PosY", ""),
            "rotation_deg": row.get("Rot", ""),
            "side": row.get("Side", "").lower(),
            "photo_bbox_xyxy": "",
            "photo_projection_status": "NOT_REGISTERED",
        })
    component_fields = ["board_id", "reference", "value", "footprint", "bom_category", "bom_population", "x_mm", "y_mm", "rotation_deg", "side", "photo_bbox_xyxy", "photo_projection_status"]
    write_csv(output / "components.csv", components, component_fields)

    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    license_files = [p.name for p in source.iterdir() if p.is_file() and p.name.lower().startswith("license")]
    images = []
    for name in PHOTO_NAMES:
        path = output / "photos" / name
        source_path = image_source / name
        image_registration = registration["images"].get(name, {})
        best_inliers = max((v.get("ransac_inliers", 0) for v in image_registration.values() if isinstance(v, dict)), default=0)
        images.append({
            "image_id": f"storm-tracker-rev-a:{Path(name).stem}",
            "photo_path": str(path.relative_to(output)),
            "source_photo_path": str(source_path.relative_to(source)),
            "photo_sha256": sha256(path),
            "width_height_px": None,
            "photo_side": "back" if "board-back" in name else "unknown",
            "photo_type": "repository_documented_prototype_photo; per-image automated type unresolved",
            "assembly_state": "repository_documented_populated_rev_a_candidate; per-image automated state unresolved",
            "board_id": "storm-tracker-rev-a",
            "revision_match_status": "SOURCE_DECLARED_REV_A_MATCH",
            "cad_path": "cad/storm_tracker.kicad_pcb",
            "schematic_path": "cad/storm_tracker.kicad_sch",
            "component_labels_path": "components.csv",
            "registration_status": "DIAGNOSTIC_ONLY_LOW_SUPPORT" if best_inliers < 20 else "CANDIDATE_REQUIRES_VALIDATION",
            "registration_diagnostics": image_registration,
            "training_eligibility": "INTERNAL_USE_CLEARED_BY_USER; BLOCKED_PHOTO_PROJECTION",
        })
        try:
            from PIL import Image
            with Image.open(path) as image:
                images[-1]["width_height_px"] = list(image.size)
        except ImportError:
            pass

    files = {}
    for label, path in copied.items():
        files[label] = {"path": str(path.relative_to(output)), "sha256": sha256(path), "size_bytes": path.stat().st_size}
    provenance = {
        "dataset_id": "storm-tracker-rev-a-photo-cad-candidate-v1",
        "status": "CANDIDATE_PAIR_PACKAGE_NOT_TRAINING_READY",
        "source_repository": "https://github.com/solderable/storm-tracker-hardware",
        "source_commit": commit,
        "source_evidence": "README states the repository contains the editable Rev A design for the revision shown in photographs and calls the images fabricated Rev A hardware.",
        "photo_cad_identity": "SOURCE_DECLARED_REV_A_MATCH; geometric photo-to-CAD registration not performed",
        "license": {
            "github_api_license_spdx": None,
            "license_files_found": license_files,
            "status": "USER_CONFIRMED_SOURCE_SET_IS_CLEARED_FOR_INTERNAL_USE",
            "training_allowed_for_internal_use": True,
            "redistribution_allowed": False,
        },
        "assets": files,
        "board_cad": "KiCad PCB and schematic source included",
        "electrical_reference": "BOM and Rev A pick-and-place coordinates included; component list joins placements to BOM reference designators",
        "photo_component_boxes": "NOT_GENERATED; automated feature registration had insufficient support for a reliable image-space projection",
        "cad_exports": cad_exports,
        "registration_diagnostics": registration,
        "split_group": "storm-tracker-rev-a",
        "manual_image_review": False,
        "notes": [
            "Keep all photos from this board revision in the same train/validation/test group.",
            "The user confirmed the selected source set is cleared for internal use; do not infer external redistribution rights.",
            "CAD positions are board-coordinate labels, not image-coordinate boxes until registration is computed and checked.",
            "BOM population and design intent do not prove a component is visibly present in each photo.",
        ],
    }
    (output / "pair_manifest.json").write_text(json.dumps({**provenance, "images": images}, indent=2) + "\n", encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "image_count": len(images), "placement_component_count": len(components), "license_files_found": license_files, "internal_use_cleared_by_user": True, "photo_projection_complete": False, "registration_diagnostics": registration, "source_commit": commit}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(collect(args.source, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
