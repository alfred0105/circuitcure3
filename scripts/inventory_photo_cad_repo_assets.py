#!/usr/bin/env python3
"""Inventory image and editable-CAD candidates across all catalog GitHub repos."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data/component_training/manifests/pcb-photo-cad-source-catalog.json"
DEFAULT_REPOS = ROOT / "data/component_training/raw/external/pcb_photo_cad_pairs/catalog_repositories"
DEFAULT_OUTPUT = ROOT / "data/component_training/processed/pcb_photo_cad_pairs/source-index/full-catalog-repository-assets.json"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CAD_EXT = {".kicad_pcb", ".kicad_sch", ".brd", ".sch", ".pcbdoc", ".schdoc", ".dsn", ".ses", ".gbr", ".ger", ".gtl", ".gbl", ".gts", ".gbs", ".gto", ".gbo", ".gko", ".gm1", ".drl", ".pos", ".ipc", ".xml", ".net", ".netlist", ".csv", ".json", ".epro", ".epcb", ".pro", ".zip"}
DESIGN_EXT = {".kicad_pcb", ".kicad_sch", ".brd", ".sch", ".pcbdoc", ".schdoc", ".dsn", ".ses", ".gbr", ".ger", ".gtl", ".gbl", ".gts", ".gbs", ".gto", ".gbo", ".gko", ".gm1", ".drl", ".epro", ".epcb", ".pro"}
IMAGE_PATH_WORDS = ("photo", "image", "picture", "media", "asset", "gallery", "docs", "output", "product", "board", "pcb")
CAD_PATH_WORDS = ("cad", "pcb", "kicad", "eagle", "hardware", "circuit", "design", "gerber", "manufact", "bom", "pick", "position", "assembly", "schematic", "altium", "hardware", "board")
RENDER_WORDS = ("render", "3d", "2d_view", "screenshot", "schematic", "diagram", "logo", "icon", "oshw_facts", "isometric")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--repositories", type=Path, default=DEFAULT_REPOS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--download-assets", action="store_true", help="Checkout discovered photo and CAD candidate files")
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    repo_index = {}
    catalog_repos = sorted(set(catalog["github_repositories"]) | set(catalog.get("discovered_github_repositories", [])))
    for i, repo_id in enumerate(catalog_repos, start=1):
        repo_path = args.repositories / repo_id.replace("/", "__")
        if not (repo_path / ".git").exists():
            repo_index[repo_id] = {"status": "NOT_CLONED", "path": str(repo_path), "files": []}
            continue
        result = subprocess.run(["git", "-C", str(repo_path), "ls-tree", "-r", "--name-only", "HEAD"], capture_output=True, text=True)
        if result.returncode:
            repo_index[repo_id] = {"status": "TREE_READ_FAILED", "error": result.stderr[-1000:], "files": []}
            continue
        files = result.stdout.splitlines()
        image_candidates, render_candidates, cad_candidates = [], [], []
        for path in files:
            lower = path.lower()
            suffix = Path(path).suffix.lower()
            if suffix in IMAGE_EXT:
                if any(word in lower for word in RENDER_WORDS):
                    render_candidates.append(path)
                else:
                    image_candidates.append(path)
            if suffix in CAD_EXT and (suffix in DESIGN_EXT or any(word in lower for word in CAD_PATH_WORDS)):
                cad_candidates.append(path)
        repo_index[repo_id] = {
            "status": "TREE_INDEXED",
            "path": str(repo_path),
            "file_count": len(files),
            "photo_candidates": image_candidates,
            "render_or_diagram_candidates": render_candidates,
            "cad_or_circuit_candidates": cad_candidates,
        }
        if args.download_assets:
            wanted = sorted(set(image_candidates + cad_candidates))
            if wanted:
                checkout = subprocess.run(["git", "-C", str(repo_path), "checkout", "HEAD", "--", *wanted], capture_output=True, text=True)
                repo_index[repo_id]["checkout_status"] = "DOWNLOADED" if checkout.returncode == 0 else "CHECKOUT_FAILED"
                if checkout.returncode:
                    repo_index[repo_id]["checkout_error"] = checkout.stderr[-1000:]
        if i % 10 == 0:
            print(f"indexed {i}/{len(catalog['github_repositories'])} repositories", flush=True)

    repo_items = list(repo_index.values())
    photo_count = sum(len(x.get("photo_candidates", [])) for x in repo_items)
    render_count = sum(len(x.get("render_or_diagram_candidates", [])) for x in repo_items)
    cad_count = sum(len(x.get("cad_or_circuit_candidates", [])) for x in repo_items)
    candidates = []
    for row in catalog["candidates"]:
        repos = sorted(set(row["github_repositories"]) | set(row.get("discovered_github_repositories", [])))
        candidates.append({
            **row,
            "repository_asset_refs": {
                repo: {
                    "repo_status": repo_index.get(repo, {}).get("status"),
                    "photos": repo_index.get(repo, {}).get("photo_candidates", []),
                    "renders_or_diagrams": repo_index.get(repo, {}).get("render_or_diagram_candidates", []),
                    "cad_and_circuit": repo_index.get(repo, {}).get("cad_or_circuit_candidates", []),
                }
                for repo in repos
            },
        })
    out = {
        "schema_version": "circuitcare.photo-cad-source-asset-index.v1",
        "catalog_source_sha256": catalog["source_sha256"],
        "candidate_count": len(candidates),
        "github_repository_count": len(repo_index),
        "repository_clone_count": sum(x.get("status") == "TREE_INDEXED" for x in repo_items),
        "repository_photo_candidate_count": photo_count,
        "repository_render_diagram_count": render_count,
        "repository_cad_circuit_file_count": cad_count,
        "assets_downloaded": bool(args.download_assets),
        "repositories": repo_index,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("candidate_count", "github_repository_count", "repository_clone_count", "repository_photo_candidate_count", "repository_render_diagram_count", "repository_cad_circuit_file_count", "assets_downloaded")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
