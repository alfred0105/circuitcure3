#!/usr/bin/env python3
"""Parse the supplied PCB photo/CAD markdown catalog into a stable 132-row index."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/Users/hyunsu/Downloads/pcb_photo_cad_sources_complete_2026-11-07.md")
DEFAULT_OUTPUT = ROOT / "data/component_training/manifests/pcb-photo-cad-source-catalog.json"
URL_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
REPO_RE = re.compile(r"https?://github\.com/([^/\s)\]]+/[^/\s)\]]+)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    text = args.source.read_text(encoding="utf-8")
    source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    active = False
    section = ""
    rows = []
    all_lines = text.splitlines()
    for line_number, line in enumerate(all_lines, start=1):
        if line.startswith("## "):
            active = line.startswith("## 2.") or line.startswith("## 3.")
            section = line[3:].strip()
            continue
        if line.startswith("### ") and active:
            section = line[4:].strip()
            continue
        if not active or not line.startswith("|") or re.match(r"^\|\s*:?-{3}", line):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 3 or not re.search(r"https?://", line):
            continue
        # Table headers are the only non-data rows with one of these first-cell labels.
        if cells[0] in {"제품 / 프로젝트", "모델", "프로젝트"}:
            continue
        links_by_cell = [[{"label": label, "url": url} for label, url in URL_RE.findall(cell)] for cell in cells]
        urls = [link["url"] for links in links_by_cell for link in links]
        repos = sorted({m.group(1).rstrip(".,;") for url in urls for m in REPO_RE.finditer(url)})
        photo_links = links_by_cell[1] if len(links_by_cell) > 1 else []
        cad_links = links_by_cell[2] if len(links_by_cell) > 2 else []
        evidence = re.search(r"`(E[0-3])`", line)
        slug = re.sub(r"[^a-z0-9]+", "-", cells[0].lower()).strip("-") or f"candidate-{len(rows)+1:03d}"
        rows.append({
            "candidate_id": f"CAT-{len(rows)+1:03d}",
            "slug": slug,
            "name": cells[0],
            "section": section,
            "source_markdown_line": line_number,
            "photo_or_product_links": photo_links,
            "cad_or_design_links": cad_links,
            "all_links": urls,
            "github_repositories": repos,
            "evidence_level": evidence.group(1) if evidence else None,
            "notes": cells[-1] if len(cells) >= 5 else "",
            "collection_status": "NOT_COLLECTED",
            "photo_board_match_status": "NOT_VERIFIED",
            "pixel_registration_status": "NOT_RUN",
        })
    if len(rows) != 132:
        raise RuntimeError(f"expected 132 product/project candidates from source catalog, parsed {len(rows)}")
    repos = sorted({repo for row in rows for repo in row["github_repositories"]})
    result = {
        "schema_version": "circuitcare.photo-cad-source-catalog.v1",
        "source_file": str(args.source),
        "source_sha256": source_sha,
        "core_candidate_count": len(rows),
        "github_repository_count": len(repos),
        "github_repositories": repos,
        "candidates": rows,
        "existing_data_policy": "Keep previous dataset directories and manifests unchanged; collect this catalog into a separate source group.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "candidate_count": len(rows), "repository_count": len(repos), "source_sha256": source_sha}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
