#!/usr/bin/env python3
"""Index, without mixing, the supplemental dataset/platform sources in the supplied catalog."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/Users/hyunsu/Downloads/pcb_photo_cad_sources_complete_2026-11-07.md")
DEFAULT_OUTPUT = ROOT / "data/component_training/manifests/pcb-photo-cad-supplementary-sources.json"
LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")


def table_rows(lines: list[str], start: int, end: int) -> list[dict]:
    result = []
    for n in range(start, min(end, len(lines))):
        line = lines[n].strip()
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [x.strip() for x in line.strip("|").split("|")]
        if not cells or cells[0] in {"출처", "출처"}:
            continue
        links = [{"label": label, "url": url} for label, url in LINK.findall(line)]
        if links:
            result.append({"name": re.sub(r"\*\*|`", "", cells[0]), "links": links,
                           "catalog_notes": [re.sub(r"\*\*|`", "", c) for c in cells[2:]],
                           "source_markdown_line": n + 1})
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    a = p.parse_args()
    text = a.source.read_text(encoding="utf-8")
    lines = text.splitlines()
    h = hashlib.sha256(text.encode()).hexdigest()
    datasets = table_rows(lines, 243, 257)
    platforms = table_rows(lines, 258, 274)
    for row in datasets:
        row.update({"source_group": "supplementary_dataset_or_design_source",
                    "catalog_use_policy": "KEEP_SEPARATE_FROM_CORE_PHOTO_CAD_PAIRS",
                    "local_bulk_copy_status": "NOT_COPIED; catalog classifies as auxiliary/excluded/uncertain"})
    for row in platforms:
        row.update({"source_group": "exploration_platform_or_product_reference",
                    "catalog_use_policy": "REFERENCE_ONLY; source describes unbounded platform or discovery lead",
                    "local_bulk_copy_status": "NOT_BULK_COPIED; source is a platform, directory, or lead rather than a bounded dataset"})
    out = {"schema_version": "circuitcare.photo-cad-supplementary-source-index.v1",
           "source_file": str(a.source), "source_sha256": h,
           "scope_note": "Section 4 contains auxiliary datasets/design sources; Section 5 contains exploration platforms and leads. The supplied catalog explicitly keeps these separate from its 132 core candidates.",
           "supplementary_dataset_source_count": len(datasets), "exploration_platform_or_lead_count": len(platforms),
           "datasets_and_design_sources": datasets, "platforms_and_leads": platforms}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"supplementary_dataset_source_count": len(datasets), "exploration_platform_or_lead_count": len(platforms)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
