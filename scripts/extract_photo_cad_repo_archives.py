#!/usr/bin/env python3
"""Safely extract PCB/CAD/manufacturing members from indexed repository ZIPs."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "data/component_training/processed/pcb_photo_cad_pairs/source-index/full-catalog-repository-assets.json"
DEFAULT_OUTPUT = ROOT / "data/component_training/raw/external/pcb_photo_cad_pairs/catalog_repository_archive_assets"
MEMBER_SUFFIXES = {".kicad_pcb", ".kicad_sch", ".brd", ".sch", ".pcbdoc", ".schdoc", ".dsn", ".ses", ".gbr", ".ger", ".gtl", ".gbl", ".gts", ".gbs", ".gto", ".gbo", ".gko", ".gm1", ".drl", ".pos", ".ipc", ".net", ".netlist", ".csv", ".xml", ".bom", ".cad", ".pho", ".rou", ".sol", ".plc", ".pls", ".sst", ".cmp", ".crs", ".ly2", ".ly3", ".ly4", ".ly5", ".ly6", ".ly7", ".ly8", ".ly9", ".ly10", ".ly11", ".ly12", ".ly13", ".ly14", ".ly15", ".ly16", ".ly17", ".ly18", ".ly19", ".ly20", ".ly21", ".ly22", ".ly23", ".ly24", ".ly25", ".ly26", ".ly27", ".ly28", ".ly29", ".ly30", ".ly31", ".ly32", ".ly33", ".ly34", ".ly35", ".ly36", ".ly37", ".ly38", ".ly39", ".ly40", ".ly41", ".ly42", ".ly43", ".ly44", ".ly45", ".ly46", ".ly47", ".ly48", ".ly49", ".ly50", ".ly51", ".ly52", ".ly53", ".ly54", ".ly55", ".ly56", ".ly57", ".ly58", ".ly59", ".ly60", ".ly61", ".ly62", ".ly63", ".ly64", ".ly65", ".ly66", ".ly67", ".ly68", ".ly69", ".ly70", ".ly71", ".ly72", ".ly73", ".ly74", ".ly75", ".ly76", ".ly77", ".ly78", ".ly79", ".ly80", ".ly81", ".ly82", ".ly83", ".ly84", ".ly85", ".ly86", ".ly87", ".ly88", ".ly89", ".ly90", ".ly91", ".ly92", ".ly93", ".ly94", ".ly95", ".ly96", ".ly97", ".ly98", ".ly99", ".pdf"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--max-member-mb", type=int, default=100)
    p.add_argument("--max-total-mb", type=int, default=4096)
    a = p.parse_args()
    index = json.loads(a.index.read_text(encoding="utf-8"))
    rows = []
    extracted_bytes = 0
    for repo, info in index["repositories"].items():
        root = Path(info["path"])
        for rel in info.get("cad_or_circuit_candidates", []):
            archive = root / rel
            if archive.suffix.lower() != ".zip" or not archive.is_file() or not zipfile.is_zipfile(archive):
                continue
            row = {"repository": repo, "archive": str(archive.relative_to(ROOT)), "archive_sha256": digest(archive), "members": []}
            try:
                with zipfile.ZipFile(archive) as zf:
                    for member in zf.infolist():
                        if member.is_dir() or member.file_size > a.max_member_mb * 1024 * 1024:
                            continue
                        rel_member = PurePosixPath(member.filename)
                        if rel_member.is_absolute() or ".." in rel_member.parts:
                            continue
                        suffix = Path(rel_member.name).suffix.lower()
                        if suffix not in MEMBER_SUFFIXES:
                            continue
                        if suffix == ".pdf" and not any(w in member.filename.lower() for w in ("schem", "pcb", "gerber", "board", "fabrication")):
                            continue
                        if extracted_bytes + member.file_size > a.max_total_mb * 1024 * 1024:
                            row.setdefault("warnings", []).append("TOTAL_EXTRACTION_LIMIT_REACHED")
                            break
                        dest = a.output / repo.replace("/", "__") / digest(archive) / Path(*rel_member.parts)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        data = zf.read(member)
                        dest.write_bytes(data)
                        extracted_bytes += len(data)
                        row["members"].append({"archive_member": member.filename, "path": str(dest.relative_to(ROOT)), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)
    result = {"schema_version": "circuitcare.repo-archive-cad-extraction.v1", "archive_count": len(rows),
              "archives_with_cad_members": sum(bool(x["members"]) for x in rows),
              "extracted_cad_member_count": sum(len(x["members"]) for x in rows), "extracted_bytes": extracted_bytes,
              "max_member_mb": a.max_member_mb, "max_total_mb": a.max_total_mb, "archives": rows}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "repository_archive_extraction_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("archive_count", "archives_with_cad_members", "extracted_cad_member_count", "extracted_bytes")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
