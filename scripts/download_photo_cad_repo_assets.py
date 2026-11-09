#!/usr/bin/env python3
"""Download indexed camera-photo and CAD candidates from the catalog repos concurrently."""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "data/component_training/processed/pcb_photo_cad_pairs/source-index/full-catalog-repository-assets.json"


def checkout(repo: str, item: dict, root: Path, include_renders: bool = False) -> dict:
    path = Path(item["path"])
    wanted = sorted(set(item.get("photo_candidates", []) + item.get("cad_or_circuit_candidates", [])))
    if include_renders:
        wanted = sorted(set(wanted + item.get("render_or_diagram_candidates", [])))
    if not wanted:
        return {"repo": repo, "status": "NO_ASSET_FILES", "file_count": 0}
    p = subprocess.run(["git", "-C", str(path), "checkout", "HEAD", "--", *wanted], capture_output=True, text=True, timeout=600)
    if p.returncode:
        return {"repo": repo, "status": "CHECKOUT_FAILED", "file_count": len(wanted), "error": p.stderr[-1200:]}
    present = sum((path / name).is_file() for name in wanted)
    return {"repo": repo, "status": "ASSETS_DOWNLOADED", "file_count": present, "render_files_downloaded": include_renders}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--include-renders", action="store_true", help="Also preserve render/diagram candidates in a separate manifest role")
    args = parser.parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    tasks = [(repo, item) for repo, item in index["repositories"].items() if item.get("status") == "TREE_INDEXED"]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(checkout, repo, item, ROOT, args.include_renders): repo for repo, item in tasks}
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception as e:
                result = {"repo": futures[future], "status": "CHECKOUT_ERROR", "error": str(e)}
            results.append(result)
            print(f"[{i}/{len(tasks)}] {result['repo']} {result['status']} ({result.get('file_count', 0)} files)", flush=True)
    out = args.index.parent / "asset-download-results.json"
    out.write_text(json.dumps({"repository_count": len(tasks), "asset_file_count": sum(x.get("file_count", 0) for x in results), "results": sorted(results, key=lambda x: x["repo"])}, indent=2) + "\n", encoding="utf-8")
    failures = [x for x in results if x["status"] in {"CHECKOUT_FAILED", "CHECKOUT_ERROR"}]
    print(json.dumps({"repositories": len(tasks), "asset_file_count": sum(x.get("file_count", 0) for x in results), "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
