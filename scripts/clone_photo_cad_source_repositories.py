#!/usr/bin/env python3
"""Shallow, blob-filtered clone of every GitHub repo in the supplied 132-item catalog."""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data/component_training/manifests/pcb-photo-cad-source-catalog.json"
DEFAULT_DEST = ROOT / "data/component_training/raw/external/pcb_photo_cad_pairs/catalog_repositories"
ALIASES = {
    "solderable/storm-tracker-hardware": "storm-tracker-hardware",
    "lakshveerrao/handy": "handy",
    "mfolejewski/FEMTO": "FEMTO",
    "mfolejewski/PicoBerry": "PicoBerry",
    "firefly/pixie-device": "pixie-device",
    "BasedHardware/omi": "omi",
}


def clone_one(repo: str, dest: Path, raw_root: Path) -> dict[str, str]:
    target = dest / repo.replace("/", "__")
    if repo in ALIASES:
        alias = raw_root / Path(ALIASES[repo])
        if target.is_symlink() or target.exists():
            return {"repo": repo, "status": "ALIAS_ALREADY_PRESENT"}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(alias.resolve(), target_is_directory=True)
        return {"repo": repo, "status": "ALIASED_EXISTING_CLONE"}
    if (target / ".git").exists():
        return {"repo": repo, "status": "ALREADY_CLONED"}
    if target.exists() and not target.is_symlink():
        return {"repo": repo, "status": "TARGET_EXISTS_NOT_GIT"}
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", url, str(target)],
        capture_output=True, text=True, timeout=240,
    )
    if result.returncode == 0:
        return {"repo": repo, "status": "CLONED", "path": str(target)}
    return {"repo": repo, "status": "CLONE_FAILED", "error": (result.stderr or result.stdout)[-1000:]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    data = json.loads(args.catalog.read_text(encoding="utf-8"))
    repos = sorted(set(data["github_repositories"]) | set(data.get("discovered_github_repositories", [])))
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    raw_root = dest.parent
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(clone_one, repo, dest, raw_root): repo for repo in repos}
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                item = future.result()
            except Exception as exc:
                item = {"repo": futures[future], "status": "CLONE_ERROR", "error": str(exc)}
            results.append(item)
            print(f"[{i}/{len(repos)}] {item['repo']} {item['status']}", flush=True)
    output = dest.parents[3] / "processed/pcb_photo_cad_pairs/source-index/repository-clone-results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"catalog_repositories": len(repos), "results": sorted(results, key=lambda x: x["repo"])}, indent=2) + "\n", encoding="utf-8")
    failures = [x for x in results if x["status"] in {"CLONE_FAILED", "CLONE_ERROR", "TARGET_EXISTS_NOT_GIT"}]
    print(json.dumps({"repositories": len(repos), "cloned_or_reused": len(results)-len(failures), "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
