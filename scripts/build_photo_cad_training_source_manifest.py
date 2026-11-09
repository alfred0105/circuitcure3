#!/usr/bin/env python3
"""Join all collected catalog assets back to their 132 source candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/component_training/raw/external/pcb_photo_cad_pairs"
DEFAULT_CATALOG = ROOT / "data/component_training/manifests/pcb-photo-cad-source-catalog.json"
DEFAULT_REPO_INDEX = ROOT / "data/component_training/processed/pcb_photo_cad_pairs/source-index/full-catalog-repository-assets.json"
DEFAULT_PAGE_IMAGES = RAW / "catalog_product_page_images/page_image_manifest.json"
DEFAULT_DIRECT_CAD = RAW / "catalog_direct_cad_assets/direct_cad_manifest.json"
DEFAULT_REPO_ARCHIVES = RAW / "catalog_repository_archive_assets/repository_archive_extraction_manifest.json"
DEFAULT_OUTPUT = ROOT / "data/component_training/manifests/pcb-photo-cad-training-source-manifest.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_meta(path: Path) -> dict:
    def git(*args: str) -> str:
        p = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else ""
    return {"commit": git("rev-parse", "HEAD"), "remote": git("remote", "get-url", "origin")}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--repo-index", type=Path, default=DEFAULT_REPO_INDEX)
    p.add_argument("--page-images", type=Path, default=DEFAULT_PAGE_IMAGES)
    p.add_argument("--direct-cad", type=Path, default=DEFAULT_DIRECT_CAD)
    p.add_argument("--repo-archives", type=Path, default=DEFAULT_REPO_ARCHIVES)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    a = p.parse_args()
    catalog = json.loads(a.catalog.read_text(encoding="utf-8"))
    repo_index = json.loads(a.repo_index.read_text(encoding="utf-8"))
    pages = json.loads(a.page_images.read_text(encoding="utf-8"))
    direct = json.loads(a.direct_cad.read_text(encoding="utf-8"))
    repo_archives = json.loads(a.repo_archives.read_text(encoding="utf-8"))
    repo_downloads_path = a.repo_index.parent / "asset-download-results.json"
    repo_downloads = json.loads(repo_downloads_path.read_text(encoding="utf-8")) if repo_downloads_path.is_file() else {"asset_file_count": 0}
    extracted_by_repo: dict[str, list[dict]] = {}
    for archive in repo_archives.get("archives", []):
        extracted_by_repo.setdefault(archive["repository"], []).extend(archive.get("members", []))

    page_by_id: dict[str, list[dict]] = {}
    for page in pages["pages"]:
        for cid in page.get("candidate_ids", []):
            page_by_id.setdefault(cid, []).append(page)
    direct_by_id: dict[str, list[dict]] = {}
    for item in direct["items"]:
        for cid in item.get("candidate_ids", []):
            direct_by_id.setdefault(cid, []).append(item)

    repo_meta = {}
    for repo, item in repo_index["repositories"].items():
        repo_path = Path(item.get("path", ""))
        repo_meta[repo] = git_meta(repo_path) if item.get("status") == "TREE_INDEXED" else {"commit": "", "remote": ""}

    rows = []
    total_git_photos = total_git_cad = total_page_images = total_direct_cad = 0
    for source in catalog["candidates"]:
        rid = source["candidate_id"]
        photo_assets, render_assets, cad_assets, archive_cad_assets, page_images, direct_assets, page_statuses = [], [], [], [], [], [], []
        associated_repos = sorted(set(source.get("github_repositories", [])) | set(source.get("discovered_github_repositories", [])))
        for repo in associated_repos:
            idx = repo_index["repositories"].get(repo, {})
            repo_path = Path(idx.get("path", ""))
            meta = repo_meta.get(repo, {})
            for role, key, target in (("REPOSITORY_PHOTO_CANDIDATE_UNCLASSIFIED", "photo_candidates", photo_assets),
                                      ("REPOSITORY_RENDER_OR_DIAGRAM_NOT_PHOTO", "render_or_diagram_candidates", render_assets),
                                      ("REPOSITORY_CAD_CIRCUIT_CANDIDATE", "cad_or_circuit_candidates", cad_assets)):
                for rel in idx.get(key, []):
                    path = repo_path / rel
                    record = {"role": role, "repository": repo, "commit": meta.get("commit"),
                              "source_url": f"https://github.com/{repo}/blob/{meta.get('commit') or 'HEAD'}/{rel}",
                              "repository_relative_path": rel, "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                              "exists": path.is_file()}
                    if path.is_file():
                        record.update({"sha256": sha(path), "size_bytes": path.stat().st_size})
                    target.append(record)
            for member in extracted_by_repo.get(repo, []):
                archive_cad_assets.append({**member, "repository": repo, "role": "REPOSITORY_ARCHIVE_CAD_OR_MANUFACTURING_FILE"})
        for page in page_by_id.get(rid, []):
            page_statuses.append({"page_url": page.get("page_url"), "status": page.get("status"),
                                  "http_status": page.get("http_status"), "saved_image_count": page.get("saved_image_count", 0)})
            for image in page.get("images", []):
                path = RAW / "catalog_product_page_images" / image["path"]
                page_images.append({**image, "role": image.get("role", "PRODUCT_PAGE_IMAGE_CANDIDATE_NOT_VISUALLY_CLASSIFIED"),
                                    "path": str(path.relative_to(ROOT)), "exists": path.is_file()})
        for item in direct_by_id.get(rid, []):
            record = {k: item.get(k) for k in ("url", "source_page", "status", "final_url", "path", "sha256", "size_bytes", "content_type", "extracted_cad_count")}
            if item.get("path"):
                record["path"] = str((RAW / "catalog_direct_cad_assets" / item["path"]).relative_to(ROOT))
                record["exists"] = (RAW / "catalog_direct_cad_assets" / item["path"]).is_file()
            record["extracted_cad"] = []
            for ext in item.get("extracted_cad", []):
                ext_path = RAW / "catalog_direct_cad_assets" / ext["path"]
                record["extracted_cad"].append({**ext, "path": str(ext_path.relative_to(ROOT)), "exists": ext_path.is_file()})
            direct_assets.append(record)
        total_git_photos += sum(x["exists"] for x in photo_assets)
        total_git_cad += sum(x["exists"] for x in cad_assets)
        total_page_images += sum(x["exists"] for x in page_images)
        total_direct_cad += sum(x.get("exists", False) for x in direct_assets)
        has_any = bool(any(x["exists"] for x in photo_assets + render_assets + cad_assets) or archive_cad_assets or any(x["exists"] for x in page_images + direct_assets))
        curation_by_candidate = {
            "CAT-081": {"source_documented_identity": "REV_A_PHOTOS_DOCUMENTED_BY_SOURCE_AUTHOR", "curation_manifest": "data/component_training/processed/pcb_photo_cad_pairs/storm-tracker-rev-a/pair_manifest.json", "registration": "ATTEMPTED_NO_ACCEPTED_TRANSFORM"},
            "CAT-082": {"source_documented_identity": "SAME_PRODUCT_PHOTO_AND_KICAD_BOARD; REVISION_UNVERIFIED", "curation_manifest": "data/component_training/processed/pcb_photo_cad_pairs/handy-robotic-hand/pair_manifest.json", "registration": "ATTEMPTED_NO_ACCEPTED_TRANSFORM"},
            "CAT-094": {"source_documented_identity": "PRODUCT_FAMILY_WITH_THREE_BOARDS; PER_IMAGE_BOARD_UNRESOLVED", "curation_manifest": "data/component_training/processed/pcb_photo_cad_pairs/omi-consumer/pair_manifest.json", "registration": "NOT_RUN_MULTI_BOARD_ASSOCIATION_UNRESOLVED"},
        }
        curated = curation_by_candidate.get(rid, {})
        rows.append({**source,
                     "collection_status": "ASSETS_COLLECTED_CANDIDATES_UNVERIFIED" if has_any else "NO_LOCAL_ASSETS_FOUND",
                     "repository_sources": [{"repository": r, **repo_meta.get(r, {})} for r in associated_repos],
                     "assets": {"repository_photo_candidates": photo_assets, "repository_cad_circuit_candidates": cad_assets,
                                "repository_render_or_diagram_not_photo": render_assets,
                                "repository_archive_extracted_cad_manufacturing": archive_cad_assets,
                                "product_page_image_candidates": page_images, "product_page_statuses": page_statuses,
                                "direct_cad_link_assets": direct_assets},
                     "pair_validation": {"image_is_camera_photo": "NOT_VISUALLY_REVIEWED",
                                         "photo_board_revision_match": curated.get("source_documented_identity", source.get("photo_board_match_status", "NOT_VERIFIED")),
                                         "curation_manifest": curated.get("curation_manifest"),
                                         "pixel_registration": curated.get("registration", source.get("pixel_registration_status", "NOT_RUN")),
                                         "training_ready": False}})

    def counts(row: dict) -> tuple[bool, bool]:
        assets = row["assets"]
        has_photo = any(x["exists"] for x in assets["repository_photo_candidates"] + assets["product_page_image_candidates"])
        has_cad = any(x["exists"] for x in assets["repository_cad_circuit_candidates"] + assets["direct_cad_link_assets"]) or bool(assets["repository_archive_extracted_cad_manufacturing"])
        return has_photo, has_cad
    coverage = [counts(row) for row in rows]
    summary = {"catalog_candidate_count": len(rows), "candidate_with_any_local_asset_count": sum(x["collection_status"] != "NO_LOCAL_ASSETS_FOUND" for x in rows),
               "candidate_without_local_asset_count": sum(x["collection_status"] == "NO_LOCAL_ASSETS_FOUND" for x in rows),
               "candidate_with_photo_source_count": sum(p for p, _ in coverage),
               "candidate_with_cad_or_source_document_count": sum(c for _, c in coverage),
               "candidate_with_both_photo_and_cad_source_count": sum(p and c for p, c in coverage),
               "repository_photo_candidate_files_indexed": repo_index.get("repository_photo_candidate_count", 0),
               "repository_render_or_diagram_files_indexed": repo_index.get("repository_render_diagram_count", 0),
               "repository_cad_circuit_files_indexed": repo_index.get("repository_cad_circuit_file_count", 0),
               "repository_source_files_downloaded": repo_downloads.get("asset_file_count", 0),
               "product_page_images_downloaded": pages.get("downloaded_image_count", 0),
               "product_page_image_asset_references_present": total_page_images,
               "direct_link_asset_references_present": total_direct_cad,
               "direct_downloaded_url_count": direct.get("downloaded_count", 0),
               "direct_extracted_cad_file_count": direct.get("extracted_cad_file_count", 0),
               "repository_archive_count": repo_archives.get("archive_count", 0),
               "repository_archive_extracted_cad_member_count": repo_archives.get("extracted_cad_member_count", 0),
               "photo_cad_pairs_training_ready": 0,
               "notes": ["The catalog defines candidates, not 132 verified distinct boards.",
                         "Downloaded images remain unclassified candidates; no manual visual review was performed.",
                         "A product/repository name match does not establish board revision or pixel registration.",
                         "Only user-approved candidate sources are included; prior datasets remain separate and untouched."]}
    output = {"schema_version": "circuitcare.photo-cad-training-source-manifest.v1",
              "created_date": "2026-11-07", "source_catalog": str(a.catalog.relative_to(ROOT)),
              "source_catalog_sha256": catalog["source_sha256"], "collection_scope": "all 132 core product/project candidates",
              "prior_data_modified_or_mixed": False, "summary": summary, "candidates": rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
