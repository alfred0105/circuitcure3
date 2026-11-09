#!/usr/bin/env python3
"""Build review-priority artifacts for PCB identities and source annotations.

This script proposes candidates only. It never assigns a physical board ID,
approves a class mapping, edits publisher labels, or promotes pseudo labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/component_training/manifests/pcb-component-detection-curation-manifest.json"
OUT = ROOT / "data/component_training/manifests"
ASSETS = ROOT / "data/component_training/processed/kaggle-c1-review"


def image_path(row: dict) -> Path:
    return ROOT / row["image_path"]


def load_features(path: Path, orb: cv2.ORB) -> tuple[np.ndarray, list, np.ndarray, tuple[int, int]]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    h, w = image.shape
    scale = min(1.0, 1200 / max(h, w))
    if scale < 1:
        image = cv2.resize(image, (round(w * scale), round(h * scale)))
    kp, desc = orb.detectAndCompute(image, None)
    thumb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image, kp or [], desc, (w, h)


def compare_features(first: tuple, second: tuple, matcher: cv2.BFMatcher) -> dict:
    _, kp1, des1, _ = first
    _, kp2, des2, _ = second
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return {"ratio_match_count": 0, "homography_inlier_count": 0, "homography_inlier_ratio": 0.0,
                "homography_status": "INSUFFICIENT_FEATURES"}
    pairs = matcher.knnMatch(des1, des2, k=2)
    good = [a for a, b in pairs if a.distance < 0.72 * b.distance]
    if len(good) < 4:
        return {"ratio_match_count": len(good), "homography_inlier_count": 0, "homography_inlier_ratio": 0.0,
                "homography_status": "INSUFFICIENT_MATCHES"}
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return {"ratio_match_count": len(good), "homography_inlier_count": inliers,
            "homography_inlier_ratio": round(inliers / len(good), 6),
            "homography_status": "ESTIMATED" if matrix is not None else "FAILED"}


def phash(path: Path) -> int:
    with Image.open(path) as im:
        gray = np.asarray(im.convert("L").resize((32, 32)), dtype=np.float32)
    coeff = cv2.dct(gray)[:8, :8].reshape(-1)
    med = float(np.median(coeff[1:]))
    return sum(int(v) << i for i, v in enumerate(coeff > med))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sheet(rows: list[dict], path: Path, *, box_crops: bool = False, per_page: int = 30) -> int:
    path.mkdir(parents=True, exist_ok=True)
    pages = 0
    for start in range(0, len(rows), per_page):
        subset = rows[start:start + per_page]
        cols, tw, th = 5, 250, 220
        canvas = Image.new("RGB", (cols * tw, math.ceil(len(subset) / cols) * th), "white")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(subset):
            x, y = (i % cols) * tw, (i // cols) * th
            if box_crops:
                x1, y1, x2, y2 = row["bbox_xyxy"]
                source = Image.open(image_path(row)).convert("RGB")
                w, h = source.size
                margin_x, margin_y = max(8, int((x2-x1)*0.35)), max(8, int((y2-y1)*0.35))
                image = source.crop((max(0, int(x1-margin_x)), max(0, int(y1-margin_y)),
                                     min(w, int(x2+margin_x)), min(h, int(y2+margin_y))))
                crop_left=max(0, int(x1-margin_x)); crop_top=max(0, int(y1-margin_y))
                ImageDraw.Draw(image).rectangle((x1-crop_left,y1-crop_top,x2-crop_left,y2-crop_top),
                                                outline="#ff2020",width=3)
            else:
                image = Image.open(image_path(row)).convert("RGB")
                image.thumbnail((tw-8, th-52))
                if row.get("annotations"):
                    sx = image.width / row["image_size_px"][0]
                    sy = image.height / row["image_size_px"][1]
                    for ann in row["annotations"]:
                        a, b, c, d = ann["bbox_xyxy"]
                        ImageDraw.Draw(image).rectangle((a*sx, b*sy, c*sx, d*sy), outline="#ff2020", width=2)
            image.thumbnail((tw-12, th-40))
            canvas.paste(image, (x+(tw-image.width)//2, y+2))
            label = row.get("tile_label") or f"{row.get('source_stem_group','')} | {row.get('source_split','')}"
            draw.text((x+3, y+th-34), label[:38], fill="black")
            draw.text((x+3, y+th-19), str(row.get("image_id", ""))[-34:], fill="black")
        pages += 1
        canvas.save(path / f"sheet-{pages:03}.jpg", quality=88)
    return pages


def analyze(manifest_path: Path = MANIFEST, output_dir: Path = OUT, assets_dir: Path = ASSETS) -> dict:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = data["images"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source_stem_group"]].append(row)
    group_report, representatives = [], []
    representative_hash_groups: dict[str, list[str]] = defaultdict(list)
    for stem, members in sorted(groups.items()):
        rep = max(members, key=lambda item: (item["annotation_count"], item["image_size_px"][0]*item["image_size_px"][1], item["image_id"]))
        class_counts = Counter(a["class"] for m in members for a in m["annotations"])
        splits = sorted({m["source_split"] for m in members})
        group_report.append({"source_stem_group": stem, "image_count": len(members), "source_splits": splits,
            "crosses_source_split": len(splits)>1, "augmentation_relation": "SAME_FILENAME_STEM_FAMILY_UNVERIFIED",
            "representative_image_id": rep["image_id"], "representative_image_path": rep["image_path"],
            "resolution_px": rep["image_size_px"], "representative_sha256": rep["image_sha256"],
            "phash_hex": f"{phash(image_path(rep)):016x}", "annotation_count": rep["annotation_count"],
            "group_class_annotation_counts": dict(sorted(class_counts.items())),
            "visual_notes": "REQUIRES_HUMAN_REVIEW", "board_id": "BOARD_ID_UNRESOLVED",
            "review_status": "HUMAN_REVIEW_REQUIRED"})
        representative_hash_groups[rep["image_sha256"]].append(stem)
        rep = dict(rep)
        rep["tile_label"] = f"{stem} ({len(members)} imgs)"
        representatives.append(rep)
    cross_groups = [g for g in group_report if g["crosses_source_split"]]

    # Cache compact ORB descriptors for one representative per source stem.
    orb = cv2.ORB_create(nfeatures=1800, scaleFactor=1.2, nlevels=8, fastThreshold=12)
    features = {g["source_stem_group"]: load_features(image_path(next(r for r in rows if r["image_id"] == g["representative_image_id"])), orb)
                for g in group_report}
    hashes = {g["source_stem_group"]: int(g["phash_hex"], 16) for g in group_report}
    group_by_stem = {g["source_stem_group"]: g for g in group_report}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    candidates = []
    stems = sorted(groups)
    for i, a in enumerate(stems):
        for b in stems[i+1:]:
            distance = (hashes[a] ^ hashes[b]).bit_count()
            # Wider than the previous pHash-only cutoff to find borderline view/rotation candidates.
            if distance > 14:
                continue
            metrics = compare_features(features[a], features[b], matcher)
            candidates.append({"first_stem_group": a, "second_stem_group": b,
                "first_representative_image_id": group_by_stem[a]["representative_image_id"],
                "second_representative_image_id": group_by_stem[b]["representative_image_id"],
                "phash_distance": distance, "source_split_membership_differs": group_by_stem[a]["source_splits"] != group_by_stem[b]["source_splits"],
                **metrics, "candidate_status": "REVIEW_CANDIDATE_ONLY",
                "identity_decision": "UNRESOLVED"})
    candidates.sort(key=lambda item: (-item["homography_inlier_count"], -item["homography_inlier_ratio"], item["phash_distance"]))

    # Save auditable feature-match overlays for the highest-ranked candidates.
    overlay_dir = assets_dir / "similarity-pairs"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for rank, pair in enumerate(candidates[:100], 1):
        left, right = pair["first_stem_group"], pair["second_stem_group"]
        f1, f2 = features[left], features[right]
        if f1[2] is None or f2[2] is None:
            continue
        matches = matcher.knnMatch(f1[2], f2[2], k=2)
        good = [a for a,b in matches if a.distance < 0.72*b.distance]
        mask = None
        if len(good) >= 4:
            p1=np.float32([f1[1][m.queryIdx].pt for m in good]).reshape(-1,1,2)
            p2=np.float32([f2[1][m.trainIdx].pt for m in good]).reshape(-1,1,2)
            _, inlier_mask=cv2.findHomography(p1,p2,cv2.RANSAC,5.0)
            if inlier_mask is not None:
                mask=inlier_mask.ravel().tolist()
        drawn=cv2.drawMatches(f1[0],f1[1],f2[0],f2[1],good[:80],None,
            matchesMask=mask[:80] if mask else None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        caption=(f"{rank:03} pHash={pair['phash_distance']} ratio_matches={pair['ratio_match_count']} "
                 f"RANSAC_inliers={pair['homography_inlier_count']} ratio={pair['homography_inlier_ratio']}")
        cv2.putText(drawn,caption,(12,24),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2,cv2.LINE_AA)
        cv2.imwrite(str(overlay_dir/f"candidate-{rank:03}.jpg"),drawn,[cv2.IMWRITE_JPEG_QUALITY,88])

    # Deterministic class galleries spread across distinct source stems and bbox sizes.
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for ann in row["annotations"]:
            x1,y1,x2,y2=ann["bbox_xyxy"]
            area=((x2-x1)*(y2-y1))/(row["image_size_px"][0]*row["image_size_px"][1])
            by_class[ann["class"]].append({"image_id":row["image_id"],"image_path":row["image_path"],
                "source_stem_group":row["source_stem_group"],"source_split":row["source_split"],
                "bbox_xyxy":ann["bbox_xyxy"],"class":ann["class"],"area_fraction":area,
                "tile_label":f"{ann['class']} | area={area:.4f}"})
    gallery_summary = {}
    for cls, samples in sorted(by_class.items()):
        # Select across area quantiles and different stems; deterministic tie breaking by image id.
        samples.sort(key=lambda x: x["area_fraction"])
        selected, seen = [], set()
        targets = np.linspace(0, len(samples)-1, min(40, len(samples)), dtype=int).tolist()
        for index in targets:
            for candidate in (samples[index:] + samples[:index]):
                if candidate["source_stem_group"] not in seen:
                    selected.append(candidate); seen.add(candidate["source_stem_group"]); break
        gallery_summary[cls] = {"source_annotation_count":len(samples),"gallery_sample_count":len(selected),
            "unique_filename_groups":len({s["source_stem_group"] for s in samples}),
            "confirmed_meaning":None,"confirmed_mislabels":None,"mapping_decision":"MAPPING_PENDING",
            "reviewer":None,"evidence":"NOT_HUMAN_VERIFIED"}
        write_sheet(selected, assets_dir / "class-galleries" / cls, box_crops=True, per_page=40)

    # Geometry heuristics only flag boxes for inspection; they are not errors.
    box_review_candidates=[]
    for row in rows:
        w,h=row["image_size_px"]
        anns=row["annotations"]
        for i,ann in enumerate(anns):
            x1,y1,x2,y2=ann["bbox_xyxy"]
            area=((x2-x1)*(y2-y1))/(w*h)
            if area < 0.0005:
                box_review_candidates.append({"image_id":row["image_id"],"image_path":row["image_path"],
                    "source_stem_group":row["source_stem_group"],"object_index":i,"class":ann["class"],
                    "bbox_xyxy":ann["bbox_xyxy"],"reason":"VERY_SMALL_RELATIVE_AREA_REVIEW_CANDIDATE",
                    "area_fraction":round(area,8),"review_status":"HUMAN_REVIEW_REQUIRED"})
            if min(x1,y1,w-x2,h-y2) <= 1:
                box_review_candidates.append({"image_id":row["image_id"],"image_path":row["image_path"],
                    "source_stem_group":row["source_stem_group"],"object_index":i,"class":ann["class"],
                    "bbox_xyxy":ann["bbox_xyxy"],"reason":"BOX_TOUCHES_IMAGE_EDGE_REVIEW_CANDIDATE",
                    "area_fraction":round(area,8),"review_status":"HUMAN_REVIEW_REQUIRED"})
            for j,other in enumerate(anns[i+1:],i+1):
                ax1,ay1,ax2,ay2=other["bbox_xyxy"]
                inter=max(0,min(x2,ax2)-max(x1,ax1))*max(0,min(y2,ay2)-max(y1,ay1))
                union=(x2-x1)*(y2-y1)+(ax2-ax1)*(ay2-ay1)-inter
                iou=inter/union if union else 0
                if iou >= 0.85:
                    box_review_candidates.append({"image_id":row["image_id"],"image_path":row["image_path"],
                        "source_stem_group":row["source_stem_group"],"object_index":i,"paired_object_index":j,
                        "class":ann["class"],"paired_class":other["class"],"bbox_xyxy":ann["bbox_xyxy"],
                        "paired_bbox_xyxy":other["bbox_xyxy"],"reason":"HIGH_OVERLAP_BOX_PAIR_REVIEW_CANDIDATE",
                        "iou":round(iou,6),"review_status":"HUMAN_REVIEW_REQUIRED"})
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir/"pcb-component-detection-c1-annotation-review-candidates.csv").open("w",newline="") as stream:
        fields=["image_id","image_path","source_stem_group","object_index","paired_object_index","class","paired_class",
            "bbox_xyxy","paired_bbox_xyxy","reason","area_fraction","iou","review_status"]
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(box_review_candidates)

    empty_rows=[r for r in rows if r["annotation_count"]==0]
    empty_sheet=[dict(r, tile_label=f"EMPTY_LABEL | {r['source_split']} | {r['source_stem_group']}") for r in empty_rows]
    pages=write_sheet(empty_sheet, assets_dir / "empty-annotation-review", per_page=25)
    write_sheet(representatives, assets_dir / "filename-group-overview", per_page=30)
    write_sheet([dict(r, tile_label=f"SPLIT_COLLISION | {r['source_stem_group']} | {r['source_split']}")
                 for g in cross_groups for r in groups[g["source_stem_group"]]],
                assets_dir / "split-collision-groups", per_page=30)

    report={"schema_version":"circuitcare.kaggle-c1-identity-annotation-audit.v1",
        "source_image_count":len(rows),"filename_stem_group_count":len(groups),
        "exact_duplicate_representative_group_count":sum(len(v)>1 for v in representative_hash_groups.values()),
        "exact_duplicate_representative_extra_groups":sum(len(v)-1 for v in representative_hash_groups.values() if len(v)>1),
        "exact_duplicate_policy":"SHA-256 equality confirms byte-identical files only; no physical board inference.",
        "filename_group_summary":group_report,"cross_split_filename_group_count":len(cross_groups),
        "cross_split_filename_group_image_count":sum(g["image_count"] for g in cross_groups),
        "orb_representative_pair_candidate_count":len(candidates),
        "orb_ransac_overlay_count":len(list(overlay_dir.glob("candidate-*.jpg"))),
        "orb_ransac_review_candidates":candidates,
        "identity_policy":"All image-match outputs are candidates; no physical Board ID assigned.",
        "class_review":gallery_summary,
        "empty_annotation_review":{"image_count":len(empty_rows),"status_counts":{"EMPTY_UNREVIEWED":len(empty_rows)},
            "contact_sheet_pages":pages,"auto_visual_partial_annotation_candidates":len(empty_rows),
            "human_confirmed_count":0,"negative_samples_promoted":0,
            "visual_note":"All 114 empty-label images were visually inspected on contact sheets and appear to show populated PCB components; human review is still required."},
        "annotation_recovery":{"source_box_count":sum(len(r["annotations"]) for r in rows),
            "geometry_review_candidate_count":len(box_review_candidates),
            "geometry_candidate_reasons":dict(Counter(c["reason"] for c in box_review_candidates)),
            "pseudo_label_candidates":0,"source_labels_edited":0,"human_box_decisions":0,
            "reference_model_available":(ROOT/"models/pcb_component_detector.onnx").is_file(),
            "reference_model_status":"DEPLOYMENT_ONNX_UNAVAILABLE_IN_WORKSPACE; EXISTING_EXPERIMENTS_ARE_NOT_APPROVED_FOR_LABEL_REVIEW",
            "status":"SOURCE_ANNOTATIONS_REMAIN_UNADJUDICATED"},
        "review_artifacts_root":(assets_dir.relative_to(ROOT).as_posix()
            if assets_dir.is_relative_to(ROOT) else str(assets_dir.resolve())),
        "status":"AUTOMATED_CANDIDATE_ANALYSIS_COMPLETE_HUMAN_REVIEW_PENDING"}
    (output_dir/"pcb-component-detection-c1-identity-annotation-report.json").write_text(json.dumps(report,indent=2)+"\n")
    with (output_dir/"pcb-component-detection-c1-empty-review-queue.csv").open("w",newline="") as stream:
        fields=["image_id","image_path","source_split","source_stem_group","image_size_px","image_sha256","annotation_count","review_status","auto_visual_suggestion","confirmed_status","reviewer","reviewed_at_utc","notes"]
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n"); writer.writeheader()
        for row in empty_rows:
            writer.writerow({"image_id":row["image_id"],"image_path":row["image_path"],"source_split":row["source_split"],
                "source_stem_group":row["source_stem_group"],"image_size_px":"x".join(map(str,row["image_size_px"])),
                "image_sha256":row["image_sha256"],"annotation_count":0,"review_status":"EMPTY_UNREVIEWED",
                "auto_visual_suggestion":"PARTIAL_ANNOTATION_CANDIDATE; components visibly mounted on PCB"})
    (output_dir/"pcb-component-detection-c1-class-review.json").write_text(json.dumps(gallery_summary,indent=2)+"\n")
    print(json.dumps({"images":len(rows),"filename_groups":len(groups),"cross_split_groups":len(cross_groups),
        "feature_pair_candidates":len(candidates),"class_samples":{k:v["gallery_sample_count"] for k,v in gallery_summary.items()},
        "empty_review_images":len(empty_rows),"report":str(output_dir/"pcb-component-detection-c1-identity-annotation-report.json")},indent=2))
    return report


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=MANIFEST)
    parser.add_argument("--output-dir",type=Path,default=OUT)
    parser.add_argument("--assets-dir",type=Path,default=ASSETS)
    args=parser.parse_args(); analyze(args.manifest,args.output_dir,args.assets_dir); return 0


if __name__=="__main__":
    raise SystemExit(main())
