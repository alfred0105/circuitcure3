#!/usr/bin/env python3
"""Create paginated contact sheets and a review index from a component manifest."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def create(manifest_path: Path, output_dir: Path, *, page_size: int = 100) -> dict:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    manifest = json.loads(manifest_path.read_text())
    rows = manifest.get("images")
    if not isinstance(rows, list):
        raise ValueError("manifest must have an images array")
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    tile_w, tile_h, thumb_w, thumb_h = 260, 220, 250, 165
    columns = 5
    index_rows = []
    for start in range(0, len(rows), page_size):
        page = start // page_size + 1
        chunk = rows[start:start + page_size]
        sheet = Image.new("RGB", (columns * tile_w, math.ceil(len(chunk) / columns) * tile_h), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(chunk):
            image_path = Path(row.get("image_path") or row.get("image", ""))
            if not image_path.is_absolute():
                candidate = ROOT / image_path
                if not candidate.is_file():
                    candidate = ROOT / "data/component_training/raw/external" / image_path
                image_path = candidate
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                source_width, source_height = image.size
                image.thumbnail((thumb_w, thumb_h))
            x, y = (offset % columns) * tile_w, (offset // columns) * tile_h
            image_x, image_y = x + (thumb_w - image.width) // 2, y
            sheet.paste(image, (image_x, image_y))
            scale_x, scale_y = image.width / source_width, image.height / source_height
            for annotation in row.get("annotations", []):
                bbox = annotation.get("bbox_xyxy")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                draw.rectangle((image_x + x1 * scale_x, image_y + y1 * scale_y,
                                image_x + x2 * scale_x, image_y + y2 * scale_y),
                               outline="#ff2020", width=1)
            identifier = str(row.get("image_id", image_path.name))
            domain = str(row.get("image_type", "UNKNOWN"))
            count = row.get("annotation_count", "?")
            review = str(row.get("review_status", "UNREVIEWED"))
            split = str(row.get("source_split", "UNKNOWN"))
            stem = str(row.get("source_stem_group", "UNKNOWN"))
            confidence = str(row.get("classification_confidence", "UNKNOWN"))
            draw.text((x + 4, y + thumb_h + 2), identifier[:32], fill="black", font=font)
            draw.text((x + 4, y + thumb_h + 18), f"{domain} | boxes={count} | {split}", fill="black", font=font)
            draw.text((x + 4, y + thumb_h + 34), f"{confidence} | {review}"[:45], fill="black", font=font)
            draw.text((x + 4, y + thumb_h + 50), f"stem={stem}"[:45], fill="black", font=font)
            index_rows.append({"page": page, "tile": offset + 1, "image_id": identifier,
                "image_path": str(image_path), "image_type": domain,
                "annotation_count": count, "source_split": split, "source_stem_group": stem,
                "classification_confidence": confidence, "review_status": review})
        sheet.save(output_dir / f"contact-sheet-{page:03}.jpg", quality=88)
    with (output_dir / "review-index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]) if index_rows else
            ["page", "tile", "image_id", "image_path", "image_type", "annotation_count",
             "source_split", "source_stem_group", "classification_confidence", "review_status"])
        writer.writeheader()
        writer.writerows(index_rows)
    return {"image_count": len(rows), "page_count": math.ceil(len(rows) / page_size),
            "output_dir": str(output_dir.resolve()), "index": str((output_dir / "review-index.csv").resolve())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(create(args.manifest, args.output_dir, page_size=args.page_size), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
