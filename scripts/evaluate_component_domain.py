#!/usr/bin/env python3
"""Report PCB detection metrics separately from reviewed empty-board/breadboard false positives."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from PIL import Image
from circuitcare.vision.component_detector import detect_components


def evaluate_negatives(model: Path, review_manifest: Path, *, confidence: float, nms_iou: float) -> dict:
    review = json.loads(review_manifest.read_text())
    grouped = defaultdict(list)
    for row in review["images"]:
        if not row.get("hard_negative_eligible"):
            continue
        image_path = ROOT / "data" / "component_training" / row["image_path"]
        started = time.perf_counter()
        record = {"image_id": row["image_id"], "board_group_id": row.get("board_group_id"),
                  "image_path": str(image_path)}
        try:
            if not image_path.is_file():
                raise FileNotFoundError(f"image not found: {image_path}")
            with Image.open(image_path) as source_image:
                result = detect_components(source_image.convert("RGB"), str(model),
                    confidence_threshold=confidence, iou_threshold=nms_iou)
            if not isinstance(result, dict) or result.get("status") != "INFERENCE_COMPLETE":
                raise RuntimeError(f"detector did not complete: {result.get('status') if isinstance(result, dict) else type(result).__name__}")
            model_info = result.get("model")
            if not isinstance(model_info, dict) or model_info.get("status") != "READY":
                raise RuntimeError("detector did not report a ready model")
            detections = result.get("detections")
            if not isinstance(detections, list):
                raise RuntimeError("detector returned no valid detections list")
            record.update({"status": "INFERENCE_COMPLETE", "false_positive_count": len(detections),
                           "has_false_positive": bool(detections)})
        except Exception as exc:  # Failed inference must never be scored as zero false positives.
            record.update({"status": "INFERENCE_FAILED", "failure_type": type(exc).__name__,
                           "failure_reason": str(exc)})
        record["inference_latency_ms"] = (time.perf_counter() - started) * 1000
        grouped[row["image_type"]].append(record)
    metrics = {}
    for domain in ("BARE_PCB", "BREADBOARD"):
        rows = grouped[domain]
        successful = [row for row in rows if row["status"] == "INFERENCE_COMPLETE"]
        failures = [row for row in rows if row["status"] != "INFERENCE_COMPLETE"]
        n = len(successful)
        metrics[domain] = {"reviewed_image_count": len(rows), "evaluated_image_count": n,
            "inference_failure_count": len(failures),
            "image_level_false_positive_rate": sum(row["has_false_positive"] for row in successful) / n if n else None,
            "mean_false_positives_per_image": sum(row["false_positive_count"] for row in successful) / n if n else None,
            "total_false_positives": sum(row["false_positive_count"] for row in successful),
            "status": ("NO_REVIEWED_IMAGES" if not rows else
                       "NO_SUCCESSFUL_INFERENCES" if not n else
                       "PARTIAL_INFERENCE_FAILURE" if failures else "MEASURED_REVIEWED_EMPTY_IMAGES"),
            "per_image": rows}
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    args = parser.parse_args()
    if not args.model.is_file(): parser.error(f"model does not exist: {args.model}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    positive_output = args.output.with_name(args.output.stem + "-positive-pcb.json")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate_pcb_detector.py"),
        "--dataset-manifest", str(args.evaluation_manifest), "--model", str(args.model),
        "--confidence", str(args.confidence), "--nms-iou", str(args.nms_iou),
        "--output", str(positive_output)], check=True, capture_output=True, text=True)
    positive = json.loads(positive_output.read_text())
    negatives = evaluate_negatives(args.model, args.review_manifest,
                                   confidence=args.confidence, nms_iou=args.nms_iou)
    report = {"status": "DEVELOPMENT_ONLY_BOARD_GROUP_VALIDATION",
        "confidence_threshold": args.confidence, "nms_iou_threshold": args.nms_iou,
        "positive_pcb": {"evaluation_status": positive["evaluation_status"],
            "image_count": positive["dataset"]["image_count"],
            "manual_board_group_count": len({row.get("board_id") for row in
                positive["dataset"]["input_fingerprints"].values() if row.get("board_id")}),
            "metrics": positive["metrics"],
            "latency": positive["steady_per_image_latency_ms"],
            "peak_process_rss_bytes": positive["peak_process_rss_bytes"]},
        "reviewed_empty_domain_metrics": negatives,
        "limits": ["Source annotations are not independently adjudicated.",
            "Board groups are manual visual groups, not source-verified physical board IDs.",
            "No sealed independent test split is available."]}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"output": str(args.output.resolve()), "positive_image_count": positive["dataset"]["image_count"],
                      "negative_counts": {k: {"reviewed": v["reviewed_image_count"],
                                          "evaluated": v["evaluated_image_count"],
                                          "failed": v["inference_failure_count"]}
                                          for k, v in negatives.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
