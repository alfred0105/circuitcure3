"""Ground-truth-aware labels for detector predictions.

Only reviewed annotations can establish correctness. Unreviewed or partial
labels leave unmatched predictions unresolved instead of calling them false
positives.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

from circuitcare.vision.detection_evaluation import box_iou_xyxy


def adjudicate_predictions(predictions_by_image: dict[str, list[dict[str, Any]]],
                           annotation_manifest: dict[str, Any], *,
                           iou_threshold: float = 0.5) -> dict[str, Any]:
    """Assign a correctness label to each prediction where reviewed truth exists.

    Results are class-aware and one-to-one. Prediction boxes use absolute image
    pixels, as do annotation ``bbox_xyxy`` values.
    """
    if not 0 < float(iou_threshold) <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    if not isinstance(predictions_by_image, dict):
        raise ValueError("predictions_by_image must be keyed by image ID")
    if not isinstance(annotation_manifest, dict) or not isinstance(annotation_manifest.get("images"), list):
        raise ValueError("annotation manifest must contain an images list")

    truth_by_id = {str(row.get("image_id")): row for row in annotation_manifest["images"]
                   if isinstance(row, dict) and row.get("image_id") is not None}
    output_images = []
    all_labels: Counter[str] = Counter()
    complete_image_count = 0
    complete_tp = complete_fp = complete_fn = 0

    for image_id in sorted(set(truth_by_id) | set(map(str, predictions_by_image))):
        truth_image = truth_by_id.get(image_id)
        predictions = list(predictions_by_image.get(image_id, []))
        if truth_image is None:
            rows = [{**prediction, "adjudication_label": "UNRESOLVED_NO_GROUND_TRUTH_IMAGE",
                     "matched_component_id": None, "matched_iou": None}
                    for prediction in predictions]
            all_labels.update(row["adjudication_label"] for row in rows)
            output_images.append({"image_id": image_id, "status": "UNRESOLVED_NO_GROUND_TRUTH_IMAGE",
                                 "predictions": rows, "missed_components": []})
            continue

        annotations = [row for row in truth_image.get("annotations", []) if isinstance(row, dict)]
        reviewed = [row for row in annotations if row.get("annotation_status") == "REVIEWED_GROUND_TRUTH"]
        complete = _is_complete_review(truth_image, annotations, reviewed)
        if complete:
            complete_image_count += 1

        candidates = []
        for pred_index, prediction in enumerate(predictions):
            pred_class = _prediction_class(prediction)
            pred_box = prediction.get("bbox_xyxy")
            if not _valid_box(pred_box):
                candidates.append((pred_index, None, [], pred_class))
                continue
            overlaps = [(box_iou_xyxy(pred_box, gt.get("bbox_xyxy", [])), gt_index, gt)
                        for gt_index, gt in enumerate(reviewed)]
            overlaps = sorted((x for x in overlaps if x[0] >= iou_threshold),
                              key=lambda row: (-row[0], str(row[2].get("component_id", ""))))
            candidates.append((pred_index, prediction, overlaps, pred_class))

        used_truth: set[int] = set()
        labeled_predictions: list[dict[str, Any] | None] = [None] * len(predictions)
        # Strongest predictions claim each reviewed object once; duplicate boxes
        # cannot inflate the number of correct detector results.
        order = sorted(range(len(predictions)), key=lambda i: (-_confidence(predictions[i]), i))
        for pred_index in order:
            _, prediction, overlaps, pred_class = candidates[pred_index]
            if prediction is None:
                label = "UNRESOLVED_INVALID_PREDICTION_BOX"
                result = {**predictions[pred_index], "adjudication_label": label,
                          "matched_component_id": None, "matched_iou": None}
            else:
                same_class = [(iou, gt_index, gt) for iou, gt_index, gt in overlaps
                              if gt.get("class") == pred_class]
                available = [entry for entry in same_class if entry[1] not in used_truth]
                if available:
                    iou, gt_index, gt = available[0]
                    used_truth.add(gt_index)
                    label = "CORRECT"
                    result = {**prediction, "adjudication_label": label,
                              "matched_component_id": gt.get("component_id"), "matched_iou": iou}
                    if complete:
                        complete_tp += 1
                elif same_class:
                    iou, _, gt = same_class[0]
                    label = "DUPLICATE_DETECTION"
                    result = {**prediction, "adjudication_label": label,
                              "matched_component_id": gt.get("component_id"), "matched_iou": iou}
                    if complete:
                        complete_fp += 1
                elif overlaps:
                    iou, _, gt = overlaps[0]
                    label = "WRONG_CLASS"
                    result = {**prediction, "adjudication_label": label,
                              "matched_component_id": gt.get("component_id"), "matched_iou": iou,
                              "expected_class": gt.get("class")}
                    if complete:
                        complete_fp += 1
                elif complete:
                    label = "FALSE_POSITIVE"
                    result = {**prediction, "adjudication_label": label,
                              "matched_component_id": None, "matched_iou": None}
                    complete_fp += 1
                else:
                    label = "UNRESOLVED_UNREVIEWED_OR_PARTIAL_GROUND_TRUTH"
                    result = {**prediction, "adjudication_label": label,
                              "matched_component_id": None, "matched_iou": None}
            labeled_predictions[pred_index] = result
            all_labels[label] += 1

        missed = []
        for gt_index, gt in enumerate(reviewed):
            if gt_index in used_truth:
                continue
            label = "MISSED_COMPONENT" if complete else "MISSED_COMPONENT_CANDIDATE"
            missed.append({"component_id": gt.get("component_id"), "class": gt.get("class"),
                           "bbox_xyxy": gt.get("bbox_xyxy"), "adjudication_label": label})
            all_labels[label] += 1
            if complete:
                complete_fn += 1

        if complete:
            status = "FULLY_ADJUDICATED"
        elif reviewed:
            status = "PARTIALLY_ADJUDICATED_UNMATCHED_ITEMS_UNRESOLVED"
        else:
            status = "UNRESOLVED_NO_REVIEWED_GROUND_TRUTH"
        output_images.append({"image_id": image_id, "status": status,
                              "ground_truth_completeness": "COMPLETE_REVIEWED" if complete else "PARTIAL_OR_UNREVIEWED",
                              "reviewed_ground_truth_count": len(reviewed),
                              "predictions": labeled_predictions, "missed_components": missed})

    precision = complete_tp / (complete_tp + complete_fp) if complete_tp + complete_fp else None
    recall = complete_tp / (complete_tp + complete_fn) if complete_tp + complete_fn else None
    return {"schema_version": "circuitcare.detector-adjudication.v1",
            "status": "SCORED_REVIEWED_IMAGES" if complete_image_count else "BLOCKED_NO_COMPLETE_REVIEWED_IMAGES",
            "matching": {"method": "CLASS_AWARE_ONE_TO_ONE_BBOX_IOU", "iou_threshold": float(iou_threshold),
                         "coordinate_space": "ABSOLUTE_IMAGE_PIXELS"},
            "summary": {"image_count": len(output_images), "fully_reviewed_image_count": complete_image_count,
                        "correct_detection_count": all_labels["CORRECT"],
                        "wrong_class_count": all_labels["WRONG_CLASS"],
                        "false_positive_count": complete_fp, "false_negative_count": complete_fn,
                        "duplicate_detection_count": all_labels["DUPLICATE_DETECTION"],
                        "unresolved_prediction_count": sum(value for key, value in all_labels.items()
                            if key.startswith("UNRESOLVED")),
                        "precision_on_fully_reviewed_images": precision,
                        "recall_on_fully_reviewed_images": recall,
                        "label_counts": dict(all_labels)},
            "ground_truth_policy": "Only REVIEWED_GROUND_TRUTH and provenance-backed REVIEWED_EMPTY images can establish correctness. Draft, source-provided, partial, or missing labels never turn an unmatched detection into a false positive.",
            "images": output_images}


def _is_complete_review(image: dict[str, Any], annotations: list[dict[str, Any]],
                        reviewed: list[dict[str, Any]]) -> bool:
    if image.get("label_state") == "REVIEWED_EMPTY":
        proof = image.get("review_provenance") or {}
        return bool(image.get("reviewer") and proof.get("reviewed_at") and proof.get("review_method")) and not annotations
    return (image.get("label_state") == "LABELED" and len(reviewed) == len(annotations)
            and len(annotations) > 0 and all(
                row.get("reviewer") and row.get("small_component") is not None
                and (row.get("provenance") or {}).get("reviewed_at")
                and (row.get("provenance") or {}).get("review_method")
                for row in reviewed))


def _prediction_class(prediction: dict[str, Any]) -> Any:
    for key in ("model_class_candidate", "predicted_class", "class_name"):
        if prediction.get(key) is not None:
            return prediction[key]
    return None


def _confidence(prediction: dict[str, Any]) -> float:
    try:
        return float(prediction.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _valid_box(box: Any) -> bool:
    try:
        x1, y1, x2, y2 = map(float, box)
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(x) for x in (x1, y1, x2, y2)) and x1 < x2 and y1 < y2
