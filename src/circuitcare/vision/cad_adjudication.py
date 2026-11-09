"""Automatic photo-to-CAD registration and component-location adjudication.

The output is explicitly a CAD-reference comparison, not reviewed visual truth.
Poor or ambiguous registration fails closed and produces UNRESOLVED labels.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image


CLASS_MAP = {
    "resistor": "resistor", "resistors": "resistor",
    "capacitor": "capacitor", "capacitors": "capacitor",
    "diode": "diode", "esd diode": "diode", "tvs diode": "diode",
    "led": "led", "light emitting diode": "led",
    "connector": "connector", "connectors": "connector", "usb connector": "connector",
    "header": "header", "pin header": "header", "test point": "debug_test_point",
    "debug/test point": "debug_test_point", "crystal": "crystal", "resonator": "crystal",
    "switch": "switch_button", "button": "switch_button", "pushbutton": "switch_button",
    "integrated circuit": "integrated_circuit", "ic": "integrated_circuit", "mcu": "integrated_circuit",
}


def auto_register_photo(photo_path: Path, renders: dict[str, Path], *, preferred_side: str | None = None) -> dict[str, Any]:
    """Estimate CAD-render-to-photo homography with SIFT; accept only strong fits."""
    with Image.open(photo_path) as opened:
        photo = cv2.cvtColor(np.asarray(opened.convert("RGB")), cv2.COLOR_RGB2GRAY)
        full_height, full_width = photo.shape
    scale = min(1.0, 1800.0 / max(full_width, full_height))
    if scale < 1:
        photo = cv2.resize(photo, (round(full_width * scale), round(full_height * scale)), interpolation=cv2.INTER_AREA)
    if not hasattr(cv2, "SIFT_create"):
        return {"status": "UNRESOLVED_SIFT_UNAVAILABLE", "candidates": {}}
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.025, edgeThreshold=12)
    photo_keypoints, photo_descriptors = sift.detectAndCompute(photo, None)
    if photo_descriptors is None or len(photo_keypoints) < 8:
        return {"status": "UNRESOLVED_NO_PHOTO_FEATURES", "candidates": {}}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    results: dict[str, Any] = {}
    accepted = []
    for side, path in renders.items():
        if preferred_side and side != preferred_side:
            continue
        template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            results[side] = {"status": "UNRESOLVED_RENDER_UNREADABLE"}
            continue
        keypoints, descriptors = sift.detectAndCompute(template, None)
        if descriptors is None or len(keypoints) < 8:
            results[side] = {"status": "UNRESOLVED_NO_CAD_FEATURES"}
            continue
        pairs = matcher.knnMatch(descriptors, photo_descriptors, k=2)
        good = [a for a, b in pairs if a.distance < 0.72 * b.distance]
        matrix = None
        inliers = 0
        median_error = None
        source_coverage = 0.0
        target_coverage = 0.0
        if len(good) >= 4:
            source = np.float32([keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            target = np.float32([photo_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0, maxIters=8000, confidence=0.999)
            if matrix is not None and mask is not None:
                selected = mask.ravel().astype(bool)
                inliers = int(selected.sum())
                projected = cv2.perspectiveTransform(source, matrix).reshape(-1, 2)
                target_points = target.reshape(-1, 2)
                errors = np.linalg.norm(projected - target_points, axis=1)
                median_error = float(np.median(errors[selected])) if inliers else None
                source_coverage = _hull_coverage(source.reshape(-1, 2)[selected], template.shape[1], template.shape[0])
                target_coverage = _hull_coverage(target_points[selected], photo.shape[1], photo.shape[0])
        inlier_ratio = inliers / max(len(good), 1)
        board_fraction = 0.0
        projected_corners = None
        if matrix is not None:
            corners = np.float32([[0, 0], [template.shape[1]-1, 0],
                                  [template.shape[1]-1, template.shape[0]-1], [0, template.shape[0]-1]]).reshape(-1, 1, 2)
            projected_corners = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
            board_fraction = abs(float(cv2.contourArea(projected_corners.astype(np.float32)))) / max(photo.shape[0]*photo.shape[1], 1)
        quality = (inliers >= 30 and inlier_ratio >= 0.20 and source_coverage >= 0.06
                   and target_coverage >= 0.005 and median_error is not None and median_error <= 4.0
                   and 0.04 <= board_fraction <= 0.90 and _convex_quad(projected_corners))
        full_matrix = None
        if quality and matrix is not None:
            # Convert reduced-photo coordinates back to native pixels.
            scale_back = np.asarray([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]], dtype=np.float64)
            full_matrix = scale_back @ matrix
            full_matrix /= full_matrix[2, 2]
            accepted.append((side, inliers, full_matrix))
        results[side] = {
            "status": "AUTO_REGISTRATION_CANDIDATE" if quality else "UNRESOLVED_WEAK_OR_IMPLAUSIBLE_FIT",
            "good_matches": len(good), "ransac_inliers": inliers, "inlier_ratio": inlier_ratio,
            "median_reprojection_error_px_reduced_image": median_error,
            "source_inlier_hull_fraction": source_coverage, "target_inlier_hull_fraction": target_coverage,
            "projected_board_fraction": board_fraction,
            "render_to_photo_homography": full_matrix.tolist() if full_matrix is not None else None,
        }
    accepted.sort(key=lambda x: (-x[1], x[0]))
    if not accepted:
        status, chosen = "UNRESOLVED_REGISTRATION", None
    elif preferred_side:
        chosen = accepted[0]
        status = "AUTO_REGISTERED_CANDIDATE"
    elif len(accepted) > 1 and accepted[0][1] < accepted[1][1] * 1.25:
        status, chosen = "UNRESOLVED_AMBIGUOUS_BOARD_SIDE", None
    else:
        chosen = accepted[0]
        status = "AUTO_REGISTERED_CANDIDATE"
    return {"status": status, "selected_side": chosen[0] if chosen else None,
            "render_to_photo_homography": chosen[2].tolist() if chosen else None,
            "photo_size_px": [full_width, full_height], "preferred_side_from_source": preferred_side,
            "candidates": results,
            "acceptance_policy": "SIFT/RANSAC: >=30 inliers, >=0.20 ratio, >=0.06 source coverage, <=4px median reprojection error, plausible projected board area; side ambiguity blocks selection."}


def adjudicate_pair_manifest(pair_manifest_path: Path, *, predictions_by_image: dict[str, list[dict[str, Any]]] | None = None,
                             location_margin_px: float = 4.0) -> dict[str, Any]:
    """Register supported single-board packages and assign CAD-reference labels."""
    manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    root = pair_manifest_path.parent
    multi_board_mapping_blocked = bool(manifest.get("boards") and any(
        not row.get("board_id") for row in manifest.get("images", [])))
    images_out = []
    for image in manifest.get("images", []):
        image_id = str(image.get("image_id", ""))
        photo = root / image.get("photo_path", "")
        cad_relative = image.get("cad_path")
        board = root / cad_relative if cad_relative else None
        if not photo.is_file() or not board or not board.is_file():
            images_out.append({"image_id": image_id, "status": "BLOCKED_MISSING_PHOTO_OR_BOARD_CAD"})
            continue
        render_dir = board.parent / "render"
        if not render_dir.is_dir():
            render_dir = root / "cad/render"
        renders = {side: render_dir / f"{side}.png" for side in ("top", "bottom")}
        renders = {side: path for side, path in renders.items() if path.is_file()}
        side_text = str(image.get("photo_side", "unknown")).lower()
        preferred = "top" if side_text in {"front", "top"} else "bottom" if side_text in {"back", "bottom"} else None
        if multi_board_mapping_blocked:
            registration = {"status": "BLOCKED_PER_IMAGE_BOARD_MAPPING", "candidates": {}}
            component_rows, component_source, unclassified_components = [], "PER_IMAGE_BOARD_MAPPING_UNRESOLVED", []
        else:
            registration = auto_register_photo(photo, renders, preferred_side=preferred)
            component_rows, component_source, unclassified_components = _load_components(root, manifest, image)
        projected = []
        if registration.get("status") == "AUTO_REGISTERED_CANDIDATE":
            board_map = _board_to_render_map(board, renders[registration["selected_side"]])
            if board_map.get("status") == "READY":
                render_to_photo = np.asarray(registration["render_to_photo_homography"], dtype=np.float64)
                for component in component_rows:
                    if preferred and component.get("side") not in {None, "unknown", preferred}:
                        continue
                    render_points = _component_render_corners(component, board_map, location_margin_px)
                    pixels = cv2.perspectiveTransform(render_points.reshape(1, -1, 2), render_to_photo).reshape(-1, 2)
                    mm_to_render = np.asarray(board_map["mm_to_render"], dtype=np.float64)
                    center = cv2.perspectiveTransform(np.asarray([[[component["x_mm"], component["y_mm"]]]], dtype=np.float32),
                                                      mm_to_render).reshape(2)
                    center_px = cv2.perspectiveTransform(np.asarray([[center]], dtype=np.float32), render_to_photo).reshape(2)
                    projected.append({**component, "bbox_xyxy": [float(pixels[:,0].min()), float(pixels[:,1].min()),
                                                                   float(pixels[:,0].max()), float(pixels[:,1].max())],
                                      "footprint_polygon_px": pixels.tolist(), "center_xy_px": center_px.tolist(),
                                      "reference_source": "CAD_FOOTPRINT_APPROXIMATION"})
            else:
                registration = {**registration, "status": "UNRESOLVED_CAD_COORDINATE_MAP", "coordinate_map": board_map}
        has_prediction_input = predictions_by_image is not None
        has_image_predictions = has_prediction_input and image_id in predictions_by_image
        preds = (predictions_by_image or {}).get(image_id, [])
        if registration.get("status") != "AUTO_REGISTERED_CANDIDATE":
            labels = [{**pred, "adjudication_label": "UNRESOLVED_REGISTRATION", "matched_component_id": None}
                      for pred in preds]
            image_status = "UNRESOLVED_REGISTRATION"
        elif has_image_predictions:
            labels, missed = _match_component_centers(preds, projected, location_margin_px)
            image_status = "CAD_REFERENCE_LOCATION_ADJUDICATED"
        elif has_prediction_input:
            labels, missed = [], []
            image_status = "CAD_REGISTRATION_READY_NO_PREDICTION_RECORD"
        else:
            labels, missed = [], []
            image_status = "CAD_REGISTRATION_READY_NO_PREDICTIONS"
        images_out.append({"image_id": image_id, "board_id": image.get("board_id"),
                           "photo_path": str(photo), "photo_sha256": _sha(photo),
                           "board_cad_path": str(board), "board_cad_sha256": _sha(board),
                           "component_source": component_source, "registration": registration,
                           "cad_component_count": len(projected), "cad_components": projected,
                           "unclassified_cad_components": unclassified_components,
                           "predictions": labels,
                           "missed_components": missed if registration.get("status") == "AUTO_REGISTERED_CANDIDATE" else [],
                           "status": image_status,
                           "truth_scope": "CAD_REFERENCE_COMPONENT_CENTER_MATCH; not independent visual ground truth."})
    ready = sum(row.get("status") in {"CAD_REFERENCE_LOCATION_ADJUDICATED", "CAD_REGISTRATION_READY_NO_PREDICTIONS"}
                for row in images_out)
    return {"schema_version": "circuitcare.cad-detection-adjudication.v1",
            "status": ("CAD_REFERENCE_LABELS_GENERATED" if ready and predictions_by_image is not None else
                       "REGISTRATION_ONLY_NO_PREDICTIONS" if ready else "BLOCKED_NO_ACCEPTED_PHOTO_CAD_REGISTRATION"),
            "pair_manifest": str(pair_manifest_path.resolve()), "location_margin_px": location_margin_px,
            "images": images_out,
            "note": "CORRECT/WRONG_CLASS/FALSE_POSITIVE labels are CAD-reference comparisons only. They are not detector ground truth unless the board/revision, assembly visibility, and photo registration are independently verified."}


def _match_component_centers(predictions: list[dict[str, Any]], components: list[dict[str, Any]], margin: float):
    candidates = []
    for pi, pred in enumerate(predictions):
        box = pred.get("bbox_xyxy")
        if not box or len(box) != 4:
            candidates.append((pi, None, [])); continue
        center = np.asarray([(float(box[0])+float(box[2]))/2, (float(box[1])+float(box[3]))/2])
        cls = pred.get("model_class_candidate") or pred.get("predicted_class") or pred.get("class_name")
        possible = []
        for ci, comp in enumerate(components):
            polygon = np.asarray(comp.get("footprint_polygon_px", []), dtype=np.float32)
            if polygon.shape != (4, 2):
                continue
            distance = float(cv2.pointPolygonTest(polygon.reshape(-1, 1, 2),
                                                  (float(center[0]), float(center[1])), True))
            outside_distance = max(0.0, -distance)
            if outside_distance <= margin:
                possible.append((outside_distance, ci, comp, cls == comp["class_name"]))
        candidates.append((pi, pred, sorted(possible, key=lambda x: (not x[3], x[0], x[2]["component_id"]))))
    used = set(); labels = [None] * len(predictions)
    order = sorted(range(len(predictions)), key=lambda i: -float(predictions[i].get("confidence", 0) or 0))
    for pi in order:
        _, pred, options = candidates[pi]
        if pred is None:
            labels[pi] = {**predictions[pi], "adjudication_label": "UNRESOLVED_INVALID_BOX"}; continue
        available = [x for x in options if x[1] not in used]
        if available:
            dist, ci, comp, same_class = available[0]
            used.add(ci)
            labels[pi] = {**pred, "adjudication_label": "CORRECT_LOCATION_AND_CLASS" if same_class else "WRONG_CLASS_AT_COMPONENT_LOCATION",
                          "matched_component_id": comp["component_id"], "expected_class": comp["class_name"], "center_distance_px": dist}
        elif options:
            dist, _, comp, _ = options[0]
            labels[pi] = {**pred, "adjudication_label": "DUPLICATE_DETECTION_AT_COMPONENT_LOCATION",
                          "matched_component_id": comp["component_id"], "center_distance_px": dist}
        else:
            labels[pi] = {**pred, "adjudication_label": "FALSE_POSITIVE_CANDIDATE", "matched_component_id": None}
    missed = [{"component_id": comp["component_id"], "class_name": comp["class_name"],
               "center_xy_px": comp["center_xy_px"], "adjudication_label": "MISSED_COMPONENT_CANDIDATE"}
              for ci, comp in enumerate(components) if ci not in used]
    return labels, missed


def _load_components(root: Path, manifest: dict[str, Any], image: dict[str, Any]):
    path = None
    source = image.get("component_labels_path") or image.get("positions_path")
    if source:
        path = root / source
    if not path or not path.is_file():
        for candidate in (root / "components.csv", root / "cad/robotic-hand-positions.csv"):
            if candidate.is_file():
                path = candidate; break
    if not path or not path.is_file():
        return [], "COMPONENT_POSITION_FILE_NOT_FOUND", []
    rows = []
    unclassified = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            reference = (row.get("reference") or row.get("Ref") or "").strip().upper()
            population = (row.get("bom_population") or row.get("Populate") or "yes").strip().lower()
            if population in {"no", "dnp", "false", "0", "not fitted", "do not populate"}:
                continue
            x = _number(row.get("x_mm") or row.get("PosX"))
            y = _number(row.get("y_mm") or row.get("PosY"))
            if x is None or y is None:
                continue
            footprint = row.get("footprint") or row.get("Package") or ""
            cls = _component_class(row, reference)
            if cls is None:
                unclassified.append({"reference": reference, "footprint": footprint,
                                     "reason": "NO_CONFIDENT_CLASS_MAPPING"})
                continue
            # KiCad position exports in these source packages use inverted Y.
            y = -y
            side_raw = (row.get("side") or row.get("Side") or "unknown").strip().lower()
            side = "top" if side_raw in {"top", "front", "f.cu"} else "bottom" if side_raw in {"bottom", "back", "b.cu"} else "unknown"
            width_mm, height_mm = _footprint_size(footprint)
            rows.append({"component_id": reference, "reference": reference, "class_name": cls,
                         "x_mm": x, "y_mm": y, "side": side,
                         "width_mm": width_mm, "height_mm": height_mm,
                         "rotation_deg": _number(row.get("rotation_deg") or row.get("Rot")) or 0.0,
                         "footprint": footprint})
    return rows, str(path), unclassified


def _component_class(row: dict[str, str], reference: str) -> str | None:
    category = (row.get("bom_category") or "").strip().lower()
    mapped = CLASS_MAP.get(category)
    if mapped:
        return mapped
    footprint = (row.get("footprint") or row.get("Package") or "").strip().lower()
    value = (row.get("value") or row.get("Val") or "").strip().lower()
    # Prefer explicit footprint/value evidence over a reference prefix. Some
    # designs use unconventional references (for example U-designators for
    # connectors), so prefix-only inference can silently create wrong labels.
    if any(token in footprint for token in ("jst", "terminalblock", "terminal_block", "bornier", "usb", "connector", "molex")):
        return "connector"
    if "header" in footprint or "pinheader" in footprint:
        return "header"
    if any(token in footprint or token in value for token in ("resistor", "r_")):
        return "resistor"
    if any(token in footprint or token in value for token in ("capacitor", "c_")):
        return "capacitor"
    if re.search(r"(^|[^a-z])led([^a-z]|$)", footprint) or re.search(r"(^|[^a-z])led([^a-z]|$)", value):
        return "led"
    if any(token in footprint or token in value for token in ("diode", "sod", "do-")):
        return "diode"
    if any(token in footprint for token in ("switch", "button", "tactile")):
        return "switch_button"
    if any(token in footprint for token in ("testpoint", "test_point", "test point")):
        return "debug_test_point"
    if any(token in footprint for token in ("crystal", "resonator")):
        return "crystal"
    if reference.startswith(("U", "IC")) and any(token in footprint for token in
                                                   ("soic", "tssop", "qfn", "tqfp", "bga", "dfn", "sot-", "sot_")):
        return "integrated_circuit"
    if "integrated circuit" in value or value.startswith(("ic", "mcu")):
        return "integrated_circuit"
    prefix = re.match(r"[A-Z]+", reference).group(0) if re.match(r"[A-Z]+", reference) else ""
    return {"R": "resistor", "C": "capacitor", "D": "diode", "LED": "led",
            "IC": "integrated_circuit", "J": "connector",
            "P": "header", "X": "crystal", "Y": "crystal", "SW": "switch_button",
            "S": "switch_button", "TP": "debug_test_point"}.get(prefix)


def _footprint_size(name: str) -> tuple[float, float]:
    metrics = {"0201": (0.6, 0.3), "0402": (1.0, 0.5), "0603": (1.6, 0.8), "0805": (2.0, 1.25),
               "1206": (3.2, 1.6), "1210": (3.2, 2.5), "1812": (4.5, 3.2), "2010": (5.0, 2.5), "2512": (6.3, 3.2)}
    upper = name.upper()
    connector_match = re.search(r"1X(\d+)_P(\d+(?:\.\d+)?)MM", upper)
    if connector_match and any(token in upper for token in ("JST", "HEADER", "CONN")):
        pins, pitch = int(connector_match.group(1)), float(connector_match.group(2))
        return max(5.0, (pins - 1) * pitch + 2.5), 8.0
    if "TERMINALBLOCK" in upper or "BORNIER" in upper:
        return (10.5, 8.0)
    if "OLED" in upper or "DISPLAY" in upper or "ESP32" in upper or "CAM" in upper:
        return (30.0, 22.0)
    if "GLYPH-S3" in upper:
        return (25.0, 20.0)
    for code, dims in metrics.items():
        if re.search(rf"(?:^|[_-]){code}(?:[_-]|$)", upper) or f"{code}METRIC" in upper:
            return dims
    patterns = ((r"SOT.?23", (3.0, 1.75)), (r"SOT.?223", (6.7, 3.7)),
                (r"SOIC.?8", (5.0, 4.0)), (r"QFN.?32", (5.0, 5.0)),
                (r"QFN.?48", (7.0, 7.0)), (r"TQFP.?48", (7.0, 7.0)),
                (r"USB.?C|TYPE.?C", (9.0, 7.0)), (r"CRYSTAL", (3.2, 2.5)))
    for pattern, dims in patterns:
        if re.search(pattern, upper): return dims
    return (2.0, 2.0)


def _component_render_corners(component: dict[str, Any], board_map: dict[str, Any], margin_px: float) -> np.ndarray:
    w, h = component["width_mm"], component["height_mm"]
    angle = math.radians(component["rotation_deg"])
    local = np.asarray([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]], dtype=np.float32)
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
    corners = local @ rotation.T + np.asarray([component["x_mm"], component["y_mm"]], dtype=np.float32)
    matrix = np.asarray(board_map["mm_to_render"], dtype=np.float64)
    return cv2.perspectiveTransform(corners.reshape(1, -1, 2), matrix).reshape(-1, 2)


def _board_to_render_map(board_path: Path, render_path: Path) -> dict[str, Any]:
    bounds = _edge_cuts_bounds(board_path.read_text(encoding="utf-8", errors="replace"))
    if not bounds:
        return {"status": "UNRESOLVED_EDGE_CUTS_BOUNDS"}
    try:
        svg_path = render_path.with_suffix(".svg")
        tree = ET.parse(svg_path)
        svg = tree.getroot()
        viewbox = [float(x) for x in svg.attrib["viewBox"].replace(",", " ").split()]
        _, _, vbw, vbh = viewbox
        with Image.open(render_path) as image:
            width, height = image.size
        xmin, ymin, xmax, ymax = bounds
        bw, bh = xmax - xmin, ymax - ymin
        if bw <= 0 or bh <= 0 or abs(vbw-bw)/bw > 0.05 or abs(vbh-bh)/bh > 0.05:
            return {"status": "UNRESOLVED_SVG_AND_BOARD_BOUNDS_DIFFER", "board_bounds_mm": bounds,
                    "svg_viewbox": viewbox}
        scale_x, scale_y = width/vbw, height/vbh
        matrix = np.asarray([[scale_x, 0, -xmin*scale_x], [0, scale_y, -ymin*scale_y], [0, 0, 1]], dtype=np.float64)
        return {"status": "READY", "mm_to_render": matrix.tolist(), "board_bounds_mm": bounds,
                "svg_viewbox": viewbox, "render_size_px": [width, height]}
    except Exception as exc:
        return {"status": "UNRESOLVED_RENDER_COORDINATE_MAP", "error": str(exc)}


def _edge_cuts_bounds(text: str) -> tuple[float, float, float, float] | None:
    starts = []
    for match in re.finditer(r"\((gr_line|gr_rect|gr_arc|gr_circle|gr_poly)\b", text):
        start = match.start(); depth = 0; quoted = False; escaped = False; end = start
        for i in range(start, len(text)):
            char = text[i]
            if quoted:
                if escaped: escaped = False
                elif char == "\\": escaped = True
                elif char == '"': quoted = False
                continue
            if char == '"': quoted = True
            elif char == "(": depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0: end = i + 1; break
        block = text[start:end]
        if '(layer "Edge.Cuts")' not in block: continue
        points = [(float(point[0]), float(point[1])) for point in
                  re.findall(r"\((?:start|end|mid|center)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)", block)]
        if match.group(1) == "gr_rect" and len(points) >= 2:
            (x1, y1), (x2, y2) = points[:2]
            starts.extend(((x1, y1), (x1, y2), (x2, y1), (x2, y2)))
        elif match.group(1) == "gr_circle" and len(points) >= 2:
            (cx, cy), (ex, ey) = points[:2]
            radius = math.hypot(ex-cx, ey-cy)
            starts.extend(((cx-radius, cy-radius), (cx-radius, cy+radius),
                           (cx+radius, cy-radius), (cx+radius, cy+radius)))
        else:
            starts.extend(points)
        if match.group(1) == "gr_poly":
            for point in re.findall(r"\(xy\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)", block):
                starts.append((float(point[0]), float(point[1])))
    if len(starts) < 4: return None
    xs, ys = zip(*starts)
    return min(xs), min(ys), max(xs), max(ys)


def _hull_coverage(points: np.ndarray, width: int, height: int) -> float:
    if len(points) < 3: return 0.0
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    return abs(float(cv2.contourArea(hull))) / max(width * height, 1)


def _convex_quad(points: np.ndarray | None) -> bool:
    if points is None or len(points) != 4 or not np.isfinite(points).all(): return False
    return abs(float(cv2.contourArea(points.astype(np.float32)))) > 1


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()
