#!/usr/bin/env python3
"""Grouped benchmark for local and long-range schematic connection candidates.

Ground truth comes from KiCad exported netlists. Input evidence comes only from
legacy schematic placement, wire/label geometry, and library pin metadata.
Product repositories are kept intact across train/test folds.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from lightgbm import LGBMClassifier
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from circuitcare.schematic.identity import parse_sexpr  # noqa: E402

DEFAULT_INPUT = ROOT / "data/component_training/raw/external/pcb_photo_cad_pairs/catalog_repositories"
DEFAULT_OUTPUT = ROOT / "experiments/connectivity_candidates/results/dual_path_benchmark.json"
POWER_NAMES = {"GND", "AGND", "DGND", "PGND", "VSS", "VSSA", "VSSD", "VCC", "VDD", "VIN", "VOUT", "VBAT", "VBUS", "VUSB", "+3V3", "+5V", "+12V", "3V3", "5V", "12V"}


def child(node, key):
    return next((x for x in node if isinstance(x, list) and x and x[0] == key), None)


def children(node, key):
    return [x for x in node if isinstance(x, list) and x and x[0] == key]


def atom(node, key, default=""):
    item = child(node, key)
    return str(item[1]) if item is not None and len(item) > 1 else default


def legacy_positions(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    positions = {}
    power_symbols = []
    for block in re.findall(r"\$Comp\s+(.*?)\$EndComp", text, flags=re.S):
        lib_match = re.search(r"^L\s+(\S+)\s+(\S+)\s*$", block, flags=re.M)
        pos_match = re.search(r"^P\s+(-?\d+)\s+(-?\d+)\s*$", block, flags=re.M)
        value_match = re.search(r'^F\s+1\s+"([^"]+)"', block, flags=re.M)
        if lib_match and pos_match:
            library_symbol, ref = lib_match.groups()
            value = value_match.group(1).strip() if value_match else library_symbol.rsplit(":", 1)[-1]
            if ref.startswith("#PWR") and value.upper() not in {"PWR_FLAG", "NC", "N/C"}:
                power_symbols.append({"position": (int(pos_match.group(1)), int(pos_match.group(2))),
                                      "name": value.upper()})
            if not ref.startswith("#"):
                positions[ref] = (int(pos_match.group(1)), int(pos_match.group(2)))
    return positions, power_symbols, text


def parse_wire_components(text: str, positions: dict[str, tuple[int, int]], power_symbols):
    """Build approximate wire-evidence groups from exact legacy wire/label geometry."""
    segments = []
    lines = text.splitlines()
    labels = []
    for i, line in enumerate(lines):
        if line.strip() == "Wire Wire Line" and i + 1 < len(lines):
            m = re.match(r"\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)", lines[i + 1])
            if m:
                x1, y1, x2, y2 = map(int, m.groups())
                segments.append(((x1, y1), (x2, y2)))
        m = re.match(r"\s*Text (?:Label|GLabel|HLabel)\s+(-?\d+)\s+(-?\d+)\b", line)
        if m and i + 1 < len(lines):
            labels.append(((int(m.group(1)), int(m.group(2))), lines[i + 1].strip()))
    graph = nx.Graph()
    for a, b in segments:
        graph.add_edge(("p", *a), ("p", *b), evidence="wire")

    def point_segment_distance(p, a, b):
        vx, vy = b[0] - a[0], b[1] - a[1]
        den = vx * vx + vy * vy
        t = 0.0 if den == 0 else max(0.0, min(1.0, ((p[0]-a[0])*vx + (p[1]-a[1])*vy) / den))
        q = (a[0] + t * vx, a[1] + t * vy)
        return math.hypot(p[0]-q[0], p[1]-q[1]), q

    # Labels are attached to nearby wire segments; equal labels join islands.
    for p, name in labels:
        if not name:
            continue
        best = min(((point_segment_distance(p, a, b)[0], point_segment_distance(p, a, b)[1])
                    for a, b in segments), default=(float("inf"), p), key=lambda x: x[0])
        if best[0] <= 250:
            q = ("p", round(best[1][0]), round(best[1][1]))
            graph.add_node(q)
            graph.add_edge(q, ("label", name), evidence="label")
    # Legacy KiCad stores global power symbols as #PWR components rather than
    # Text Label records. Link each symbol to its local wire and join equal names.
    for symbol in power_symbols:
        p = symbol["position"]
        best = min(((point_segment_distance(p, a, b)[0], point_segment_distance(p, a, b)[1])
                    for a, b in segments), default=(float("inf"), p), key=lambda x: x[0])
        if best[0] <= 600:
            q = ("p", round(best[1][0]), round(best[1][1]))
            graph.add_edge(q, ("label", "power:" + symbol["name"]), evidence="power_symbol")
    # Component body centers are an imperfect stand-in for pins. Attach only the
    # nearest wire component within 600 mil to limit accidental joins.
    seg_endpoints = sorted({pt for seg in segments for pt in seg})
    comp_wire_nodes = defaultdict(set)
    for ref, pos in positions.items():
        near = sorted((math.hypot(pos[0]-p[0], pos[1]-p[1]), p) for p in seg_endpoints)
        if near and near[0][0] <= 600:
            pnode = ("p", *near[0][1])
            cnode = ("c", ref)
            graph.add_edge(cnode, pnode, evidence="component_wire_nearby")
            comp_wire_nodes[ref].add(pnode)
    refs_by_wire_group = defaultdict(set)
    for members in nx.connected_components(graph):
        refs = {str(n[1]) for n in members if len(n) == 2 and n[0] == "c"}
        for ref in refs:
            refs_by_wire_group[ref].update(refs - {ref})
    wire_pairs = set()
    for refs in refs_by_wire_group.values():
        wire_pairs.update(tuple(sorted(pair)) for pair in itertools.combinations(refs, 2))
    return wire_pairs


def parse_design(net_path: Path, input_root: Path = DEFAULT_INPUT):
    tree = parse_sexpr(net_path.read_text(encoding="utf-8", errors="replace"))
    components_group = child(tree, "components")
    libparts_group = child(tree, "libparts")
    nets_group = child(tree, "nets")
    if not (components_group and libparts_group and nets_group):
        return None
    comps, libpart_pins = {}, {}
    for row in children(components_group, "comp"):
        ref = atom(row, "ref")
        libsrc = child(row, "libsource")
        if ref and libsrc:
            comps[ref] = {"lib": atom(libsrc, "lib"), "part": atom(libsrc, "part"), "value": atom(row, "value")}
    for row in children(libparts_group, "libpart"):
        key = (atom(row, "lib"), atom(row, "part"))
        pins = child(row, "pins")
        libpart_pins[key] = [{"num": atom(p, "num"), "name": atom(p, "name").upper(), "type": atom(p, "type").lower()}
                             for p in children(pins, "pin")] if pins else []
    comp_pins = {ref: libpart_pins.get((v["lib"], v["part"]), []) for ref, v in comps.items()}
    positive = set()
    pin_truth_pairs = set()
    intra_component_pin_pairs = set()
    for row in children(nets_group, "net"):
        nodes = sorted({(atom(n, "ref"), atom(n, "pin")) for n in children(row, "node")
                        if atom(n, "ref") in comps and atom(n, "pin")})
        refs = sorted({ref for ref, _ in nodes})
        positive.update(tuple(pair) for pair in itertools.combinations(refs, 2))
        for left, right in itertools.combinations(nodes, 2):
            if left[0] == right[0]:
                intra_component_pin_pairs.add(tuple(sorted((left, right))))
            else:
                pin_truth_pairs.add(tuple(sorted((left, right))))
    sch = net_path.with_suffix(".sch")
    pcb = net_path.with_suffix(".kicad_pcb")
    if not sch.is_file() or not pcb.is_file():
        return None
    positions, power_symbols, sch_text = legacy_positions(sch)
    refs = sorted(set(comps) & set(positions))
    if len(refs) < 2:
        return None
    positive = {p for p in positive if p[0] in positions and p[1] in positions and p[0] != p[1]}
    pin_truth_pairs = {p for p in pin_truth_pairs if p[0][0] in positions and p[1][0] in positions}
    pin_truth_neighbors = defaultdict(set)
    for a, b in pin_truth_pairs:
        pin_truth_neighbors[a].add(b); pin_truth_neighbors[b].add(a)
    all_pins = {ref: [(pin["num"], pin["name"], pin["type"]) for pin in comp_pins.get(ref, [])]
                for ref in refs}
    wire_pairs = parse_wire_components(sch_text, positions, power_symbols)
    relative = net_path.relative_to(input_root)
    return {"design_id": str(relative),
            "family": relative.parts[0], "components": comps,
            "comp_pins": comp_pins, "positions": {r: positions[r] for r in refs},
            "all_pins": all_pins, "positive": positive, "pin_truth_pairs": pin_truth_pairs,
            "pin_truth_neighbors": dict(pin_truth_neighbors),
            "intra_component_pin_pairs": intra_component_pin_pairs,
            "wire_pairs": wire_pairs, "net_path": str(net_path),
            "schematic_path": str(sch), "component_count": len(refs)}


def power_tokens(pins):
    out = set()
    for pin in pins:
        name = re.sub(r"\s+", "", pin["name"].upper())
        if name in POWER_NAMES or re.fullmatch(r"\+?\d+(?:\.\d+)?V", name):
            if name in {"GND", "AGND", "DGND", "PGND", "VSS", "VSSA", "VSSD"}:
                out.add("power:gnd")
            else:
                out.add("power:" + name)
    return out


def pin_tokens(pins):
    return {re.sub(r"[^A-Z0-9+]+", "", pin["name"].upper()) for pin in pins if pin["name"]}


def candidate_sets(design, radius_mm=250.0, k=96):
    refs = list(design["positions"])
    xy = np.asarray([[design["positions"][r][0] * 0.0254, design["positions"][r][1] * 0.0254] for r in refs])
    tree = cKDTree(xy)
    local = set()
    for i, ref in enumerate(refs):
        idx = tree.query_ball_point(xy[i], radius_mm)
        idx = [j for j in idx if j != i]
        idx.sort(key=lambda j: (float(np.linalg.norm(xy[i] - xy[j])), refs[j]))
        for j in idx[:k]:
            local.add(tuple(sorted((ref, refs[j]))))

    # NetworkX role graph gives long-range candidates with a shared confirmed
    # ground/supply pin function. Ubiquitous role nodes are bounded.
    role_graph = nx.Graph()
    for ref in refs:
        role_graph.add_node("c:" + ref, bipartite="component")
        pins = design["comp_pins"].get(ref, [])
        for token in power_tokens(pins):
            role_graph.add_node("r:" + token, bipartite="role")
            role_graph.add_edge("c:" + ref, "r:" + token)
    long_power = set()
    for role in [n for n in role_graph if str(n).startswith("r:")]:
        owners = sorted(n[2:] for n in role_graph.neighbors(role))
        if 2 <= len(owners) <= 60:
            long_power.update(tuple(pair) for pair in itertools.combinations(owners, 2))
    long_wire = {p for p in design["wire_pairs"] if p[0] in design["positions"] and p[1] in design["positions"]}
    dual = local | long_power | long_wire
    return local, long_power, long_wire, dual


FEATURES = ["distance_mm", "dx_mm", "dy_mm", "abs_dx_mm", "abs_dy_mm", "direction_cos", "direction_sin",
            "pins_a", "pins_b", "pin_count_ratio", "shared_pin_name_count", "shared_ground_role",
            "shared_supply_name_count", "wire_evidence", "power_candidate", "local_candidate"]


def candidate_row(design, pair, local, long_power, long_wire):
    a, b = pair
    pa, pb = design["positions"][a], design["positions"][b]
    dx = (pb[0] - pa[0]) * .0254
    dy = (pb[1] - pa[1]) * .0254
    dist = math.hypot(dx, dy)
    ap, bp = design["comp_pins"].get(a, []), design["comp_pins"].get(b, [])
    names_a, names_b = pin_tokens(ap), pin_tokens(bp)
    pow_a, pow_b = power_tokens(ap), power_tokens(bp)
    shared_power = pow_a & pow_b
    supply = {x for x in shared_power if x != "power:gnd"}
    return [dist, dx, dy, abs(dx), abs(dy), dx / max(dist, 1e-9), dy / max(dist, 1e-9),
            len(ap), len(bp), min(len(ap), len(bp)) / max(1, max(len(ap), len(bp))), len(names_a & names_b),
            float("power:gnd" in shared_power), len(supply), float(pair in long_wire), float(pair in long_power), float(pair in local)]


def binary_metrics(y, pred):
    tp = int(np.sum((y == 1) & pred)); fp = int(np.sum((y == 0) & pred)); fn = int(np.sum((y == 1) & ~pred))
    return {"precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "predicted_edges": int(pred.sum()), "true_positive_edges": tp, "false_positive_edges": fp,
            "false_negative_edges": fn}


def best_f1_threshold(y, scores):
    thresholds = sorted(set([0.0, 1.0, *map(float, np.quantile(scores, np.linspace(0, 1, 101)))]))
    best = (0.5, -1.0)
    for threshold in thresholds:
        m = binary_metrics(y, scores >= threshold)
        if m["f1"] > best[1]:
            best = (threshold, m["f1"])
    return best[0]


def examples_for_designs(designs, variant):
    x_rows, y_rows, group_rows = [], [], []
    for design in designs:
        local, power, wire = design["pools"]["local"], design["pools"]["power"], design["pools"]["wire"]
        for pair in sorted(design["pools"][variant]):
            x_rows.append(candidate_row(design, pair, local, power, wire))
            y_rows.append(int(pair in design["positive"]))
            group_rows.append(design["family"])
    return np.asarray(x_rows), np.asarray(y_rows), np.asarray(group_rows)


def new_lgbm():
    return LGBMClassifier(n_estimators=160, learning_rate=.05, num_leaves=15, max_depth=6,
                          min_child_samples=25, reg_lambda=1.0, class_weight="balanced",
                          random_state=42, n_jobs=-1, verbosity=-1)


def grouped_oof_calibration(train, variant):
    """Fit a calibration map and decision threshold from grouped OOF rows only."""
    groups = sorted({d["family"] for d in train})
    splitter = GroupKFold(n_splits=min(4, len(groups)))
    oof_scores, oof_labels, oof_groups = [], [], []
    for fit_idx, valid_idx in splitter.split(np.arange(len(train)), groups=[d["family"] for d in train]):
        fit_designs = [train[i] for i in fit_idx]
        valid_designs = [train[i] for i in valid_idx]
        x_fit, y_fit, _ = examples_for_designs(fit_designs, variant)
        x_valid, y_valid, _ = examples_for_designs(valid_designs, variant)
        if len(np.unique(y_fit)) < 2 or len(x_valid) == 0:
            continue
        model = new_lgbm()
        model.fit(x_fit, y_fit)
        oof_scores.extend(model.predict_proba(x_valid)[:, 1].tolist())
        oof_labels.extend(y_valid.tolist())
        oof_groups.extend([d["family"] for d in valid_designs for _ in d["pools"][variant]])
    labels = np.asarray(oof_labels, dtype=np.int8)
    scores = np.asarray(oof_scores, dtype=float)
    oof_groups = np.asarray(oof_groups)
    if len(np.unique(labels)) < 2:
        calibrator = None
        threshold = 0.5
        calibration_name = "none_single_class_oof"
    else:
        # Keep calibration and threshold-selection families separate. Both are
        # OOF to the feature model, and the outer test family is untouched.
        unique_groups = np.asarray(sorted(set(oof_groups)))
        splitter = GroupShuffleSplit(n_splits=1, test_size=max(1 / len(unique_groups), 0.2), random_state=42)
        cal_group_idx, threshold_group_idx = next(splitter.split(unique_groups, groups=unique_groups))
        cal_groups = set(unique_groups[cal_group_idx])
        threshold_groups = set(unique_groups[threshold_group_idx])
        cal_mask = np.asarray([group in cal_groups for group in oof_groups])
        threshold_mask = np.asarray([group in threshold_groups for group in oof_groups])
        # Isotonic calibration corrects class-balanced scores against natural
        # validation prevalence; threshold families are disjoint from this fit.
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(scores[cal_mask], labels[cal_mask])
        threshold_scores = calibrator.predict(scores[threshold_mask])
        threshold = best_f1_threshold(labels[threshold_mask], threshold_scores)
        calibration_name = "isotonic_oof_calibration_separate_oof_threshold_groups"
    return calibrator, threshold, {"oof_rows": int(len(labels)),
                                   "oof_positive_rate": float(labels.mean()) if len(labels) else None,
                                   "decision_threshold": float(threshold),
                                   "calibration_fit_groups": sorted(cal_groups) if calibrator else [],
                                   "threshold_validation_groups": sorted(threshold_groups) if calibrator else [],
                                   "calibration": calibration_name}


def pin_query_metrics(design, candidate_pairs, candidate_scores, accepted_pairs):
    """Pin-level retrieval from component-pair scores; never claim pin-localization."""
    pin_lists = design["all_pins"]
    truths = design["pin_truth_neighbors"]
    queries = [pin for pin, neighbors in truths.items() if neighbors]
    p_at = {k: [] for k in (1, 3, 5)}
    for source in queries:
        source_ref = source[0]
        ranked = []
        for pair, score in zip(candidate_pairs, candidate_scores):
            if pair[0] == source_ref:
                target_ref = pair[1]
            elif pair[1] == source_ref:
                target_ref = pair[0]
            else:
                continue
            for pin_num, _pin_name, _pin_type in pin_lists.get(target_ref, []):
                target = (target_ref, str(pin_num))
                ranked.append((float(score), target))
        ranked.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
        relevant = truths[source]
        for k in p_at:
            hits = sum(target in relevant for _score, target in ranked[:k])
            p_at[k].append(hits / k)
    pin_edge_count = len(design["pin_truth_pairs"])
    covered = sum(tuple(sorted((a[0], b[0]))) in set(candidate_pairs) for a, b in design["pin_truth_pairs"])
    accepted = sum(tuple(sorted((a[0], b[0]))) in accepted_pairs for a, b in design["pin_truth_pairs"])
    return {"pin_query_count": len(queries),
            "precision_at_1": float(np.mean(p_at[1])) if queries else None,
            "precision_at_3": float(np.mean(p_at[3])) if queries else None,
            "precision_at_5": float(np.mean(p_at[5])) if queries else None,
            "precision_at_1_sum": float(sum(p_at[1])),
            "precision_at_3_sum": float(sum(p_at[3])),
            "precision_at_5_sum": float(sum(p_at[5])),
            "end_to_end_pin_recall_candidate_pool": covered / max(1, pin_edge_count),
            "end_to_end_pin_recall_after_threshold": accepted / max(1, pin_edge_count),
            "covered_cross_component_pin_truth_edges": covered,
            "accepted_cross_component_pin_truth_edges": accepted,
            "cross_component_pin_truth_edges": pin_edge_count,
            "intra_component_pin_truth_edges_excluded": len(design["intra_component_pin_pairs"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--radius-mm", type=float, default=250.0)
    parser.add_argument("--k", type=int, default=96)
    parser.add_argument("--max-designs", type=int, default=0, help="debug only; 0 means all")
    args = parser.parse_args()
    designs = []
    for net_path in sorted(args.input.rglob("*.net")):
        try:
            design = parse_design(net_path, args.input)
        except (OSError, ValueError, IndexError, TypeError):
            design = None
        if design:
            designs.append(design)
        if args.max_designs and len(designs) >= args.max_designs:
            break
    if len(designs) < 3:
        raise SystemExit(f"Need at least 3 paired designs; parsed {len(designs)}")

    # Fold by repository/product family, so revisions of one board cannot leak.
    families = sorted({d["family"] for d in designs})
    fold_rows = []
    for design in designs:
        local, long_power, long_wire, dual = candidate_sets(design, args.radius_mm, args.k)
        positives = design["positive"]
        long_only = (long_power | long_wire) - local
        power_only = long_power - local
        wire_only = long_wire - local
        row = {"design_id": design["design_id"], "family": design["family"], "components": design["component_count"],
               "positive_edges": len(positives), "local_candidates": len(local), "local_hits": len(local & positives),
               "power_candidates": len(long_power), "wire_candidates": len(long_wire), "dual_candidates": len(dual),
               "dual_hits": len(dual & positives), "local_recall": len(local & positives) / max(1, len(positives)),
               "dual_recall": len(dual & positives) / max(1, len(positives)),
               "local_precision": len(local & positives) / max(1, len(local)),
               "dual_precision": len(dual & positives) / max(1, len(dual)),
               "long_only_candidates": len(long_only), "long_only_positive_hits": len(long_only & positives),
               "long_only_precision": len(long_only & positives) / max(1, len(long_only)),
               "power_only_candidates": len(power_only), "power_only_positive_hits": len(power_only & positives),
               "wire_only_candidates": len(wire_only), "wire_only_positive_hits": len(wire_only & positives),
               "power_all_candidates": len(long_power), "power_all_positive_hits": len(long_power & positives),
               "wire_all_candidates": len(long_wire), "wire_all_positive_hits": len(long_wire & positives)}
        fold_rows.append(row)
        design["pools"] = {"local": local, "power": long_power, "wire": long_wire, "dual": dual}

    # Grouped cross-validation for LightGBM. The wire feature is measured from
    # source schematic geometry and is reported as a CAD-source upper-bound cue.
    prediction_rows = {"local": [], "dual": []}
    for variant in ("local", "dual"):
        for heldout in families:
            train = [d for d in designs if d["family"] != heldout]
            test = [d for d in designs if d["family"] == heldout]
            if len({d["family"] for d in train}) < 3:
                continue
            calibrator, threshold, calibration_meta = grouped_oof_calibration(train, variant)
            x_train, y_train, _ = examples_for_designs(train, variant)
            if len(x_train) == 0 or len(np.unique(y_train)) < 2:
                continue
            model = new_lgbm()
            model.fit(x_train, y_train)
            for d in test:
                pairs = sorted(d["pools"][variant])
                x_test = np.asarray([candidate_row(d, p, d["pools"]["local"], d["pools"]["power"], d["pools"]["wire"])
                                     for p in pairs])
                raw_scores = model.predict_proba(x_test)[:, 1]
                scores = calibrator.predict(raw_scores) if calibrator is not None else raw_scores
                y = np.asarray([int(p in d["positive"]) for p in pairs])
                accepted = {pair for pair, score in zip(pairs, scores) if score >= threshold}
                pin_metrics = pin_query_metrics(d, pairs, scores, accepted)
                prediction_rows[variant].append({"design_id": d["design_id"], "family": heldout,
                                                 "threshold": threshold, "candidate_pairs": len(pairs),
                                                 "candidate_recall": int(y.sum()) / max(1, len(d["positive"])),
                                                 "candidate_pool_average_precision": float(average_precision_score(y, scores)) if len(set(y)) > 1 else None,
                                                 "candidate_pool_brier_score_calibrated": float(brier_score_loss(y, scores)) if len(set(y)) > 1 else None,
                                                 "calibration": calibration_meta,
                                                 "pin_metrics": pin_metrics,
                                                 **binary_metrics(y, scores >= threshold)})

    total_pos = sum(r["positive_edges"] for r in fold_rows)
    total_local = sum(r["local_candidates"] for r in fold_rows)
    total_dual = sum(r["dual_candidates"] for r in fold_rows)
    total_local_hit = sum(r["local_hits"] for r in fold_rows)
    total_dual_hit = sum(r["dual_hits"] for r in fold_rows)
    lgbm_summary = {}
    for variant in ("local", "dual"):
        rows = prediction_rows[variant]
        total_tp = sum(r["true_positive_edges"] for r in rows)
        total_fp = sum(r["false_positive_edges"] for r in rows)
        total_fn = sum(r["false_negative_edges"] for r in rows)
        pool_positive = sum(round(r["candidate_recall"] * next(d["positive_edges"] for d in fold_rows if d["design_id"] == r["design_id"])) for r in rows)
        pin_rows = [r["pin_metrics"] for r in rows]
        pin_truth_total = sum(r["cross_component_pin_truth_edges"] for r in pin_rows)
        pin_covered = sum(r["covered_cross_component_pin_truth_edges"] for r in pin_rows)
        pin_accepted = sum(r["accepted_cross_component_pin_truth_edges"] for r in pin_rows)
        pin_queries = sum(r["pin_query_count"] for r in pin_rows)
        intra_component_truth = sum(r["intra_component_pin_truth_edges_excluded"] for r in pin_rows)
        lgbm_summary[variant] = {
            "design_count": len(rows),
            "mean_precision": float(np.mean([r["precision"] for r in rows])) if rows else None,
            "mean_recall_within_candidate_pool": float(np.mean([r["recall"] for r in rows])) if rows else None,
            "mean_end_to_end_recall": float(np.mean([r["candidate_recall"] * r["recall"] for r in rows])) if rows else None,
            "mean_candidate_pool_average_precision": float(np.mean([r["candidate_pool_average_precision"] for r in rows if r["candidate_pool_average_precision"] is not None])) if rows else None,
            "mean_f1": float(np.mean([r["f1"] for r in rows])) if rows else None,
            "mean_calibrated_candidate_pool_brier_score": float(np.mean([r["candidate_pool_brier_score_calibrated"] for r in rows if r["candidate_pool_brier_score_calibrated"] is not None])) if rows else None,
            "mean_predicted_edges": float(np.mean([r["predicted_edges"] for r in rows])) if rows else None,
            "mean_candidate_recall_ceiling": float(np.mean([r["candidate_recall"] for r in rows])) if rows else None,
            "mean_validation_threshold": float(np.mean([r["threshold"] for r in rows])) if rows else None,
            "micro_precision": total_tp / max(1, total_tp + total_fp),
            "micro_recall_within_candidate_pool": total_tp / max(1, total_tp + total_fn),
            "micro_end_to_end_recall": total_tp / max(1, total_pos),
            "micro_candidate_recall_ceiling": pool_positive / max(1, total_pos),
            "pin_ranking": {
                "query_count": pin_queries,
                "precision_at_1_mean_over_pin_queries": sum(r["precision_at_1_sum"] for r in pin_rows) / max(1, pin_queries),
                "precision_at_3_mean_over_pin_queries": sum(r["precision_at_3_sum"] for r in pin_rows) / max(1, pin_queries),
                "precision_at_5_mean_over_pin_queries": sum(r["precision_at_5_sum"] for r in pin_rows) / max(1, pin_queries),
                "cross_component_pin_truth_edges": pin_truth_total,
                "end_to_end_pin_recall_candidate_pool": pin_covered / max(1, pin_truth_total),
                "end_to_end_pin_recall_after_threshold": pin_accepted / max(1, pin_truth_total),
                "intra_component_pin_truth_edges_excluded": intra_component_truth,
                "ranking_scope": "Per source pin, target pins are ranked by the predicted score of their parent component pair; pin coordinates are unavailable in these CAD exports."}}
    result = {
        "status": "COMPLETED_GROUPED_CAD_BENCHMARK" if prediction_rows["dual"] else "CANDIDATE_ONLY_NO_MODEL_FOLDS",
        "method": "Local cKDTree radius/k candidates; long-range NetworkX pin-role and source-wire/label candidates; deduplicate; grouped LightGBM link prediction.",
        "ground_truth": "Pairs of components that share one or more nets in the exported KiCad netlist.",
        "source_features": "Legacy schematic component centers, explicit wire/label geometry, and KiCad libpart pin names/types. The netlist is only used for targets, not candidate features.",
        "candidate_settings": {"radius_mm": args.radius_mm, "k": args.k, "long_range_paths": ["shared confirmed power pin function", "source schematic wire/label connected group"],
                               "power_role_degree_cap": 60, "wire_attach_radius_mil": 600},
        "split": {"type": "leave_one_repository_family_out", "families": families, "family_count": len(families), "design_count": len(designs)},
        "summary": {
            "positive_edges": total_pos,
            "intra_component_pin_truth_edges_excluded_from_component_pair_task": sum(len(d["intra_component_pin_pairs"]) for d in designs),
            "local": {"candidate_edges": total_local, "candidate_recall": total_local_hit / max(1, total_pos),
                      "candidate_precision": total_local_hit / max(1, total_local), "mean_candidates_per_design": total_local / len(designs)},
            "dual": {"candidate_edges": total_dual, "candidate_recall": total_dual_hit / max(1, total_pos),
                     "candidate_precision": total_dual_hit / max(1, total_dual), "mean_candidates_per_design": total_dual / len(designs)},
            "long_range_only": {
                "candidate_edges": sum(r["long_only_candidates"] for r in fold_rows),
                "positive_edges_recovered": sum(r["long_only_positive_hits"] for r in fold_rows),
                "candidate_precision": sum(r["long_only_positive_hits"] for r in fold_rows) / max(1, sum(r["long_only_candidates"] for r in fold_rows)),
                "incremental_positive_recall": sum(r["long_only_positive_hits"] for r in fold_rows) / max(1, total_pos),
                "mean_added_candidates_per_design": sum(r["long_only_candidates"] for r in fold_rows) / len(designs)},
            "long_range_paths": {
                "power_pattern_only": {
                    "all_candidates_before_local_dedup": sum(r["power_all_candidates"] for r in fold_rows),
                    "all_positive_edges_covered": sum(r["power_all_positive_hits"] for r in fold_rows),
                    "candidate_edges": sum(r["power_only_candidates"] for r in fold_rows),
                    "positive_edges_recovered": sum(r["power_only_positive_hits"] for r in fold_rows),
                    "candidate_precision": sum(r["power_only_positive_hits"] for r in fold_rows) / max(1, sum(r["power_only_candidates"] for r in fold_rows)),
                    "incremental_positive_recall": sum(r["power_only_positive_hits"] for r in fold_rows) / max(1, total_pos)},
                "wire_evidence_only": {
                    "all_candidates_before_local_dedup": sum(r["wire_all_candidates"] for r in fold_rows),
                    "all_positive_edges_covered": sum(r["wire_all_positive_hits"] for r in fold_rows),
                    "candidate_edges": sum(r["wire_only_candidates"] for r in fold_rows),
                    "positive_edges_recovered": sum(r["wire_only_positive_hits"] for r in fold_rows),
                    "candidate_precision": sum(r["wire_only_positive_hits"] for r in fold_rows) / max(1, sum(r["wire_only_candidates"] for r in fold_rows)),
                    "incremental_positive_recall": sum(r["wire_only_positive_hits"] for r in fold_rows) / max(1, total_pos)}},
            "lgbm_holdout": lgbm_summary,
        },
        "per_design": fold_rows,
        "lgbm_per_design": prediction_rows,
        "limitations": [
            "The raw dataset contains exported CAD schematics rather than noisy image-extracted pins; wire evidence is an optimistic source-CAD cue.",
            "Component centers stand in for pin locations in the cKDTree baseline because this legacy source format has no exported placed-pin coordinates.",
            "Netlist-shared component pairs define labels; multi-pin nets can create many valid positive pairs.",
            "This is an offline retrieval/ranking benchmark, not a replacement for CircuitCare graph behavior."
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "design_count": len(designs), "family_count": len(families),
                      "summary": result["summary"], "output": str(args.output.resolve())}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
