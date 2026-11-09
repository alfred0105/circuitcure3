from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/connectivity_candidates"))

from benchmark_candidate_generation import (  # noqa: E402
    candidate_sets,
    legacy_positions,
    parse_design,
    pin_query_metrics,
    power_tokens,
)


def test_power_roles_recognize_ground_and_supply_names():
    assert power_tokens([{"name": "GND"}, {"name": "VDD"}, {"name": "SDA"}]) == {
        "power:gnd", "power:VDD"
    }


def test_candidate_union_deduplicates_local_power_and_wire_pairs():
    design = {
        "positions": {"A": (0, 0), "B": (100, 0), "C": (10000, 0)},
        "comp_pins": {
            "A": [{"name": "GND", "type": "power_in"}],
            "B": [{"name": "GND", "type": "power_in"}],
            "C": [{"name": "SIG", "type": "input"}],
        },
        "wire_pairs": {("A", "B"), ("A", "C")},
    }
    local, power, wire, dual = candidate_sets(design, radius_mm=250, k=96)
    assert ("A", "B") in local and ("A", "B") in power and ("A", "B") in wire
    assert len(dual) == len(local | power | wire)
    assert all(a != b for a, b in dual)


def test_parse_design_reports_intra_component_pin_edges_separately(tmp_path):
    net = tmp_path / "demo.net"
    net.write_text('''(export
      (components
        (comp (ref U1) (value MCU) (libsource (lib Demo) (part MCU)))
        (comp (ref R1) (value 1k) (libsource (lib Demo) (part R)))
      )
      (libparts
        (libpart (lib Demo) (part MCU) (pins
          (pin (num 1) (name GND) (type power_in))
          (pin (num 2) (name GPIO) (type bidirectional))
          (pin (num 3) (name VDD) (type power_in))))
        (libpart (lib Demo) (part R) (pins
          (pin (num 1) (name A) (type passive))
          (pin (num 2) (name B) (type passive))))
      )
      (nets
        (net (code 1) (name GND)
          (node (ref U1) (pin 1)) (node (ref R1) (pin 1)))
        (net (code 2) (name N1)
          (node (ref U1) (pin 2)) (node (ref U1) (pin 3)) (node (ref R1) (pin 2)))
      ))''', encoding="utf-8")
    net.with_suffix(".kicad_pcb").write_text("fixture", encoding="utf-8")
    net.with_suffix(".sch").write_text(textwrap.dedent('''
        EESchema Schematic File Version 4
        $Comp
        L Demo:MCU U1
        U 1 1 aaa
        P 1000 1000
        F 0 "U1" H 0 0 50 0000 C CNN
        F 1 "MCU" H 0 0 50 0000 C CNN
        \t1 1000 1000
        $EndComp
        $Comp
        L Demo:R R1
        U 1 1 bbb
        P 1500 1000
        F 0 "R1" H 0 0 50 0000 C CNN
        F 1 "1k" H 0 0 50 0000 C CNN
        \t1 1500 1000
        $EndComp
        '''), encoding="utf-8")

    design = parse_design(net, tmp_path)
    assert design is not None
    assert design["positive"] == {("R1", "U1")}
    assert len(design["pin_truth_pairs"]) == 3
    assert len(design["intra_component_pin_pairs"]) == 1
    assert design["pin_truth_neighbors"][("U1", "2")] == {("R1", "2")}


def test_pin_precision_at_k_and_end_to_end_pin_recall_are_reported():
    design = {
        "all_pins": {"A": [("1", "OUT", "output")],
                     "B": [("1", "NC", "no_connect"), ("2", "IN", "input")]},
        "pin_truth_neighbors": {("A", "1"): {("B", "2")}},
        "pin_truth_pairs": {(('A', '1'), ('B', '2'))},
        "intra_component_pin_pairs": {(('A', '2'), ('A', '3'))},
    }
    pairs = [("A", "B")]
    scores = [0.8]
    report = pin_query_metrics(design, pairs, scores, set(pairs))
    assert report["pin_query_count"] == 1
    assert report["precision_at_1"] == 0
    assert report["precision_at_3"] == 1 / 3
    assert report["precision_at_5"] == 1 / 5
    assert report["end_to_end_pin_recall_candidate_pool"] == 1
    assert report["end_to_end_pin_recall_after_threshold"] == 1
    assert report["intra_component_pin_truth_edges_excluded"] == 1
