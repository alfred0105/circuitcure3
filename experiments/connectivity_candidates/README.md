# Dual-path connectivity candidate experiment

This offline experiment compares local-neighbor retrieval (250 mm, up to 96
neighbors) against its union with long-range candidate paths. Local search uses
`scipy.spatial.cKDTree`; `NetworkX` builds shared electrical-role and
schematic-wire/label candidate graphs; LightGBM scores the deduplicated pool.

## Inputs and evaluation scope

- 41 paired legacy KiCad schematics, exported netlists, and PCB files.
- 11 repository/product families. Leave-one-family-out evaluation keeps
  revisions from each repository together across train and test.
- A positive component-pair label means both components occur on at least one
  common net in the exported netlist. Netlist membership is used for targets,
  not candidate features.
- Same-component pin pairs are outside this component-pair task and are counted
  separately instead of disappearing from the audit.
- Candidate features include distance, direction, component pin counts, shared
  pin-name features, explicit power-pin roles, and schematic wire/label/power
  symbol geometry. Candidate edges are deduplicated.
- LightGBM uses class-balanced fitting, grouped out-of-fold isotonic probability
  calibration and a separate grouped OOF threshold-validation split. Test-family
  labels do not set the threshold or calibration map.
- No negative sampling was used; class balancing is applied during fitting only.

## Candidate-generation results

The local cKDTree pool produced 120,095 candidates with 69.02% pooled recall and
25.91% precision. Adding long-range paths produced 122,156 candidates with
72.86% recall and 26.89% precision. The additional 2,061 candidates included
1,731 positive connections (83.99% precision), recovering 3.84 percentage points
of all positive component pairs for a 1.72% increase in candidate count.

Most incremental recall came from wire, label, and power-symbol geometry:
1,747 incremental candidates recovered 1,429 positive pairs (81.80% precision,
3.17 points of recall). Shared power-pin-function candidates added 314
candidates and recovered 302 positives (96.18% precision, 0.67 points).

## LightGBM results

Mean per-board **candidate-pool** average precision under family-held-out
evaluation rose from 0.716 to 0.735. This ranking metric only contains generated
candidates; it does not include missed connections. At separate grouped OOF
thresholds, pooled end-to-end component-pair recall (including candidate misses)
rose from 39.88% to 42.49%, and pooled precision rose from 64.21% to 68.20%.
Class-balanced model scores were isotonic-calibrated on OOF families, and the
F1 threshold was selected on disjoint OOF families. Mean selected thresholds
were 0.324 for local and 0.344 for dual-path; these are validation-selected
operating points, not a universal 0.5 probability cutoff. Mean calibrated
Brier score on the candidate pool decreased from 0.138 to 0.135.

The held-out designs contain 66,282 cross-component pin-pair truths across
11,431 source-pin queries. Expanding each predicted component-pair score to its
target pins yields query-weighted Pin Precision@1/3/5 of
14.63%/12.24%/11.01% for local-only and 14.58%/12.32%/11.24% for dual-path.
These P@k rankings are over target pins whose parent component pair was
generated as a candidate. The changes are small and mixed. Candidate-pool
end-to-end pin recall rises
from 70.45% to 74.14%; after the validation-selected threshold it rises from
46.98% to 49.66%. This is a diagnostic ranking measure: all target pins of a
component pair inherit the same parent-pair score, and ties are resolved by
stable reference/pin ordering. It is not a pin-coordinate classifier. The source
designs contain 931 same-component pin-pair truths; these are reported
separately and excluded from the cross-component candidate denominator.

## Limitations

The files contain source CAD rather than noisy image-extracted pins. Component
centers stand in for pin locations because exact placed-pin coordinates were
not exported. Wire geometry is therefore an optimistic evidence source. Pin
Precision@k expands component-pair scores to pins and cannot determine which
specific pin is connected. High-degree power nets create many positive pairs.
These figures are CAD-source retrieval results, not production image-reconstruction
accuracy.

Focused synthetic and parser tests are in `tests/test_connectivity_candidates.py`.
The `connectivity-validation` GitHub Actions job installs the optional
`connectivity` dependencies and runs those tests on Python 3.11 and 3.12. CI does
not run the full benchmark because its external CAD corpus is not in the
repository checkout.

## Reproduce

```bash
./.venv-component-train/bin/python experiments/connectivity_candidates/benchmark_candidate_generation.py
```

The script writes per-board and per-family metrics to
`experiments/connectivity_candidates/results/dual_path_benchmark.json`.
