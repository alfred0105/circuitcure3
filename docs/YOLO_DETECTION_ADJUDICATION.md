# YOLO detection correctness labels

`adjudicate_yolo_against_cad.py` compares YOLO boxes with CAD component locations for a curated single-board photo/CAD package. It estimates a photo-to-CAD-render homography automatically, checks inlier support and geometric plausibility, then labels each prediction as:

- `CORRECT_LOCATION_AND_CLASS`
- `WRONG_CLASS_AT_COMPONENT_LOCATION`
- `DUPLICATE_DETECTION_AT_COMPONENT_LOCATION`
- `FALSE_POSITIVE_CANDIDATE`
- `UNRESOLVED_REGISTRATION`

Unmatched CAD components are recorded as `MISSED_COMPONENT_CANDIDATE`. The comparison uses the prediction-box center against an approximate projected footprint polygon with a 4 px margin. It judges component location and mapped class, not box-edge quality or mAP.

Run the automatic registration check without predictions:

```bash
.venv-component-train/bin/python scripts/adjudicate_yolo_against_cad.py \
  --pair-manifest data/component_training/processed/pcb_photo_cad_pairs/handy-robotic-hand/pair_manifest.json \
  --output data/component_training/processed/pcb_photo_cad_pairs/handy-robotic-hand/automated_adjudication/registration.json
```

Then provide YOLO output JSON. A single-image detector JSON can be joined by its image SHA-256 or an explicit `--image-id`; a multi-image `predictions` object can use the pair manifest's image ID, path, filename, or SHA-256.

```bash
.venv-component-train/bin/python scripts/adjudicate_yolo_against_cad.py \
  --pair-manifest data/component_training/processed/pcb_photo_cad_pairs/handy-robotic-hand/pair_manifest.json \
  --predictions path/to/pcb_ai_detection.json \
  --output data/component_training/processed/pcb_photo_cad_pairs/handy-robotic-hand/automated_adjudication/yolo-labels.json
```

## Interpretation limits

These are CAD-reference labels, not independently reviewed visual ground truth. The matching rule uses the component center and an approximate body outline inferred from the footprint name; visibility, occlusion, manufacturing substitutions, assembly differences, or incomplete source BOMs can make a CAD candidate differ from the photo. `FALSE_POSITIVE_CANDIDATE` and `MISSED_COMPONENT_CANDIDATE` must not be treated as confirmed errors for model training or official metrics.

If registration fails or front/back is ambiguous, the script labels predictions `UNRESOLVED_REGISTRATION` and does not assign correctness. It never falls back to filename similarity. Current automatic run: Handy front photo reached a registration candidate and projected 10 supported component references; Handy back and all three Storm Tracker photos remain unresolved. No YOLO predictions were adjudicated in that run.

The collected catalog still has no verified image-space component-box ground-truth set. For benchmark mAP, use a separately reviewed annotation manifest; the CAD candidate labels can help focus that process but are not a substitute for it.

## Generate CAD-projected YOLO label candidates

The generator writes projected CAD footprint boxes in YOLO text format and a provenance/completeness manifest. It keeps unresolved photos out of the generated image set and marks every image ineligible for training until visible-component coverage is known.

```bash
.venv-component-train/bin/python scripts/generate_cad_candidate_yolo_labels.py \
  --pair-manifest data/component_training/processed/pcb_photo_cad_pairs/handy-robotic-hand/pair_manifest.json \
  --output-dir data/component_training/processed/pcb_photo_cad_pairs/handy-robotic-hand/cad_label_candidates
```

The current automatic run produced one photo registration candidate and seven supported connector boxes for the Handy front photo. The remaining CAD entries include unknown/unmapped components, and photo-level visibility/completeness is unverified, so this output is a partial candidate set and must not be used to compute benchmark metrics or train as complete labels. Storm Tracker photos did not pass registration; OMI images remain blocked because the individual photo-to-board mapping is ambiguous.
