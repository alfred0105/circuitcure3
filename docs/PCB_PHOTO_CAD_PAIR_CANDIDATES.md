# PCB photo/CAD pair candidates

## Intake summary

The complete core catalog contains 132 product/project candidates across 118 discovered GitHub repositories. All repositories were cloned or linked to an existing local clone, and all indexed source files were fetched. The 132 entries are candidate records, not 132 verified distinct physical boards.

The complete per-candidate file map is in [`pcb-photo-cad-training-source-manifest.json`](../data/component_training/manifests/pcb-photo-cad-training-source-manifest.json). It records original links, source commit, local path, SHA-256, file role, download state, and whether the photo-to-board/revision association has been verified. Collection totals and the boundary from earlier datasets are recorded in [`phase21-3-2-g-dataset-selection.json`](../data/component_training/manifests/phase21-3-2-g-dataset-selection.json). Earlier datasets were not moved, changed, deleted, or mixed into this collection.

Collected in the new raw source area:

- 4,840 GitHub repository source files from 118 repositories, including 1,992 image candidates, 633 separately marked render/diagram images, and 2,215 CAD/circuit/manufacturing files.
- 2,409 manufacturer/product-page image files from 62 of 67 pages; five pages exposed no downloadable images in their HTML.
- 69 direct CAD/schematic download links, all fetched, with 61 CAD/manufacturing files extracted from their archives.
- 242 ZIP archives discovered in GitHub repository CAD/manufacturing paths; 237 contained relevant files, and 1,620 CAD/manufacturing members were extracted while preserving the original archives.
- Twelve additional repositories were discovered through the product pages and added to the download set. Filename and archive rules were expanded to catch board photos, Eagle CAD files, and Gerber archives.

At the item level, 117 of 132 candidates currently have a downloaded photo/image candidate and 130 have a CAD or source document; 115 have both kinds of source file. The remaining 15 have no downloaded photo/image candidate, and two have no downloaded CAD/source document. Their catalog rows and source links remain in the manifest so they are visible for the user's own organization.

Image counts include unverified candidates. No manual image inspection was performed, as requested; the manifest marks product-page images as unclassified and repository image candidates as unverified. Some candidates may be renders, diagrams, assemblies, or repeated product-gallery views. Repository/product name agreement does not prove that the photographed revision matches a particular CAD file.

The supplied catalog's eight supplementary dataset/design sources and eleven exploration/platform leads are separately indexed in [`pcb-photo-cad-supplementary-sources.json`](../data/component_training/manifests/pcb-photo-cad-supplementary-sources.json), following the catalog's instruction to keep them separate from the 132 core photo/CAD candidates. Earlier data manifests remain preserved as well.

The collected materials are source candidates, not detector-ready labels. No candidate is enabled for object-detection training until image-space component boxes and board/revision associations are established automatically with adequate evidence.

Two repositories document a single-board photo set: Storm Tracker Rev A (3 photos) and Handy Robotic Hand (2 photos). FEMTO (18), PicoBerry (8), Pixie (2), and Omi Consumer (4) remain photo-to-board/revision candidates with their ambiguity recorded. Render-like paths are marked out of camera-photo training based on filenames and folders. No manual image inspection was used.

Prepared outputs:

- Storm Tracker: [`pair_manifest.json`](../data/component_training/processed/pcb_photo_cad_pairs/storm-tracker-rev-a/pair_manifest.json), [`components.csv`](../data/component_training/processed/pcb_photo_cad_pairs/storm-tracker-rev-a/components.csv), KiCad schematic netlist, and CAD-side renders.
- Handy: [`pair_manifest.json`](../data/component_training/processed/pcb_photo_cad_pairs/handy-robotic-hand/pair_manifest.json), PCB-derived IPC-2581 and placement coordinates. The repository has no schematic source.
- Omi Consumer: [`pair_manifest.json`](../data/component_training/processed/pcb_photo_cad_pairs/omi-consumer/pair_manifest.json) with three extracted KiCad PCB/schematic sets, three exported netlists, IPC-2581 and placement data, and four assembly photos. Images remain linked to the three-board product family, not arbitrarily assigned to one board.

CAD-to-image feature registration was attempted automatically for the curated single-board photo sets. The current strict registration pass accepts a candidate transform for the Handy front photo (52 top-side inliers; source metadata identifies the front), then projects seven confidently classified connector footprints. The Handy back and all three Storm Tracker photos remain unresolved. This yields a partial CAD-derived label candidate, not complete image ground truth: other CAD entries are unmapped and per-photo visible-component completeness is unknown. The board identity pairings and electrical CAD exports are saved, but no photo is training-ready yet.

## Storm Tracker Rev A — first source-declared match

Curated locally from [`solderable/storm-tracker-hardware`](https://github.com/solderable/storm-tracker-hardware), source commit `1b19f377725d68d4e05cae1bb77ca1c6f572141e`.

The repository README explicitly says the editable Rev A design is the revision shown in its photos. The local candidate package contains three repository photos, the KiCad PCB and schematic, a KiCad-exported XML netlist, top/bottom CAD renders, the BOM, and the Rev A pick-and-place coordinate file. The component table joins 91 placed reference designators to BOM categories and values.

| Item | Status |
|---|---|
| Source-declared photo ↔ Rev A identity | Recorded for 3 photos |
| PCB ↔ schematic | Linked to `storm_tracker.kicad_pcb` and `storm_tracker.kicad_sch` from the same source commit |
| Component coordinate/value references | 91 placement rows joined to BOM |
| KiCad-derived circuit connectivity | Exported to `cad/storm_tracker.net.xml` |
| Automated photo ↔ render registration | Tried on both sides; fewer than 20 RANSAC inliers for each photo, so no transform accepted |
| Per-image visible-instance confirmation | Not performed |
| Internal training use | User confirmed selected sources are cleared for internal use |
| Photo pixel boxes / homography | Not generated; CAD coordinates have not been projected into photos |
| Training-ready for object detection | **Not yet** — image-space labels require photo-to-CAD registration |

Local files are under the ignored data directory `data/component_training/processed/pcb_photo_cad_pairs/storm-tracker-rev-a/`. The original cloned source is preserved at `data/component_training/raw/external/pcb_photo_cad_pairs/storm-tracker-hardware/`. `pair_manifest.json` records each photo's hash and board association; `components.csv` contains CAD positions, footprint/value, side, and BOM category. Coordinates remain in board millimeters and must not be treated as image-space object boxes.

The curation command is:

```bash
.venv-component-train/bin/python scripts/curate_storm_tracker_cad_pair.py
```

The full catalog collection is stored locally under `data/component_training/raw/external/pcb_photo_cad_pairs/`; extracted/curated outputs are under `data/component_training/processed/pcb_photo_cad_pairs/`. The source index keeps photo and CAD candidates separate where board or revision mapping remains uncertain. The user confirmed internal use for the selected sources; the source catalog's redistribution cautions still apply.
