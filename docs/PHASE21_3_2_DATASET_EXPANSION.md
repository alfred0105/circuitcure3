# Phase 21.3.2 — dataset expansion and provenance

Last checked: 2026-11-06. No new images or annotations were downloaded into the repository. Remote published counts below are source-reported, not counts acquired and audited locally.

| Source | Published scale / labels | Access check | License / use status | Decision |
|---|---|---|---|---|
| CircuitCare pinned dataset `wangshnegqi/pcb-component-detection-dataset-yolo` | Downloaded pinned GitHub commit `0d7ae7310fa607389779b6132285336ad73e96e1`; dataset has 117 train images / 3,104 boxes and 13 val images / 196 boxes. Three augmentation variants per some originals result in 84 train filename groups. | GitHub source clone succeeded; Hugging Face API returned 401. | `LICENSE-DATASET` explicitly grants CC BY 4.0; model weights are separately declared Apache-2.0. Dataset clone measured 282 MiB including source and data. | Acquired and used for a 5-epoch provisional fine-tune. Board IDs remain unresolved and split overlap has not been verified at physical-board level. |
| FICS-PCB (2020/366) | Publication describes a 9,912-image multisensor PCB dataset; component/defect modalities require source inspection. | Trust-Hub access is not an anonymous direct download in this environment. | Must check the current Trust-Hub terms at retrieval; not confirmed here. | Hold. Do not count as acquired. |
| PCB-WACV 2019 | Author page reports 47 high-resolution images, 31 types, about 62k instances; published ZIP linked (287 MB). | Public download URL is exposed by author site; payload not fetched in this run. | Author page does not state a dataset license. | Hold pending terms. |
| FPIC | Official PhysicalDB description says 261 high-resolution images of 93 PCBs, >71k instances including text and components. | Download button is present, but no direct anonymous URL surfaced in page content. | Official page does not state dataset license. FPIC component derivative sources report CC BY-NC-ND 4.0, which is not approved for model training without further review. | Hold; do not infer that source and derivative share terms. |
| PCB-DSLR (Zenodo 3886553) | Zenodo metadata reports 748 DSLR images from 165 boards, 9,313 labeled IC instances; 1,740 have label information. Eight archives total about 6.2 GB. | Metadata API is public; archives are directly downloadable. Not downloaded due size and terms review. | Zenodo record metadata says CC BY 4.0, while the record description says free for non-commercial research. This conflict requires clarification; IC-only annotations do not improve resistor/capacitor labels. | Hold; not counted as acquired or trainable. |

Sources: [pinned training dataset commit](https://github.com/wangshnegqi/pcb-component-detection-dataset-yolo/tree/0d7ae7310fa607389779b6132285336ad73e96e1), [dataset CC BY 4.0 terms](https://github.com/wangshnegqi/pcb-component-detection-dataset-yolo/blob/0d7ae7310fa607389779b6132285336ad73e96e1/LICENSE-DATASET), [CircuitCare detector provenance](PCB_AI_DETECTION.md), [FICS-PCB paper](https://eprint.iacr.org/2020/366), [WACV author page](https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection), [FPIC official dataset page](https://physicaldb.ece.ufl.edu/index.php/fics-pcb-image-collection-fpic/), [FPIC component license metadata](https://datasetninja.com/fpic-component), [PCB-DSLR Zenodo record and file metadata](https://zenodo.org/records/3886553).

## Present local data state

The raw pinned source is present under ignored `data/component_training/raw/source-pinned/`; do not commit it. Exact SHA-256 image overlap between the train and val file lists is zero. The source split has no board IDs, cannot establish board independence, and validation was already used during development. It is neither an independent held-out test set nor a basis for generalization claims. Across train and val, there are 130 files, 3,300 annotations, and 10 source classes. Source `train` includes generated augmentations; raw views should not be counted as 117 independent boards.

### Domain filtering result

The complete source was reviewed visually and recorded in `data/component_training/manifests/pinned-source-domain-review.json`. Original views received a human contact-sheet review; augmentations inherit the reviewed parent category. The eight image types are `POPULATED_PCB`, `PARTIALLY_POPULATED_PCB`, `PCB_COMPONENT_CLOSEUP`, `BARE_PCB`, `BREADBOARD`, `ISOLATED_COMPONENT`, `LOOSE_COMPONENT_COLLECTION`, and `UNKNOWN`.

| Measure | Before filtering | Main train + validation | Excluded from main set |
|---|---:|---:|---:|
| Images | 130 | 101 (77 train, 24 validation) | 29 |
| Manually reviewed visual board groups | 34 | 33 (29 train, 4 validation) | — |

Included image types: 76 populated PCB, 12 partially populated PCB, 13 PCB component closeups. Excluded: 19 loose component collections and 10 breadboards. No bare PCB, isolated component, or unknown images were found. The 10 breadboards are human-reviewed and explicitly confirmed as containing no target PCB components, so they are stored separately for false-positive evaluation (not ordinary training). Empty annotations by themselves do not mark images as empty. No false exclusion of a populated PCB or false inclusion of an isolated product photo was identified in contact-sheet review; 13 rendered/cutout-style views were included only after visual confirmation that PCB-mounted components were visible, with medium confidence and a style caveat. This is a human visual audit, not a pixel-level annotation audit.

The dataset has zero source-verified physical board IDs. The “board” numbers above are manual visual capture groups; do not describe them as actual unique PCB counts. Board-group leakage check passes across the 33 included groups. The 24-image validation set has 705 source-provided boxes, but labels have not been independently adjudicated and there is no sealed test set. See `PHASE21_3_2_MODEL_BENCHMARK.md` for separate positive-PCB metrics and breadboard FPR / mean false positives per image.

## Curation rules before training

Do not mix partially labeled images into ordinary YOLO object-detection training: unlabeled real parts become false background. Retain source class and canonical class separately; map `integrated_circuit` to `ic` or existing class contract only with an explicit mapping. Keep original board identity and source split in a manifest. Group all same-board sides, captures, crops, and augmentations together. Exact SHA-256 duplicates must be removed and perceptual near-duplicates reviewed. Split by physical board before augmentation. Any board with unresolved identity is excluded from locked test claims.

Suggested target classes should first respect the existing deployed ten-class contract. New component classes require re-evaluation of the ONNX output contract and existing caller mappings; adding broad classes such as mosfet or fuse is not a simple class-name rename. The lab config currently matches the deployment detector's ten classes.

## Phase 21.3.2-B 후속 수집

실제 공개 데이터 다운로드, 라이선스/access audit, image-by-image manifest, source class mapping 및 loader/ONNX smoke 결과는 [Phase 21.3.2-B dataset acquisition report](PHASE21_3_2_B_DATASET_ACQUISITION.md)를 참조한다. 신규 source data는 현재 학습 준비 상태가 아니며, 이 보고서의 `DATASET_EXPANSION_BLOCKED` 판정을 유지한다.
