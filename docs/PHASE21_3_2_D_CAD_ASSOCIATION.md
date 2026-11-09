# Phase 21.3.2-D — CAD 파일 연계 확인

Read-only extension scan covered the repository and accessible `Documents`, `Desktop`, and `Downloads` directories. It found 16 `.kicad_pcb` files and 29 `.kicad_sch` files, 56 netlist-named/XML candidates, and no Gerber, BOM, or Pick-and-Place files. Counts include test fixtures, KiCad examples/history and repeated working/candidate/recheck files.

Some files belong to a separate PRISM circuit CAD dataset. Another separate artwork checkpoint contains a Korean-named KiCad board and schematic, including candidate/recheck copies. No file path, PCB-DSLR `pcb<ID>/rec<N>` identifier, Kaggle image filename, source checksum, BOM reference, or other stable key ties these CAD projects to the image/annotation records used in this Phase.

No CAD-to-photo association was made and no CAD file was used to create or modify an Annotation. A CAD footprint coordinate is not an image bounding box: a verified image-to-board registration, side/orientation match and component correspondence would be required before using it as supervision. The files were only inventoried; they were not opened in an editor or manually matched.
