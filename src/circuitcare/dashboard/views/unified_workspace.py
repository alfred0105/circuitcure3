"""PCB-first F2 workbench shell with real intake images and an evidence-neutral panel."""

from __future__ import annotations

import hashlib
import base64
import io
import json
from dataclasses import replace
from pathlib import Path
import tempfile
from typing import Any

from circuitcare.frontend.models import ProjectContext
from circuitcare.symptom_diagnosis.diagnosis_state import (
    component_inventory_version,
)
from circuitcare.frontend.component_overlay_adapter import (
    adapt_component_map_components, adapt_original_photo_components,
)
from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.frontend.board_component_registry import (
    REGISTRY_KEY, build_board_component_registry, confirm_component_match,
    diagnosis_components, measurement_component_aliases, remove_component_match,
    source_key_for_component,
)
from circuitcare.frontend.measurement_state import get_measurements
from circuitcare.analysis.models import AnalysisScope, AnalysisStatus
from circuitcare.frontend.component_map_adapter import (
    COMPONENT_MAP_STATE_KEY, LOGGER, REGISTRATION_LABELS,
    clear_component_map_state,
    component_map_cache_key, component_map_entry_is_current, decode_canonical_image,
    map_backend_photo_ids, optional_backend_provider, photo_input_fingerprint,
    validate_component_map,
)
from circuitcare.frontend.photo_adapter import adapt_component_map_input
from circuitcare.dashboard.components.pcb_workspace import (
    OVERLAPPING_SELECTION_KEY, _record_selection_trace,
    render_component_viewer, scale_overlay_to_display,
)
from circuitcare.dashboard.views.inspection_panel import (
    COMPONENT_EDIT_KEY, COMPONENT_INVENTORY_KEY, MAP_VERSION_KEY,
    board_suspects, render_inspection_panel,
    _diagnosis_status_label,
)
from circuitcare.dashboard.views.symptom_diagnosis_panel import (
    candidate_rows_for_strategy, current_analysis_snapshot, render_symptom_diagnosis_panel,
)
from circuitcare.dashboard.views.measurement_panel import render_measurement_panel
from circuitcare.dashboard.views.diagnosis_panel import render_diagnosis_panel
from circuitcare.dashboard.views.report_panel import render_report_panel
from circuitcare.reporting.report_builder import build_pcb_diagnostic_snapshot
from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.vision.component_recognition_panel import (
    ensure_component_detection, render_component_recognition_panel,
    review_keys, review_scope as component_review_scope,
)
from circuitcare.vision.box_reconciliation import board_visible_review_rows, same_component_box_pair
from circuitcare.frontend.state import (
    INTAKE_IMAGES_KEY, INTAKE_PROJECT_KEY, clear_intake_draft,
    reset_intake_for_new_project,
)
from circuitcare.frontend.workbench_state import (
    ACTIVE_PHOTO_KEY, BOARD_COMPONENT_SELECTION_KEY, MODE_KEY, SCREEN_KEY, SELECTION_KEY, VIEW_ZOOM_KEY,
    VIEW_TRANSFORM_KEY, enter_workbench, update_selection,
)
from circuitcare.frontend.viewer_geometry import requested_view_transform
from circuitcare.schematic.exporter import export_schematic
from circuitcare.schematic.geometry import read_svg_view_box
from circuitcare.schematic.viewer import render_interactive_viewer
from circuitcare.vision.image_loader import ImageLoadError, load_board_image


def _style(st: Any) -> None:
    st.markdown("""<style>
    [data-testid="stMainBlockContainer"] { max-width:none; padding:1rem 1.5rem 2rem; }
    .cc-wb-header { display:flex; align-items:center; gap:.6rem; color:#e9eef4; font-weight:700; font-size:1.1rem; }
    .cc-wb-subtle { color:#9aa9b8; font-size:.82rem; }
    .cc-analysis-hero { background:linear-gradient(135deg,#172536,#111923); border:1px solid #34475c; border-radius:12px; padding:1rem 1.15rem; margin:.25rem 0 1rem; }
    .cc-analysis-step { color:#7fb3ef; font-size:.78rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase; }
    .cc-analysis-title { color:#edf3f9; font-size:1.15rem; font-weight:700; margin:.2rem 0 .45rem; }
    .cc-analysis-copy { color:#aab8c7; font-size:.9rem; }
    .cc-analysis-section { border-top:1px solid #334252; margin-top:1.1rem; padding-top:.75rem; }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.cc-inspection-marker) { position:sticky; top:1.25rem; align-self:start; }
    [class*="st-key-cc_viewer_canvas"] { overflow:auto; max-height:calc(100vh - 13rem); background:#101820; border:1px solid #334252; border-radius:8px; padding:.5rem; }
    div[role="radiogroup"] label { border:1px solid #334252; border-radius:6px; padding:.35rem .8rem; }
    @media (max-width:900px) { [data-testid="stMainBlockContainer"] { padding:.8rem; } }
    @media (max-width:900px) {
      .st-key-cc_workbench_body [data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
      .st-key-cc_workbench_body [data-testid="column"] { flex:1 1 100%; width:100%; min-width:100%; }
      [data-testid="stVerticalBlockBorderWrapper"]:has(.cc-inspection-marker) { position:static; }
    }
    @media (max-width:700px) { [class*="st-key-cc_viewer_canvas"] { max-height:60vh; } }
    </style>""", unsafe_allow_html=True)


def _bind_review_models_to_map(
    map_models: tuple[ComponentOverlayViewModel, ...],
    review_models: tuple[ComponentOverlayViewModel, ...],
    review_rows: list[dict[str, Any]],
) -> tuple[tuple[ComponentOverlayViewModel, ...], list[dict[str, Any]]]:
    """Use review rows as canonical boxes; only attach map evidence on certain IDs."""
    review_by_id = {str(row.get("component_id")): row for row in review_rows if row.get("component_id")}
    auto_by_id = {item.component_id: item for item in review_models}
    consumed_map_ids: set[str] = set()
    ambiguous_map_ids: set[str] = set()
    handled_review_ids: set[str] = set()
    rendered: list[ComponentOverlayViewModel] = []

    # Stable aliases are authoritative. Geometry is only a fallback and must be
    # unique in both directions; choosing the highest-confidence nearby map box
    # silently attached evidence to the wrong physical component.
    map_by_id = {item.component_id: item for item in map_models}
    matches_by_row: dict[str, list[str]] = {}
    rows_by_map: dict[str, list[str]] = {}
    for row_id, row in review_by_id.items():
        box = [row.get(key) for key in ("x1", "y1", "x2", "y2")]
        if not all(isinstance(value, (int, float)) for value in box):
            continue
        aliases = row.get("associated_component_ids", [])
        if not isinstance(aliases, (list, tuple, set)):
            aliases = [aliases]
        aliases = [str(item) for item in aliases if item]
        exact = [item for item in aliases if item in map_by_id]
        if exact:
            matches_by_row[row_id] = exact[:1]
            rows_by_map.setdefault(exact[0], []).append(row_id)
            continue
        matches = [item.component_id for item in map_models
                   if same_component_box_pair(box, item.bbox_xyxy)]
        matches_by_row[row_id] = matches
        for map_id in matches:
            rows_by_map.setdefault(map_id, []).append(row_id)

    certain_map_by_row: dict[str, str] = {}
    for row_id, map_ids in matches_by_row.items():
        # Require one candidate and a reciprocal one-row match, even for stored
        # aliases, so duplicated or conflicting review rows remain unbound.
        if len(map_ids) == 1 and len(rows_by_map.get(map_ids[0], [])) == 1:
            certain_map_by_row[row_id] = map_ids[0]
        elif map_ids:
            ambiguous_map_ids.update(map_ids)

    for row_id, row in review_by_id.items():
        box = [row.get(key) for key in ("x1", "y1", "x2", "y2")]
        if not all(isinstance(value, (int, float)) for value in box):
            continue
        handled_review_ids.add(row_id)
        map_id = certain_map_by_row.get(row_id)
        map_match = map_by_id.get(map_id) if map_id else None
        if map_match is not None:
            consumed_map_ids.add(map_match.component_id)
            aliases = row.get("associated_component_ids", [])
            if not isinstance(aliases, (list, tuple, set)):
                aliases = [aliases]
            if map_match.component_id not in aliases:
                row["associated_component_ids"] = [*aliases, map_match.component_id]
            row.pop("map_match_status", None)
        elif matches_by_row.get(row_id):
            row["map_match_status"] = "AMBIGUOUS_GEOMETRY"
        if row.get("상태") == "제외":
            continue
        auto = auto_by_id.get(row_id)
        if auto is None:
            continue
        if map_match is not None:
            auto = replace(auto,
                source_backend_photo_ids=map_match.source_backend_photo_ids,
                observation_ids=map_match.observation_ids,
                identity_match_status=map_match.identity_match_status,
                fault_history=tuple((*auto.fault_history, *map_match.fault_history)),
                evidence=tuple(dict.fromkeys((*auto.evidence, *map_match.evidence))))
        rendered.append(auto)
    rendered.extend(item for item in review_models if item.component_id not in handled_review_ids)
    rendered.extend(item for item in map_models
                    if item.component_id not in consumed_map_ids and item.component_id not in ambiguous_map_ids)
    return tuple(rendered), review_rows


def _attach_verified_history(
    components: tuple[ComponentOverlayViewModel, ...],
    history_by_id: dict[str, list[dict[str, Any]]],
) -> tuple[ComponentOverlayViewModel, ...]:
    """Attach retained verification records without promoting them to live status."""
    return tuple(
        replace(model, fault_history=tuple(history_by_id[model.component_id]))
        if model.component_id in history_by_id else model
        for model in components
    )


def _prepare_photo_component_snapshot(
    st: Any, project: ProjectContext, photos: list[dict[str, Any]], *,
    photo_id: str | None = None, activate: bool = True,
) -> dict[str, Any]:
    """Prepare one photo's review snapshot without depending on Viewer rendering."""
    records = {item.image_id: item for item in project.image_records}
    valid = [item for item in photos if item.get("image_id") in records
             and isinstance(item.get("data"), bytes)]
    if not valid:
        return {"photo_id": None, "components": (), "review_scope": None, "error": None}
    ids = [item["image_id"] for item in valid]
    photo_id = photo_id or st.session_state.get("circuitcare_v2_photo_choice")
    if photo_id not in ids:
        photo_id = st.session_state.get(ACTIVE_PHOTO_KEY)
    if photo_id not in ids:
        photo_id = ids[0]
    if activate:
        st.session_state["circuitcare_v2_photo_choice"] = photo_id
    if activate and st.session_state.get(ACTIVE_PHOTO_KEY) != photo_id:
        update_selection(st.session_state, project=project, active_photo_id=photo_id)
    image_data = next(item["data"] for item in valid if item["image_id"] == photo_id)
    try:
        loaded = load_board_image(image_data)
        image_buffer = io.BytesIO()
        loaded.image.save(image_buffer, format="PNG")
        detector_image_bytes = image_buffer.getvalue()
    except (ImageLoadError, ValueError):
        return {"photo_id": photo_id, "components": (), "review_scope": None,
                "error": "PCB 이미지 로드 실패"}
    auto_result, auto_error = ensure_component_detection(
        st, image_bytes=image_data, project_id=project.project_id, photo_id=photo_id,
        review_image_bytes=detector_image_bytes, review_image_size=loaded.processed_size,
    )
    review_scope = component_review_scope(detector_image_bytes, project.project_id, photo_id)
    _result_key, records_key, _error_key = review_keys(review_scope)
    rows = list(st.session_state.get(records_key, []))
    inventory_version = component_inventory_version(
        project_id=project.project_id, board_id=project.board_id, photo_id=photo_id,
        image_sha256=hashlib.sha256(image_data).hexdigest(), components=[
            {"component_id": row.get("component_id"), "component_type": row.get("소자 종류", "unknown"),
             "bbox_xyxy": [row.get("x1", 0), row.get("y1", 0), row.get("x2", 0), row.get("y2", 0)]}
            for row in rows if isinstance(row, dict)
        ],
    )
    if activate:
        st.session_state[MAP_VERSION_KEY] = inventory_version
        st.session_state["circuitcare_v4_component_inventory_scope"] = review_scope
    visible_rows = board_visible_review_rows(rows)
    review_hash = hashlib.sha256(detector_image_bytes).hexdigest()[:16]
    side = {"front": "TOP", "back": "BOTTOM"}.get(records[photo_id].side, records[photo_id].side)
    review_models = tuple(ComponentOverlayViewModel(
        project_id=project.project_id, board_id=project.board_id or "", board_side=side,
        component_id=str(row.get("component_id") or f"auto-{review_hash}-{index + 1}"),
        component_type=str(row.get("소자 종류", "unknown")),
        display_label=(f"AI 제안 · {row.get('소자 종류', 'unknown')}" if row.get("source") == "qwen_suggestion"
                       else f"수동 · {row.get('소자 종류', 'unknown')}" if row.get("source") == "manual"
                       else f"{index + 1} · {row.get('소자 종류', 'unknown')}"),
        canonical_frame_id=photo_id, coordinate_frame="SOURCE_PHOTO_PIXEL",
        bbox_xyxy=(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
        polygon_xy=(), source_backend_photo_ids=(photo_id,), observation_ids=(),
        detection_confidence=float(row.get("신뢰도", 0.0)),
        identification_status="USER_CONFIRMED" if row.get("상태") == "확인" else "SUGGESTED_CLASS_ONLY",
        identity_match_status=None, ocr_marking=row.get("body_marking"), possible_part_number=None,
        evidence=("Qwen 미확인 박스 제안" if row.get("source") == "qwen_suggestion"
                  else "사용자 추가 박스" if row.get("source") == "manual"
                  else "YOLO component proposal; awaiting user review",),
        overlay_status="EXCLUDED" if row.get("상태") == "제외" else "CANDIDATE",
        diagnosis_display_status="DIAGNOSIS_NOT_RUN",
        location_confidence=float(row.get("location_confidence", row.get("신뢰도", 0.0))),
        type_confidence=row.get("type_confidence"), match_score=row.get("match_score"),
        component_type_status="USER_CONFIRMED" if row.get("상태") == "확인" else "UNCONFIRMED",
        presence_status="CONFIRMED" if row.get("상태") == "확인" else "UNCONFIRMED",
        ocr_status=row.get("ocr_association_status"),
        ocr_candidates=tuple(row.get("ocr_candidates", [])),
        ocr_class_hints=tuple(row.get("ocr_class_hints", [])),
        reference_designators=tuple(row.get("reference_designators", [])),
        overlay_source=row.get("source", "yolo"),
    ) for index, row in enumerate(visible_rows))
    edits = st.session_state.get(COMPONENT_EDIT_KEY, {})
    if edits:
        review_models = tuple(replace(model,
            ocr_marking=edits.get(model.component_id, {}).get("ocr_marking", model.ocr_marking) or None,
            component_info=edits.get(model.component_id, {}).get("component_info", model.component_info) or None)
            for model in review_models)
    owner = f"{project.project_id}|{project.board_id or ''}|{review_scope or ''}"
    inventories = st.session_state.get(COMPONENT_INVENTORY_KEY, {})
    inventory = inventories.get(owner, {}) if isinstance(inventories, dict) else {}
    review_models = tuple(replace(model,
        confirmed_functional_roles=tuple(inventory.get(model.component_id, {}).get("confirmed_functional_roles", [])),
        part_identity_status=inventory.get(model.component_id, {}).get("part_identity_status", "UNCONFIRMED"),
    ) for model in review_models)
    scope = AnalysisScope(project.project_id, project.board_id or "", str(photo_id))
    image_fingerprint = hashlib.sha256(image_data).hexdigest()
    review_payload = [{"id": item.component_id, "type": item.component_type, "bbox": item.bbox_xyxy,
                       "marking": item.ocr_marking, "info": item.component_info,
                       "presence": item.presence_status, "type_status": item.component_type_status,
                       "identity": item.part_identity_status,
                       "roles": item.confirmed_functional_roles} for item in review_models]
    review_version = hashlib.sha256(json.dumps(review_payload, sort_keys=True, default=str).encode()).hexdigest()
    by_scope = dict(st.session_state.get("circuitcare_v6_photo_component_snapshots", {}))
    previous_snapshot = by_scope.get(scope.key, {})
    cached_final = (previous_snapshot.get("components")
                    if previous_snapshot.get("image_fingerprint") == image_fingerprint
                    and previous_snapshot.get("review_version") == review_version else None)
    models = tuple(cached_final) if cached_final is not None else review_models
    snapshot = {"project_id": project.project_id, "board_id": project.board_id or "",
                "photo_id": photo_id, "review_scope": review_scope,
                "image_fingerprint": image_fingerprint,
                "review_version": review_version, "review_components": review_models,
                "registry_components": tuple({
                    "component_id": str(row.get("component_id") or ""),
                    "component_type": str(row.get("소자 종류", "unknown")),
                    "bbox_xyxy": [row.get("x1", 0), row.get("y1", 0), row.get("x2", 0), row.get("y2", 0)],
                    "coordinate_frame": "SOURCE_PHOTO_PIXEL",
                    "identification_status": ("EXCLUDED" if row.get("상태") == "제외" else
                                              "USER_CONFIRMED" if row.get("상태") == "확인" else "UNCONFIRMED"),
                    "presence_status": ("EXCLUDED" if row.get("상태") == "제외" else
                                        "CONFIRMED" if row.get("상태") == "확인" else "UNCONFIRMED"),
                    "component_type_status": "USER_CONFIRMED" if row.get("상태") == "확인" else "UNCONFIRMED",
                    "ocr_marking": row.get("body_marking"),
                    "ocr_candidates": row.get("ocr_candidates", []),
                    "possible_part_number": None,
                    "confirmed_functional_roles": inventory.get(str(row.get("component_id")), {}).get(
                        "confirmed_functional_roles", []),
                    "evidence": ("사용자 제외 기록",) if row.get("상태") == "제외" else
                                ("사용자 검수 확인",) if row.get("상태") == "확인" else
                                ("사진별 탐지 후보",),
                    "overlay_source": row.get("source", "yolo"),
                } for row in rows if row.get("component_id")),
                "components": models, "component_ids": tuple(item.component_id for item in models),
                "detection_status": ("BLOCKED_MODEL_UNAVAILABLE" if auto_error else
                                     "INFERENCE_COMPLETE" if auto_result is not None else "UNKNOWN"),
                "error": auto_error, "version": inventory_version,
                "analysis_scope_key": scope.key}
    by_scope[scope.key] = snapshot
    st.session_state["circuitcare_v6_photo_component_snapshots"] = by_scope
    if activate:
        st.session_state["circuitcare_v4_current_components"] = models
        st.session_state["circuitcare_v6_current_component_scope"] = scope.key
        st.session_state["circuitcare_v4_detection_status"] = snapshot["detection_status"]
        st.session_state["circuitcare_v4_component_inventory_scope"] = review_scope
    return snapshot


def _prepare_board_registry_snapshot(
    st: Any, project: ProjectContext, photos: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Refresh cached photo snapshots, then build the independent board registry."""
    records = {item.image_id: item for item in project.image_records}
    selection_before = st.session_state.get(SELECTION_KEY)
    previous_photo_id = getattr(selection_before, "active_photo_id", None)
    previous_local_id = getattr(selection_before, "selected_component", None)
    by_photo: dict[str, dict[str, Any]] = {}
    for photo in photos:
        photo_id = str(photo.get("image_id") or "")
        if photo_id not in records or not isinstance(photo.get("data"), bytes):
            continue
        by_photo[photo_id] = _prepare_photo_component_snapshot(
            st, project, photos, photo_id=photo_id, activate=False,
        )
    selected_id = st.session_state.get("circuitcare_v2_photo_choice")
    if selected_id not in by_photo:
        selected_id = st.session_state.get(ACTIVE_PHOTO_KEY)
    if selected_id not in by_photo:
        selected_id = next(iter(by_photo), None)
    by_photo = {photo_id: {**snapshot,
                           "side": records[photo_id].side,
                           "capture_type": records[photo_id].capture_type}
                for photo_id, snapshot in by_photo.items()}
    selected_snapshot = by_photo.get(selected_id)
    if selected_snapshot is not None:
        st.session_state["circuitcare_v2_photo_choice"] = selected_id
        if st.session_state.get(ACTIVE_PHOTO_KEY) != selected_id:
            update_selection(st.session_state, project=project, active_photo_id=selected_id)
        st.session_state[MAP_VERSION_KEY] = selected_snapshot.get("version")
        st.session_state["circuitcare_v4_component_inventory_scope"] = selected_snapshot.get("review_scope")
        st.session_state["circuitcare_v4_current_components"] = tuple(selected_snapshot.get("components", ()))
        st.session_state["circuitcare_v6_current_component_scope"] = selected_snapshot.get("analysis_scope_key")
        st.session_state["circuitcare_v4_detection_status"] = selected_snapshot.get("detection_status", "UNKNOWN")
    registry = build_board_component_registry(
        st.session_state, project_id=project.project_id, board_id=project.board_id,
        photo_snapshots=by_photo,
    )
    for photo_id, snapshot in by_photo.items():
        models = tuple(replace(item, board_component_id=registry["source_to_board_id"].get(
            source_key_for_component(photo_id, item.component_id))) for item in snapshot.get("components", ()))
        by_photo[photo_id] = {**snapshot, "components": models,
                              "component_ids": tuple(item.component_id for item in models)}
    registry_scopes = dict(st.session_state.get("circuitcare_v6_photo_component_snapshots", {}))
    for snapshot in by_photo.values():
        registry_scopes[snapshot["analysis_scope_key"]] = snapshot
    st.session_state["circuitcare_v6_photo_component_snapshots"] = registry_scopes
    selected_snapshot = by_photo.get(selected_id)
    if selected_snapshot is not None:
        st.session_state["circuitcare_v4_current_components"] = tuple(selected_snapshot.get("components", ()))
    st.session_state["circuitcare_v7_photo_snapshots_by_id"] = by_photo
    st.session_state["circuitcare_v7_active_board_registry"] = registry
    selected_board_id = st.session_state.get(BOARD_COMPONENT_SELECTION_KEY)
    if not selected_board_id and previous_photo_id and previous_local_id:
        selected_board_id = registry["source_to_board_id"].get(
            source_key_for_component(previous_photo_id, previous_local_id))
    if selected_board_id:
        target_source = next((source for row in registry.get("components", [])
                             if row.get("board_component_id") == selected_board_id
                             for source in row.get("source_components", [])
                             if source.get("photo_id") == selected_id
                             and source.get("review_status") != "EXCLUDED"), None)
        if target_source:
            local_id = target_source.get("component_id")
            if (getattr(selection_before, "active_photo_id", None) != selected_id
                    or getattr(selection_before, "selected_component", None) != local_id):
                update_selection(st.session_state, project=project, active_photo_id=selected_id,
                                 selected_component=local_id)
        else:
            # An excluded, retired, or non-visible registry source must not
            # overwrite a newer local viewer click with a null/stale selection.
            st.session_state.pop(BOARD_COMPONENT_SELECTION_KEY, None)
    return registry, selected_snapshot


def _render_pcb(st: Any, project: ProjectContext, photos: list[dict[str, Any]]) -> None:
    records = {item.image_id: item for item in project.image_records}
    valid = [item for item in photos if item.get("image_id") in records and isinstance(item.get("data"), bytes)]
    if not valid:
        st.info("PCB 사진 없음")
        if st.button("사진 추가", key="cc_v2_preview_add_photo", use_container_width=True):
            st.session_state["circuitcare_v2_edit_project_id"] = project.project_id
            st.session_state["circuitcare_v2_edit_board_id"] = project.board_id
            clear_intake_draft(st.session_state)
            st.session_state[SCREEN_KEY] = "entry"
            st.session_state.pop(SELECTION_KEY, None)
            st.session_state.pop("circuitcare_v2_photo_choice", None)
            st.session_state.pop(VIEW_ZOOM_KEY, None)
            st.rerun()
        return
    ids = [item["image_id"] for item in valid]
    selected_id = st.session_state.get("circuitcare_v2_photo_choice")
    if selected_id not in ids:
        selected_id = st.session_state.get(ACTIVE_PHOTO_KEY)
    if selected_id not in ids:
        selected_id = ids[0]
    st.session_state["circuitcare_v2_photo_choice"] = selected_id
    prepared_snapshot = _prepare_photo_component_snapshot(st, project, photos)
    active_side = records[selected_id].side

    # A single board map covers every supported photo on both physical sides.
    ready_inputs = []
    if project.board_id:
        for item in valid:
            adapted = adapt_component_map_input(records[item["image_id"]], board_id=project.board_id, data=item["data"])
            if adapted.request is not None:
                ready_inputs.append(adapted.request)
    photo_fingerprint = photo_input_fingerprint(list(project.image_records))
    cache_key = component_map_cache_key(
        project_id=project.project_id, board_id=project.board_id or "", inputs=ready_inputs,
        photo_manifest=list(project.image_records),
    )
    provider = optional_backend_provider()
    provider_options = getattr(provider, "cache_options", {}) if provider else {}
    if provider_options:
        cache_key = component_map_cache_key(
            project_id=project.project_id, board_id=project.board_id or "", inputs=ready_inputs,
            photo_manifest=list(project.image_records), options=provider_options,
        )
    entry = st.session_state.get(COMPONENT_MAP_STATE_KEY)
    if not component_map_entry_is_current(
        entry, project_id=project.project_id, board_id=project.board_id,
        photo_fingerprint=photo_fingerprint, cache_key=cache_key,
    ):
        clear_component_map_state(st.session_state, clear_viewer_selection=False)
        entry = None
    if provider is None:
        st.caption("백엔드 미연결 · 원본 사진 보기")
    elif not ready_inputs:
        st.caption("정합 가능한 사진 없음 · 원본 사진 보기")
    else:
        if st.button("사진 정합 실행", key="circuitcare_v3_build_map"):
            try:
                raw_map = provider.build_component_map(
                    board_id=project.board_id,
                    photos=[{"data": p.data, "side": p.side, "role": p.role, "name": p.name}
                            for p in ready_inputs],
                    options=provider_options,
                )
                validated_map = validate_component_map(raw_map, board_id=project.board_id)
                photo_id_map = map_backend_photo_ids(
                    project_id=project.project_id, board_id=project.board_id,
                    inputs=ready_inputs, result=validated_map,
                )
                # Publish map and ID mapping as one owner-checked cache entry.
                entry = {
                    "project_id": project.project_id,
                    "board_id": project.board_id,
                    "input_fingerprint": photo_fingerprint,
                    "component_map_cache_key": cache_key,
                    "component_map": validated_map,
                    "photo_id_map": photo_id_map,
                    "failed_artifacts": [],
                }
                st.session_state[COMPONENT_MAP_STATE_KEY] = entry
                st.session_state.pop("circuitcare_v3_processing_error", None)
            except Exception as exc:
                LOGGER.exception("Component Map build failed")
                st.session_state["circuitcare_v3_processing_error"] = str(exc)
                st.warning("정합 불가 · 원본 사진 보기")
        entry = st.session_state.get(COMPONENT_MAP_STATE_KEY)
        if component_map_entry_is_current(
            entry, project_id=project.project_id, board_id=project.board_id,
            photo_fingerprint=photo_fingerprint, cache_key=cache_key,
        ):
            st.caption("정합 결과 후보 · 정확도 미검증 · 사진면별 결과")

    entry = st.session_state.get(COMPONENT_MAP_STATE_KEY)
    if not component_map_entry_is_current(
        entry, project_id=project.project_id, board_id=project.board_id,
        photo_fingerprint=photo_fingerprint, cache_key=cache_key,
    ):
        entry = None
    component_map = entry["component_map"] if entry else None
    st.session_state[MAP_VERSION_KEY] = cache_key if component_map else None
    side_enum = {"front": "TOP", "back": "BOTTOM"}.get(active_side)
    side_result = component_map.get("sides", {}).get(side_enum, {}) if component_map and side_enum else {}
    # Keep the viewer on the selected source photo; no separate view-mode control.
    canonical = None
    st.session_state.pop("circuitcare_v3_view_mode", None)
    mode = "원본 사진"
    if not ids:
        st.info("선택한 면의 원본 사진 없음")
        return
    selected = st.selectbox(
        "원본 사진", ids, index=ids.index(selected_id),
        format_func=lambda image_id: _photo_label(records[image_id], ids.index(image_id) + 1),
        key="circuitcare_v2_photo_choice",
    )
    if selected != st.session_state.get(ACTIVE_PHOTO_KEY):
        update_selection(st.session_state, project=project, active_photo_id=selected)
    record = records[selected]
    image_data = next(item["data"] for item in valid if item["image_id"] == selected)
    size = record.image_dimensions
    if size:
        st.caption(f"{record.filename or 'PCB 사진'} · {record.capture_type} · {size[0]}×{size[1]} px")
    else:
        st.caption(f"{record.filename or 'PCB 사진'} · {record.capture_type}")
    canonical_png_bytes = None
    try:
        if mode == "통합 이미지" and canonical:
            # Decoded above before the widget is built so failures can safely switch its state.
            image = decoded_canonical.image
            source_size = (decoded_canonical.width, decoded_canonical.height)
            display_size = source_size
            active_image_id = decoded_canonical.frame_id
            canonical_png_bytes = base64.b64decode(canonical["image_png_base64"], validate=True)
            st.caption(f"MOSAIC_PIXEL · 기준 사진 {decoded_canonical.reference_backend_photo_id or '미지정'} · 정합 후보 · 미검증")
        else:
            loaded = load_board_image(image_data)
            image = loaded.image
            source_size = loaded.original_size
            display_size = loaded.processed_size
            active_image_id = selected
    except (ImageLoadError, ValueError):
        st.warning("이미지 표시 실패 · 원본 사진 보기")
        return
    auto_detection_result = None
    auto_detection_error = None
    detector_image_bytes = None
    if mode == "원본 사진":
        detector_image = io.BytesIO()
        image.save(detector_image, format="PNG")
        detector_image_bytes = detector_image.getvalue()
        auto_detection_result, auto_detection_error = ensure_component_detection(
            st, image_bytes=image_data, project_id=project.project_id,
            photo_id=selected, review_image_bytes=detector_image_bytes,
            review_image_size=display_size,
        )
        st.session_state["circuitcare_v4_auto_detection_result"] = auto_detection_result
        st.session_state["circuitcare_v4_auto_detection_error"] = auto_detection_error
        review_scope = component_review_scope(detector_image_bytes, project.project_id, selected)
        _result_key, records_key, _error_key = review_keys(review_scope)
        review_rows = st.session_state.get(records_key, [])
        active_inventory_version = component_inventory_version(
            project_id=project.project_id, board_id=project.board_id, photo_id=selected,
            image_sha256=hashlib.sha256(image_data).hexdigest(), components=[
                {"component_id": row.get("component_id"), "component_type": row.get("소자 종류", "unknown"),
                 "bbox_xyxy": [row.get("x1", 0), row.get("y1", 0), row.get("x2", 0), row.get("y2", 0)]}
                for row in review_rows if isinstance(row, dict)
            ],
        )
        st.session_state[MAP_VERSION_KEY] = active_inventory_version
        st.session_state["circuitcare_v4_component_inventory_scope"] = review_scope
    side_result = side_result if isinstance(side_result, dict) else {}
    detection_status = str(side_result.get("detection_status", "UNKNOWN"))
    if auto_detection_error:
        detection_status = "BLOCKED_MODEL_UNAVAILABLE"
    elif auto_detection_result is not None:
        detection_status = "INFERENCE_COMPLETE"
    overlay_result = adapt_component_map_components(
        component_map, project_id=project.project_id, board_id=project.board_id or "", side=side_enum or "",
    ) if component_map and side_enum else None
    coordinate_frame_valid = True
    if mode == "통합 이미지" and canonical:
        component_models = overlay_result.components if overlay_result and overlay_result.status == "READY" else ()
        frame_id = decoded_canonical.frame_id
        viewer_size = source_size
        if (overlay_result and overlay_result.status not in {"READY", "NO_COMPONENTS"}) or (
            overlay_result and (overlay_result.frame_id != frame_id or overlay_result.image_size != viewer_size
                                or overlay_result.coordinate_frame != "MOSAIC_PIXEL")
        ):
            coordinate_frame_valid = False
            component_models = ()
    else:
        component_models = ()
        frame_id = selected
        viewer_size = display_size
        if side_enum and entry:
            photo_mapping = next((row for row in entry.get("photo_id_map", [])
                                  if row.project_id == project.project_id and row.board_id == project.board_id
                                  and row.frontend_photo_id == selected and row.mapping_status == "MAPPED"), None)
            if photo_mapping and photo_mapping.backend_photo_id:
                source_models = adapt_original_photo_components(
                    component_map, project_id=project.project_id, board_id=project.board_id or "",
                    side=side_enum, frontend_photo_id=selected,
                    backend_photo_id=photo_mapping.backend_photo_id, image_size=source_size,
                )
                component_models = scale_overlay_to_display(
                    source_models, source_size=source_size, display_size=display_size,
                )
    if overlay_result and overlay_result.status == "INVALID_DATA":
        st.warning("일부 부품 좌표를 검증할 수 없어 표시하지 않았습니다.")
    review_rows: list[dict[str, Any]] = []
    auto_models: tuple[ComponentOverlayViewModel, ...] = ()
    records_key: str | None = None
    if mode == "원본 사진" and detector_image_bytes is not None:
        review_hash = hashlib.sha256(detector_image_bytes).hexdigest()[:16]
        review_scope = component_review_scope(
            detector_image_bytes, project.project_id, selected,
        )
        result_key, records_key, _error_key = review_keys(review_scope)
        st.session_state["cc_active_component_review_scope"] = review_scope
        st.session_state["cc_active_component_review_image"] = detector_image_bytes
        review_rows = st.session_state.get(records_key, [])
        photo_snapshot = prepared_snapshot
        auto_models = tuple(photo_snapshot.get("review_components", ()))
        st.session_state["circuitcare_v4_selected_marking_context"] = None
    else:
        review_scope = None
        active_inventory_version = cache_key if component_map else None
        st.session_state[MAP_VERSION_KEY] = active_inventory_version
        st.session_state["circuitcare_v4_component_inventory_scope"] = active_inventory_version
        st.session_state.pop("cc_active_component_review_scope", None)
        st.session_state.pop("cc_active_component_review_image", None)
        st.session_state["circuitcare_v4_selected_marking_context"] = None
    component_edits = st.session_state.get(COMPONENT_EDIT_KEY, {})
    if component_edits:
        updated_models = []
        for model in component_models:
            edit = component_edits.get(model.component_id, {})
            if edit:
                model = replace(
                    model,
                    ocr_marking=edit.get("ocr_marking", model.ocr_marking) or None,
                    possible_part_number=(
                        edit.get("ocr_marking", model.possible_part_number)
                        if model.component_type.casefold() in {"ic", "integrated_circuit"}
                        else model.possible_part_number
                    ),
                    component_info=edit.get("component_info", model.component_info) or None,
                )
            updated_models.append(model)
        component_models = tuple(updated_models)
    target_id = str(selected or st.session_state.get(ACTIVE_PHOTO_KEY) or "target:default")
    snapshots = st.session_state.get("circuitcare_v6_photo_component_snapshots", {})
    photo_scope = AnalysisScope(project.project_id, project.board_id or "", target_id)
    scoped_snapshot = snapshots.get(photo_scope.key) if isinstance(snapshots, dict) else None
    board_registry = st.session_state.get("circuitcare_v7_active_board_registry")
    board_mode = (isinstance(board_registry, dict)
                  and board_registry.get("project_id") == project.project_id
                  and board_registry.get("board_id") == (project.board_id or ""))
    analysis_scope = (AnalysisScope(project.project_id, project.board_id or "", "board:registry:v1")
                      if board_mode else photo_scope)
    analysis_components = (diagnosis_components(board_registry) if board_mode else
                           tuple(scoped_snapshot.get("components", ())) if isinstance(scoped_snapshot, dict)
                           else tuple(component_models))
    review_scope = (scoped_snapshot.get("review_scope") if isinstance(scoped_snapshot, dict) else review_scope)
    inventory_owner = f"{project.project_id}|{project.board_id or ''}|{review_scope or ''}"
    inventory_rows = st.session_state.get(COMPONENT_INVENTORY_KEY, {})
    analysis_inventory = ({"__board_registry__": {"registry_version": board_registry.get("version")}}
                          if board_mode else
                          inventory_rows.get(inventory_owner, {}) if isinstance(inventory_rows, dict) else {})
    aliases = measurement_component_aliases(board_registry) if board_mode else None
    analysis_snapshot = current_analysis_snapshot(
        st, project, components=analysis_components, component_inventory=analysis_inventory,
        analysis_target_id=analysis_scope.target_id,
        component_id_aliases=aliases,
        board_registry=board_registry if board_mode else None,
    )
    analysis_state = analysis_snapshot["record"]
    # Legacy photo analyses and Component Map verdicts do not paint the board.
    # Only the current Analysis State Manager scope may paint live statuses.
    # the live PCB; retained verdicts stay in fault_history.
    component_models = tuple(replace(
        model, fault_candidate_status=None, diagnosis_display_status="DIAGNOSIS_NOT_RUN",
        fault_hypothesis=None, fault_role_status=None, measurement_status=None,
        solder_status=None, fault_score=None, fault_evidence=(), recommended_tests=(),
    ) for model in component_models)
    candidate_state_by_scope = st.session_state.get("circuitcare_v6_diagnosis_candidates_by_scope", {})
    scoped_overlay = (candidate_state_by_scope.get(analysis_scope.key)
                     if isinstance(candidate_state_by_scope, dict) else {})
    current_result = analysis_snapshot.get("result")
    candidate_state = ({str(row.get("component_id")): row
                        for row in current_result.get("component_candidates", [])[:3]
                        if isinstance(row, dict) and row.get("component_id")}
                       if analysis_state.get("status") == AnalysisStatus.CURRENT.value
                       and isinstance(current_result, dict) else {})
    verified_history = (
        [row for row in scoped_overlay.get("verified_component_history", []) if isinstance(row, dict)]
        if isinstance(scoped_overlay, dict)
        and scoped_overlay.get("project_id") == project.project_id
        and scoped_overlay.get("board_id") == project.board_id else []
    )
    history_by_id: dict[str, list[dict[str, Any]]] = {}
    for history_row in verified_history:
        history_by_id.setdefault(str(history_row.get("component_id") or ""), []).append(history_row)
    if isinstance(candidate_state, dict) and candidate_state:
        status_map = {
            "SUSPECTED": ("SUSPECTED_FAULT", "의심 후보 · 미확정"),
            "FAULT_SUPPORTED": ("FAULT_SUPPORTED", "검증 검사 결과 · 해당 고장 가설 지지"),
            "FAULT_REFUTED": ("FAULT_REFUTED", "검증 검사 결과 · 해당 고장 가설 반박"),
            "INCONCLUSIVE": ("INCONCLUSIVE", "측정 판정 불가"),
            "CONFLICTED": ("CONFLICTED", "측정 결과 충돌 · 판단 보류"),
        }
        component_models = tuple(
            replace(model,
                    fault_candidate_status=status_map.get(candidate_state[str(
                        model.board_component_id if board_mode else model.component_id)].get("visual_status"),
                                                          ("INCONCLUSIVE", "판정 불가"))[0],
                    diagnosis_display_status=status_map.get(candidate_state[str(
                        model.board_component_id if board_mode else model.component_id)].get("visual_status"),
                                                            ("INCONCLUSIVE", "판정 불가"))[1],
                    fault_hypothesis=candidate_state[str(model.board_component_id if board_mode else model.component_id)].get("fault_hypothesis"),
                    fault_role_status=candidate_state[str(model.board_component_id if board_mode else model.component_id)].get("role_status"),
                    measurement_status=candidate_state[str(model.board_component_id if board_mode else model.component_id)].get("measurement_status"),
                    solder_status=candidate_state[str(model.board_component_id if board_mode else model.component_id)].get("solder_status"),
                    fault_score=candidate_state[str(model.board_component_id if board_mode else model.component_id)].get("score"),
                    fault_evidence=tuple(candidate_state[str(model.board_component_id if board_mode else model.component_id)].get("evidence", [])),
                    recommended_tests=tuple(candidate_state[str(model.board_component_id if board_mode else model.component_id)].get("recommended_tests", [])),
                    fault_history=tuple(history_by_id.get(str(model.board_component_id if board_mode else model.component_id), [])))
            if str(model.board_component_id if board_mode else model.component_id) in candidate_state else model
            for model in component_models
        )
    if history_by_id:
        component_models = _attach_verified_history(component_models, history_by_id)
    if review_scope and records_key:
        previous_rows = [dict(row) for row in review_rows]
        component_models, review_rows = _bind_review_models_to_map(
            component_models, auto_models, review_rows)
        if review_rows != previous_rows:
            st.session_state[records_key] = review_rows
        selection = st.session_state.get(SELECTION_KEY)
        selected_id = getattr(selection, "selected_component", None)
        replacement_id = next((str(row["component_id"]) for row in review_rows
                               if selected_id in row.get("associated_component_ids", [])), None)
        if replacement_id:
            update_selection(st.session_state, project=project, selected_component=replacement_id)
    if board_mode and candidate_state:
        status_map = {
            "SUSPECTED": ("SUSPECTED_FAULT", "의심 후보 · 미확정"),
            "FAULT_SUPPORTED": ("FAULT_SUPPORTED", "검증 검사 결과 · 해당 고장 가설 지지"),
            "FAULT_REFUTED": ("FAULT_REFUTED", "검증 검사 결과 · 해당 고장 가설 반박"),
            "INCONCLUSIVE": ("INCONCLUSIVE", "측정 판정 불가"),
            "CONFLICTED": ("CONFLICTED", "측정 결과 충돌 · 판단 보류"),
        }
        projected = []
        for model in component_models:
            candidate = candidate_state.get(str(model.board_component_id or ""))
            if candidate is None:
                projected.append(model)
                continue
            status, label = status_map.get(candidate.get("visual_status"), ("INCONCLUSIVE", "판정 불가"))
            projected.append(replace(
                model, fault_candidate_status=status, diagnosis_display_status=label,
                fault_hypothesis=candidate.get("fault_hypothesis"),
                fault_role_status=candidate.get("role_status"),
                measurement_status=candidate.get("measurement_status"),
                solder_status=candidate.get("solder_status"), fault_score=candidate.get("score"),
                fault_evidence=tuple(candidate.get("evidence", [])),
                recommended_tests=tuple(candidate.get("recommended_tests", [])),
                fault_history=tuple(history_by_id.get(str(model.board_component_id), [])),
            ))
        component_models = tuple(projected)
    component_models = tuple(component_models)
    target_scope = AnalysisScope(project.project_id, project.board_id or "", str(selected))
    snapshot_map = dict(st.session_state.get("circuitcare_v6_photo_component_snapshots", {}))
    photo_snapshot = snapshot_map.get(target_scope.key)
    if isinstance(photo_snapshot, dict):
        photo_snapshot = {**photo_snapshot, "components": component_models,
                          "component_ids": tuple(item.component_id for item in component_models)}
        snapshot_map[target_scope.key] = photo_snapshot
        st.session_state["circuitcare_v6_photo_component_snapshots"] = snapshot_map
    st.session_state["circuitcare_v4_current_components"] = component_models
    st.session_state["circuitcare_v6_current_component_scope"] = target_scope.key
    st.session_state["circuitcare_v4_detection_status"] = detection_status
    current_selection = st.session_state.get(SELECTION_KEY)
    selected_component_id = getattr(current_selection, "selected_component", None)
    _record_selection_trace(st.session_state,
                            f"unified_workspace · 현재 선택 {selected_component_id or '없음'} · 모델 {len(component_models)}개")
    if selected_component_id and selected_component_id not in {item.component_id for item in component_models}:
        LOGGER.warning("pcb_selection stage=unified_invalidate reason=component_missing selected=%s current_ids=%s",
                       selected_component_id, [item.component_id for item in component_models])
        _record_selection_trace(st.session_state,
                               f"unified_workspace · 현재 박스 목록에 ID 없음 · {selected_component_id}")
        update_selection(st.session_state, project=project, selected_component=None)
        selected_component_id = None
    transform = requested_view_transform(
        source_size=source_size,
        display_size=viewer_size,
        active_photo_id=active_image_id,
    )
    transform.update({
        "project_id": project.project_id,
        "board_id": project.board_id,
        "active_board_side": side_enum or active_side,
        "active_view_mode": mode,
        "source_frame_id": active_image_id,
        "canonical_frame_id": canonical.get("image_id") if canonical else None,
        "source_dimensions": list(source_size),
        "canonical_dimensions": canonical.get("image_size_px") if canonical else None,
        "registration_status": "CANDIDATE_GEOMETRY_ONLY" if mode == "통합 이미지" else "SOURCE_PHOTO",
        "coordinate_status": "UNVERIFIED_RESPONSIVE_CSS",
    })

    def add_manual_box(bbox: list[int]) -> None:
        if mode != "원본 사진" or not detector_image_bytes or not review_scope:
            return
        _result_key, records_key, _error_key = review_keys(review_scope)
        rows = list(st.session_state.get(records_key, []))
        manual_id = f"manual-{review_hash}-{hashlib.sha256(repr(tuple(bbox)).encode()).hexdigest()[:12]}"
        rows.append({"component_id": manual_id, "소자 종류": "unknown", "상태": "검수 필요", "신뢰도": 0.0,
                     "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3], "source": "manual"})
        st.session_state[records_key] = rows
        from circuitcare.vision.component_recognition_panel import invalidate_component_review_state
        invalidate_component_review_state(st.session_state, scope=review_scope,
                                         project_id=project.project_id, board_id=project.board_id)
        update_selection(st.session_state, project=project,
                         selected_component=manual_id)

    interactive, _viewer_key, viewer_coordinate_status = render_component_viewer(
        st, image=image, source_image_bytes=image_data, canonical_png_bytes=canonical_png_bytes,
        image_size_px=viewer_size, components=tuple(component_models), project=project,
        board_side=side_enum or active_side, frame_id=frame_id,
        selected_component_id=selected_component_id,
        # Keep the selection owner key identical to _render_pcb's invalidation
        # key even when this photo has no registered component map yet.
        map_cache_key=cache_key, image_mode=mode,
        on_box_draw=add_manual_box if mode == "원본 사진" and detector_image_bytes is not None else None,
    )
    st.session_state["cc_active_component_overlap_key"] = f"{OVERLAPPING_SELECTION_KEY}_{_viewer_key}"
    current_selection = st.session_state.get(SELECTION_KEY)
    current_local_id = getattr(current_selection, "selected_component", None)
    if board_mode and current_local_id:
        board_component_id = board_registry.get("source_to_board_id", {}).get(
            source_key_for_component(str(selected), current_local_id))
        if board_component_id:
            st.session_state[BOARD_COMPONENT_SELECTION_KEY] = board_component_id
        else:
            st.session_state.pop(BOARD_COMPONENT_SELECTION_KEY, None)
    if candidate_state:
        st.caption("진단 표시 · 노랑 SUSPECTED: 미확정 후보 · 빨강 FAULT_SUPPORTED: 측정이 특정 가설을 지지 · "
                   "초록 FAULT_REFUTED: 특정 가설 반박 · 회색 INCONCLUSIVE: 판정 불가 · "
                   "충돌 CONFLICTED: 증거 상충. 소자 전체의 고장·정상을 확정하지 않습니다.")
    if not interactive:
        with st.container(key="cc_viewer_canvas"):
            st.image(image, width=720)
        if component_models:
            st.warning("인터랙티브 Viewer를 사용할 수 없습니다 · 부품 Overlay를 숨겼습니다.")
        else:
            st.caption("Zoom / Pan Viewer를 사용할 수 없어 정적 이미지로 표시합니다.")
    if not coordinate_frame_valid:
        viewer_coordinate_status = "INVALID_COORDINATE_FRAME"
    transform["coordinate_status"] = viewer_coordinate_status
    st.session_state[VIEW_TRANSFORM_KEY] = transform
    if mode == "원본 사진":
        mapped = entry.get("photo_id_map", []) if entry else []
        photo_mapping = next((row for row in mapped if row.frontend_photo_id == selected and row.project_id == project.project_id), None)
        st.session_state["circuitcare_v4_detection_summary"] = {
            "count": len(auto_detection_result.get("detections", [])) if auto_detection_result else None,
            "error": auto_detection_error,
            "registration": REGISTRATION_LABELS.get(photo_mapping.registration_status, "상태 미확인") if photo_mapping else "정합 결과 없음",
        }


def _photo_label(record: Any, number: int) -> str:
    role = {
        "full_view": "전체", "close_up": "확대", "oblique": "비스듬히",
        "marking_detail": "부품 표기", "unknown": "미지정",
    }[record.capture_type]
    return f"사진 {number} · {role}"


def render_workbench_shell(st: Any, project: ProjectContext) -> None:
    """Keep the PCB Viewer fixed while rendering one selected task panel."""
    _style(st)
    selection = st.session_state.get(SELECTION_KEY)
    if getattr(selection, "project_id", None) != project.project_id:
        enter_workbench(
            st.session_state, project,
            preview=st.session_state.get(MODE_KEY) == "PREVIEW_ONLY",
        )
    photos = st.session_state.get(INTAKE_IMAGES_KEY, [])
    board_registry, photo_snapshot = _prepare_board_registry_snapshot(st, project, photos)
    snapshot_components = diagnosis_components(board_registry)
    measurement_aliases = measurement_component_aliases(board_registry)
    analysis_target_id = "board:registry:v1"
    st.session_state["circuitcare_v7_analysis_target_id"] = analysis_target_id
    component_inventory = {"__board_registry__": {"registry_version": board_registry.get("version")}}
    suspected_components = board_suspects(st.session_state, project)
    live_analysis_snapshot = current_analysis_snapshot(
        st, project, components=snapshot_components, component_inventory=component_inventory,
        analysis_target_id=analysis_target_id, component_id_aliases=measurement_aliases,
        board_registry=board_registry, suspected_components=suspected_components,
    )

    title, add_photo, edit, new = st.columns([5, 1, 1, 1])
    with title:
        st.markdown('<div class="cc-wb-header">CircuitCare AI</div>', unsafe_allow_html=True)
        st.caption(f"{project.project_id[:8]} · 사진 {len(project.image_records)}장 · "
                   f"{_diagnosis_status_label(st, project, analysis_snapshot=live_analysis_snapshot)}")
        confirmed_count = sum(item.presence_status == "CONFIRMED" for item in snapshot_components)
        st.markdown(f'<div class="cc-wb-subtle">보드 확인 소자 {confirmed_count}개 · Registry {board_registry.get("version", "미구성")[-12:]}</div>', unsafe_allow_html=True)
    if add_photo.button("사진 추가", use_container_width=True, key="cc_v2_workbench_add_photo"):
        st.session_state["circuitcare_v2_edit_project_id"] = project.project_id
        st.session_state["circuitcare_v2_edit_board_id"] = project.board_id
        clear_intake_draft(st.session_state)
        st.session_state[SCREEN_KEY] = "entry"
        st.session_state.pop(SELECTION_KEY, None)
        st.session_state.pop("circuitcare_v2_photo_choice", None)
        st.session_state.pop(VIEW_ZOOM_KEY, None)
        st.rerun()
    if edit.button("입력 수정", use_container_width=True, key="cc_v2_workbench_edit"):
        st.session_state["circuitcare_v2_edit_project_id"] = project.project_id
        st.session_state["circuitcare_v2_edit_board_id"] = project.board_id
        clear_intake_draft(st.session_state)
        st.session_state[SCREEN_KEY] = "entry"
        st.session_state.pop(SELECTION_KEY, None)
        st.session_state.pop("circuitcare_v2_photo_choice", None)
        st.session_state.pop(VIEW_ZOOM_KEY, None)
        st.rerun()
    if new.button("새 프로젝트", use_container_width=True, key="cc_v2_workbench_new"):
        reset_intake_for_new_project(st.session_state)
        st.session_state[SCREEN_KEY] = "entry"
        st.rerun()

    preview = st.session_state.get(MODE_KEY) == "PREVIEW_ONLY"
    with st.container(key="cc_workbench_body"):
        panel_key = f"cc_workbench_panel_{project.project_id}"
        modes = ("검수", "증상", "원인", "측정")
        if panel_key not in st.session_state:
            legacy_view = st.session_state.get(f"cc_workbench_view_{project.project_id}")
            st.session_state[panel_key] = "증상" if legacy_view == "고장 분석" else "검수"
        if st.session_state.get(panel_key) not in modes:
            st.session_state[panel_key] = "검수"
        left, right = st.columns([1.85, 1], gap="large")
        with left:
            _render_pcb(st, project, st.session_state.get(INTAKE_IMAGES_KEY, []))
        with right:
            st.radio("작업 패널", modes, horizontal=True, key=panel_key,
                     label_visibility="collapsed")
            selected_mode = st.session_state[panel_key]
            if selected_mode == "검수":
                _render_component_inspector(st, project, preview)
            else:
                _render_fault_analysis_workspace(
                    st, project, preview, mode=selected_mode,
                    analysis_snapshot=live_analysis_snapshot,
                    photo_snapshot=photo_snapshot,
                    board_registry=board_registry,
                    board_components=snapshot_components,
                    component_id_aliases=measurement_aliases,
                    analysis_target_id=analysis_target_id,
                    suspected_components=suspected_components,
                )


def _selected_component(st: Any) -> ComponentOverlayViewModel | None:
    current_components = tuple(st.session_state.get("circuitcare_v4_current_components", ()))
    selection = st.session_state.get(SELECTION_KEY)
    selected_id = getattr(selection, "selected_component", None)
    return next((item for item in current_components if item.component_id == selected_id), None)


def _focus_board_registry_component(
    st: Any, project: ProjectContext, registry: dict[str, Any] | None, board_component_id: str,
) -> bool:
    rows = registry.get("components", []) if isinstance(registry, dict) else []
    row = next((item for item in rows if item.get("board_component_id") == board_component_id), None)
    sources = sorted(row.get("source_components", []),
                     key=lambda item: item.get("review_status") == "EXCLUDED") if row else []
    if not sources:
        return False
    source = sources[0]
    photo_id, local_id = source.get("photo_id"), source.get("component_id")
    st.session_state[BOARD_COMPONENT_SELECTION_KEY] = board_component_id
    st.session_state["circuitcare_v2_photo_choice"] = photo_id
    update_selection(st.session_state, project=project, active_photo_id=photo_id,
                     selected_component=local_id)
    return True


def _handle_measurement_target_change(
    state: Any, project: ProjectContext, registry: dict[str, Any] | None, widget_key: str,
) -> None:
    """Change the shared selection only for an explicit target-widget event."""
    board_component_id = state.get(widget_key)
    rows = registry.get("components", []) if isinstance(registry, dict) else []
    row = next((item for item in rows
                if item.get("board_component_id") == board_component_id), None)
    sources = sorted(row.get("source_components", []),
                     key=lambda item: item.get("review_status") == "EXCLUDED") if row else []
    if not sources:
        return
    source = sources[0]
    state[BOARD_COMPONENT_SELECTION_KEY] = board_component_id
    state["circuitcare_v2_photo_choice"] = source.get("photo_id")
    update_selection(state, project=project, active_photo_id=source.get("photo_id"),
                     selected_component=source.get("component_id"))


def _render_component_inspector(st: Any, project: ProjectContext, preview: bool) -> None:
    current_components = tuple(st.session_state.get("circuitcare_v4_current_components", ()))
    render_inspection_panel(
        st, project, preview=preview, components=current_components,
        selected_component=_selected_component(st),
        detection_status=st.session_state.get("circuitcare_v4_detection_status", "UNKNOWN"),
        selected_marking_context=st.session_state.get("circuitcare_v4_selected_marking_context"),
        component_map_version=st.session_state.get(MAP_VERSION_KEY),
        inventory_scope=st.session_state.get("circuitcare_v4_component_inventory_scope"),
        review_image_bytes=st.session_state.get("cc_active_component_review_image"),
        review_scope=st.session_state.get("cc_active_component_review_scope"),
        board_registry=st.session_state.get("circuitcare_v7_active_board_registry"),
    )
    review_image = st.session_state.get("cc_active_component_review_image")
    review_scope = st.session_state.get("cc_active_component_review_scope")
    summary = st.session_state.get("circuitcare_v4_detection_summary", {})
    if review_image is not None and review_scope and review_scope.startswith(f"{project.project_id}_"):
        photo_id = st.session_state.get("circuitcare_v2_photo_choice")
        detection_count = summary.get("count")
        suffix = f" · {detection_count}개" if detection_count is not None else ""
        with st.expander(f"소자 탐지 · 박스 검수{suffix}", expanded=False):
            if summary.get("error"):
                st.error(f"자동 탐지 실패 · {summary['error']}")
            elif detection_count == 0:
                st.info("이 사진에서 탐지 박스를 찾지 못했습니다.")
            elif detection_count is None:
                st.caption(f"검출 상태 · {st.session_state.get('circuitcare_v4_detection_status', '미확인')}")
            st.caption(f"정합 상태 · {summary.get('registration', '정합 결과 없음')}")
            render_component_recognition_panel(
                st, image_bytes=review_image, project_id=project.project_id,
                board_id=project.board_id, photo_id=photo_id,
                auto_result=st.session_state.get("circuitcare_v4_auto_detection_result"),
                auto_error=st.session_state.get("circuitcare_v4_auto_detection_error"),
            )
    _render_board_identity_controls(st, project)


def _render_board_identity_controls(st: Any, project: ProjectContext) -> None:
    registry = st.session_state.get("circuitcare_v7_active_board_registry")
    if not isinstance(registry, dict):
        return
    inactive_links = registry.get("inactive_links", [])
    sources = [source for row in registry.get("components", [])
               for source in row.get("source_components", [])]
    can_merge = (len(registry.get("source_to_board_id", {})) >= 2
                 and len({source.get("photo_id") for source in sources}) >= 2)
    if not can_merge and not inactive_links:
        return
    labels = {source["source_key"]: (
        f"{source['photo_id']} · {source['component_id']} · "
        f"{source.get('component_type', 'unknown')} · {source.get('review_status', 'UNCONFIRMED')}"
    ) for source in sources}
    entry_owner = json.dumps([project.project_id, project.board_id or ""], separators=(",", ":"))
    entry = st.session_state.get(REGISTRY_KEY, {}).get(entry_owner, {})
    links = entry.get("confirmed_links", [])
    with st.expander("여러 사진에서 같은 소자 확인", expanded=False):
        st.caption("사진 간 자동 위치 병합은 하지 않습니다. 같은 소자임을 직접 확인한 관측만 연결됩니다.")
        if can_merge:
            left = st.selectbox("사진 관측 A", list(labels), format_func=labels.get,
                                key=f"cc_registry_left_{project.project_id}")
            right = st.selectbox("사진 관측 B", list(labels), format_func=labels.get,
                                 key=f"cc_registry_right_{project.project_id}")
            if st.button("같은 보드 소자로 연결", key=f"cc_registry_merge_{project.project_id}",
                         disabled=left == right or labels[left].split(" · ")[0] == labels[right].split(" · ")[0]):
                try:
                    confirm_component_match(st.session_state, project_id=project.project_id,
                                            board_id=project.board_id, left_source_key=left,
                                            right_source_key=right)
                    st.rerun()
                except ValueError as exc:
                    st.warning(str(exc))
            if links:
                st.caption("사용자 확인으로 연결한 관계")
                for index, link in enumerate(links):
                    if len(link) != 2:
                        continue
                    a, b = link
                    st.caption(f"{labels.get(a, a)} ↔ {labels.get(b, b)}")
                    if st.button("연결 해제", key=f"cc_registry_unlink_{project.project_id}_{index}"):
                        remove_component_match(st.session_state, project_id=project.project_id,
                                               board_id=project.board_id, left_source_key=a,
                                               right_source_key=b)
                        st.rerun()
        for item in inactive_links:
            left_key, right_key = item.get("source_keys", ["", ""])
            reason = str(item.get("reason", "RECONFIRMATION_REQUIRED"))
            reason_labels = {
                "PHOTO_OR_COMPONENT_MISSING": "사진 또는 소자 없음",
                "RECONFIRMATION_REQUIRED": "확인 이력 없음",
                "PHOTO_REPLACED_RECONFIRMATION_REQUIRED": "사진 교체됨",
                "PHOTO_REDETECTED_RECONFIRMATION_REQUIRED": "소자 재탐지·검수 변경됨",
                "PHOTO_COMPONENT_MAP_CHANGED_RECONFIRMATION_REQUIRED": "사진별 소자 목록 변경됨",
                "SAME_PHOTO_COMPONENT_COLLISION_RECONFIRMATION_REQUIRED": "같은 사진의 서로 다른 소자 포함",
            }
            st.caption(f"비활성 연결 · {labels.get(left_key, left_key)} ↔ "
                       f"{labels.get(right_key, right_key)} · "
                       f"{reason_labels.get(reason, reason)} · 다시 선택해 확인하세요.")


def _render_fault_analysis_workspace(
    st: Any, project: ProjectContext, preview: bool, *,
    mode: str = "증상",
    analysis_snapshot: dict[str, Any] | None = None,
    photo_snapshot: dict[str, Any] | None = None,
    board_registry: dict[str, Any] | None = None,
    board_components: tuple[Any, ...] | None = None,
    component_id_aliases: dict[str, str | None] | None = None,
    analysis_target_id: str | None = None,
    suspected_components: list[dict[str, Any]] | None = None,
) -> None:
    if photo_snapshot is None:
        target_id = str(st.session_state.get(ACTIVE_PHOTO_KEY) or "target:default")
        scope = AnalysisScope(project.project_id, project.board_id or "", target_id)
        snapshots = st.session_state.get("circuitcare_v6_photo_component_snapshots", {})
        photo_snapshot = snapshots.get(scope.key) if isinstance(snapshots, dict) else None
        if not isinstance(photo_snapshot, dict):
            photo_snapshot = {"components": tuple(st.session_state.get("circuitcare_v4_current_components", ())),
                              "review_scope": st.session_state.get("circuitcare_v4_component_inventory_scope")}
    if board_components is None and isinstance(board_registry, dict):
        board_components = diagnosis_components(board_registry)
    if board_components is not None:
        components = tuple(board_components)
    else:
        components = tuple(photo_snapshot.get("components", ()))
    if analysis_snapshot is None:
        inventory = ({"__board_registry__": {"registry_version": board_registry.get("version")}}
                     if isinstance(board_registry, dict) else {})
        analysis_snapshot = current_analysis_snapshot(
            st, project, components=components,
            component_inventory=inventory,
            analysis_target_id=analysis_target_id, component_id_aliases=component_id_aliases,
            board_registry=board_registry, suspected_components=suspected_components,
        )
    selection = st.session_state.get(SELECTION_KEY)
    selected = _selected_component(st)
    review_scope = st.session_state.get("circuitcare_v4_component_inventory_scope")
    owner = f"{project.project_id}|{project.board_id or ''}|{review_scope or st.session_state.get(MAP_VERSION_KEY) or ''}"
    inventories = st.session_state.get(COMPONENT_INVENTORY_KEY, {})
    inventory = inventories.get(owner, {}) if isinstance(inventories, dict) else {}
    result = analysis_snapshot.get("result")
    labels = {"증상": "증상 입력", "원인": "원인 분석", "측정": "측정 검증"}
    with st.container(border=True):
        st.markdown(f'<div class="cc-analysis-step">{labels.get(mode, "분석")}</div>', unsafe_allow_html=True)
        st.markdown('<div class="cc-analysis-copy">규칙 기반 결과를 기준으로 표시합니다. 후보는 측정으로 검증되기 전까지 미확정 가설입니다.</div>', unsafe_allow_html=True)
    record = (analysis_snapshot or {}).get("record", {})
    state_status = str(record.get("status", "NOT_RUN"))
    state_labels = {"NOT_RUN": "분석 전", "RUNNING": "분석 중", "CURRENT": "현재 입력 기준 최신",
                    "STALE": "입력 변경 · 재분석 필요", "FAILED": "분석 실패"}
    st.caption(f"분석 상태 · {state_labels.get(state_status, state_status)}")
    if state_status == "STALE" and record.get("reason"):
        st.warning(f"기존 결과는 현재 입력과 달라 사용하지 않습니다 · {record['reason']}")
    if mode == "증상":
        render_symptom_diagnosis_panel(
            st, project, components=components, preview=preview,
            component_inventory=({"__board_registry__": {"registry_version": board_registry.get("version")}}
                                 if isinstance(board_registry, dict) else inventory),
            analysis_target_id=analysis_target_id, component_id_aliases=component_id_aliases,
            board_registry=board_registry,
            suspected_components=suspected_components,
        )
    elif mode == "원인":
        _render_cause_dashboard(st, project, result, analysis_snapshot=analysis_snapshot,
                                board_registry=board_registry)
    elif mode == "측정":
        st.markdown("#### 측정 기록과 가설 검증")
        if components:
            component_ids = [item.component_id for item in components]
            option_labels = {item.component_id: f"{item.component_id} · {item.component_type}" for item in components}
            selected_id = st.session_state.get(BOARD_COMPONENT_SELECTION_KEY)
            if selected_id not in component_ids:
                selected_local_id = getattr(selection, "selected_component", None)
                selected_id = (component_id_aliases or {}).get(selected_local_id) if selected_local_id else None
            if selected_id not in component_ids:
                selected_id = component_ids[0]
            target_key = f"cc_measurement_target_{project.project_id}"
            sync_key = f"cc_measurement_target_synced_{project.project_id}"
            if (target_key not in st.session_state
                    or st.session_state.get(sync_key) != st.session_state.get(BOARD_COMPONENT_SELECTION_KEY)):
                st.session_state[target_key] = selected_id
                st.session_state[sync_key] = st.session_state.get(BOARD_COMPONENT_SELECTION_KEY)
            st.selectbox(
                "측정 대상 소자", component_ids, index=component_ids.index(selected_id),
                format_func=option_labels.get,
                key=target_key, on_change=_handle_measurement_target_change,
                kwargs={"state": st.session_state, "project": project,
                        "registry": board_registry, "widget_key": target_key},
            )
        render_measurement_panel(
            st, project, selected_component=selected, preview=preview,
            component_map_version=(board_registry.get("version") if isinstance(board_registry, dict)
                                   else st.session_state.get(MAP_VERSION_KEY)),
            board_registry=board_registry,
        )
        measurements = get_measurements(
            st.session_state, project_id=project.project_id, board_id=project.board_id or "",
        )
        with st.expander("회로 모델 기반 진단 · 고급", expanded=False):
            render_diagnosis_panel(
                st, project, measurements=measurements,
                analysis_context_id=getattr(selection, "analysis_context_id", None),
                component_map_version=st.session_state.get(MAP_VERSION_KEY),
            )

        st.caption("PCB Viewer는 왼쪽에 고정되어 있으며, 측정 대상 변경은 같은 보드 소자 선택에 반영됩니다.")


def _render_cause_dashboard(
    st: Any, project: ProjectContext, result: Any, *,
    analysis_snapshot: dict[str, Any] | None = None,
    board_registry: dict[str, Any] | None = None,
) -> None:
    """Compact, evidence-aware view of the existing symptom diagnosis result."""
    if not isinstance(result, dict):
        reason = (analysis_snapshot or {}).get("record", {}).get("reason")
        if reason:
            st.warning(f"현재 입력에 대한 분석 결과가 없습니다 · {reason} · 증상 입력 탭에서 다시 분석하세요.")
        else:
            st.info("증상 입력 탭에서 증상을 기록하고 원인 분석을 실행하세요.")
        return
    strategy_key = f"cc_candidate_strategy_{project.project_id}"
    strategy_options = ["baseline", "evidence_v1"]
    if isinstance(result.get("candidate_fusion"), dict):
        strategy_options.append("candidate_fusion_v1")
    strategy_labels = {"baseline": "Baseline 규칙 순위", "evidence_v1": "evidence_v1 실험 순위",
                       "candidate_fusion_v1": "Qwen 통합 실험 순위"}
    current_strategy = st.session_state.get(strategy_key, "baseline")
    if current_strategy not in strategy_options:
        current_strategy = "baseline"
    st.selectbox("후보 표시 전략", strategy_options, key=strategy_key,
                 index=strategy_options.index(current_strategy), format_func=strategy_labels.get,
                 help="증상 분석 탭과 같은 전략을 사용합니다.")
    selected_strategy = st.session_state.get(strategy_key, "baseline")
    candidates = candidate_rows_for_strategy(result, selected_strategy)
    groups = result.get("hypothesis_groups", {})
    next_tests = result.get("next_tests", [])
    next_action = next((row for row in next_tests if row.get("type") != "question"), None)
    if next_action is None:
        next_action = next(iter(next_tests), None)
    with st.container(border=True):
        st.markdown(f"**소자 검사 후보 · {strategy_labels.get(selected_strategy, selected_strategy)} · Top 3**")
        if candidates:
            status_labels = {"SUSPECTED": "검사 전 의심", "FAULT_SUPPORTED": "특정 가설 지지",
                             "FAULT_REFUTED": "특정 가설 반박", "CONFLICTED": "근거 충돌",
                             "INCONCLUSIVE": "판정 불가"}
            for index, candidate in enumerate(candidates[:3], start=1):
                component_id = str(candidate.get("board_component_id") or candidate.get("component_id") or "")
                fault_id = str(candidate.get("fault_hypothesis") or candidate.get("fault_id") or "")
                status = str(candidate.get("visual_status") or candidate.get("status") or "SUSPECTED")
                st.markdown(f"**{index}. `{component_id or '소자 미특정'}` · {candidate.get('component_type', '종류 미확정')}**")
                source_labels = candidate.get("candidate_source", [])
                source_text = ("규칙 기반" if "RULE_BASELINE" in source_labels else
                               "Qwen 미검증" if "QWEN_UNVERIFIED" in source_labels else
                               "사용자 의심 단서" if "USER_SUSPICION" in source_labels else
                               str(candidate.get("verification_status", "미확정")))
                verification = (" · 소자 확인 필요" if candidate.get("verification_status") == "NEEDS_COMPONENT_REVIEW" else "")
                st.caption(f"{candidate.get('fault_title', fault_id or '원인 후보')} · {status_labels.get(status, '미확정')} · {source_text}{verification} · 순위는 검사 우선순위이지 고장 확률이 아닙니다.")
                supporting = candidate.get("supporting_evidence") or candidate.get("evidence", [])
                if supporting:
                    st.caption("지지 · " + " · ".join(supporting[:3]))
                opposing = candidate.get("opposing_evidence", [])
                if opposing:
                    st.caption("반대 · " + " · ".join(opposing[:3]))
                if component_id and board_registry and any(
                    item.get("board_component_id") == component_id
                    for item in board_registry.get("components", [])
                ):
                    if st.button("PCB에서 위치 보기", key=f"cc_analysis_dashboard_focus_{project.project_id}_{component_id}",
                                 type="primary" if index == 1 else "secondary", use_container_width=True):
                        _focus_board_registry_component(st, project, board_registry, component_id)
                        st.rerun()
                elif component_id and board_registry is None and any(
                    item.component_id == component_id
                    for item in st.session_state.get("circuitcare_v4_current_components", ())
                ):
                    if st.button("PCB에서 위치 보기", key=f"cc_analysis_dashboard_focus_{project.project_id}_{component_id}",
                                 type="primary" if index == 1 else "secondary", use_container_width=True):
                        update_selection(st.session_state, project=project, selected_component=component_id)
                        st.rerun()
        else:
            st.info("확인된 소자 후보가 없습니다. 증상·검수 결과를 확인하고 기판·납땜 또는 외부 시스템 원인을 살펴보세요.")
        if selected_strategy == "candidate_fusion_v1":
            additional = result.get("candidate_fusion", {}).get("additional_qwen_candidates", [])
            if additional:
                with st.expander(f"추가 확인 후보 · Qwen 미검증 {len(additional)}개", expanded=False):
                    for candidate in additional:
                        cid = str(candidate.get("board_component_id") or candidate.get("component_id") or "")
                        st.caption(f"{cid} · {candidate.get('fault_title', candidate.get('fault_id'))} · 미검증")
                        if candidate.get("qwen_reason"):
                            st.caption(candidate["qwen_reason"])
                        if cid and st.button("PCB 위치 보기", key=f"cc_qwen_extra_{project.project_id}_{cid}"):
                            _focus_board_registry_component(st, project, board_registry, cid)
                            st.rerun()
        if next_action:
            st.info("다음 검사 · " + str(next_action.get("label") or next_action.get("name") or next_action.get("question") or next_action.get("test_id") or "추가 확인"))

    category_labels = (("component", "소자 내부·기능 블록"), ("board", "기판·납땜"), ("system", "외부 시스템"))
    for group_key, label in category_labels:
        rows = groups.get(group_key, []) if isinstance(groups, dict) else []
        if not rows:
            continue
        st.markdown(f"#### {label}")
        for index, row in enumerate(rows[:5], start=1):
            with st.expander(f"{index}. {row.get('title', '원인 후보')} · 미확정", expanded=index == 1):
                st.caption("지지 근거 · " + (" · ".join(row.get("supporting_evidence", [])) or "직접 근거 없음"))
                st.caption("반대 근거 · " + (" · ".join(row.get("opposing_evidence", [])) or "확인된 반대 근거 없음"))
                if row.get("related_component_ids"):
                    st.caption("관련 소자 · " + " · ".join(row["related_component_ids"]))
                st.caption("검증 상태 · 미확정 가설")

    if result.get("llm_assist") or result.get("llm_component_explanations") or result.get("llm_unverified_suggestions") or result.get("matched_rules") or result.get("functional_blocks") or result.get("measurement_evidence"):
        with st.expander("상세 근거 · LLM 설명 · 적용 규칙 · 측정", expanded=False):
            assist = result.get("llm_assist")
            if assist:
                st.markdown("**LLM 설명 보조**")
                st.write(assist.get("summary", ""))
                st.caption(f"모델 · {assist.get('model', '확인 불가')} · 후보 확정 없음")
            elif result.get("llm_error"):
                st.warning(f"LLM 설명을 사용할 수 없습니다 · {result['llm_error']} · 규칙 기반 결과는 유지됩니다.")
            if result.get("llm_component_explanations"):
                explanations = result["llm_component_explanations"].get("explanations", {})
                for component_id, explanation in explanations.items():
                    st.caption(f"{component_id} · {explanation}")
            if result.get("matched_rules"):
                st.caption("적용 규칙 · " + " · ".join(row.get("name", "규칙") for row in result["matched_rules"]))
            if result.get("functional_blocks"):
                for block in result["functional_blocks"]:
                    st.caption(f"기능 역할 추정 · {block.get('component_id')} · {', '.join(block.get('candidate_blocks', [])) or '미분류'} · 미확정")
            if result.get("measurement_evidence"):
                for row in result["measurement_evidence"]:
                    st.caption(f"측정 근거 · {row.get('component_id')} · {row.get('measurement_type')} {row.get('value')} {row.get('unit')} · {row.get('verification_status')}")
            for suggestion in result.get("llm_unverified_suggestions", []):
                st.caption(f"미등록 원인 제안 · {suggestion.get('title')} · {suggestion.get('reason')}")
    questions = [row for row in next_tests if row.get("type") == "question"]
    if questions:
        st.markdown("#### 추가 확인 질문")
        st.caption("질문 답변과 검사 추천은 증상 분석 패널에서 입력·반영합니다.")
        for row in questions[:3]:
            st.caption("· " + row.get("question", "추가 정보 확인"))
    if analysis_snapshot is not None and isinstance(board_registry, dict):
        with st.expander("진단 보고서 · JSON / Markdown / HTML / Evidence Bundle", expanded=False):
            snapshot = build_pcb_diagnostic_snapshot(
                root=Path(__file__).resolve().parents[4], project=project,
                registry=board_registry, analysis_snapshot=analysis_snapshot,
                photos=st.session_state.get(INTAKE_IMAGES_KEY, []),
                selected_board_component_id=st.session_state.get(BOARD_COMPONENT_SELECTION_KEY),
            )
            render_report_panel(st, snapshot=snapshot,
                                root=Path(__file__).resolve().parents[4],
                                workspace_id=project.project_id,
                                expected_circuit_hash=None)
def _render_top_cause(st: Any, project: ProjectContext, result: Any) -> None:
    candidates = result.get("component_candidates", []) if isinstance(result, dict) else []
    next_tests = result.get("next_tests", []) if isinstance(result, dict) else []
    top = candidates[0] if candidates else None
    next_test = next((item for item in next_tests if item.get("type") != "question"), None)
    if next_test is None:
        next_test = next(iter(next_tests), None)
    with st.container(border=True):
        st.markdown("**우선 확인할 내용**")
        if not top:
            st.info("증상을 입력하고 원인 후보 분석을 실행하면 우선 후보와 다음 검사 항목이 표시됩니다.")
            return
        st.markdown(f"**{top.get('component_id', '소자 미특정')} · {top.get('component_type', '종류 미확정')}**  ")
        st.write(top.get("fault_title", "원인 후보"))
        st.caption("미확정 검사 가설 · 상대 점수는 실제 고장 확률을 뜻하지 않습니다.")
        if top.get("evidence"):
            st.caption("근거 · " + " · ".join(top["evidence"][:3]))
        if next_test:
            st.caption("다음 검사 · " + str(next_test.get("label") or next_test.get("name") or next_test.get("question") or next_test.get("test_id") or "추가 확인"))
        if st.button("PCB에서 후보 위치 보기", key=f"cc_analysis_focus_{project.project_id}_{top.get('component_id')}", use_container_width=True):
            update_selection(st.session_state, project=project, selected_component=top.get("component_id"))
            st.rerun()


def _render_schematic(st: Any, project: ProjectContext, circuit_bytes: bytes | None = None) -> None:
    source = project.circuit_source
    if not source:
        st.info("회로도 없음")
        st.caption("설계 자료 추가")
        return
    if source.get("source_type") == "kicad_schematic":
        st.info("KiCad 회로도 자료가 있습니다.")
        raw = circuit_bytes
        if not raw:
            st.warning("회로도 파일 데이터를 찾을 수 없습니다. 입력을 다시 확인하세요.")
            return
        try:
            pages = _export_schematic_pages(raw, source.get("filename") or "uploaded.kicad_sch")
        except (OSError, ValueError) as exc:
            st.warning(f"회로도 Viewer를 열 수 없습니다 · {exc}")
            return
        if not pages:
            st.warning("KiCad SVG Export 결과가 없습니다.")
            return
        page_names = [page[0] for page in pages]
        page_name = st.selectbox("회로도 페이지", page_names, key="circuitcare_v2_schematic_page") if len(pages) > 1 else page_names[0]
        _, svg, view_box = next(page for page in pages if page[0] == page_name)
        context_key = hashlib.sha256(raw).hexdigest()
        rendered = render_interactive_viewer(
            st, svg=svg, asset_key=f"f2-{context_key}-{page_name}", view_box=view_box,
            geometry_status="UNVERIFIED_GEOMETRY", regions=[], overlay=None,
            selected_ref="", navigation_id=0, context_key=context_key,
            event_sequence_seed=0, key=f"cc-v2-schematic-{context_key[:12]}",
            on_selection_change=lambda *_: None,
        )
        if rendered is None:
            st.html(svg, width="stretch")
        st.caption("KiCad 원본 SVG · PCB 정합 및 진단 미실행")
    elif source.get("source_type") == "kicad_xml_netlist":
        st.info("회로도 레이아웃 없음")
        st.caption("XML Netlist는 연결 데이터이며 원본 회로도 배치를 포함하지 않습니다.")
    else:
        st.info("지원되지 않는 설계 자료")


def _export_schematic_pages(raw: bytes, filename: str) -> list[tuple[str, str, tuple[float, float, float, float]]]:
    """Reuse the bounded existing KiCad SVG exporter inside an ephemeral project root."""
    if not raw or len(raw) > 20_000_000:
        raise ValueError("회로도 파일 크기를 확인해 주세요.")
    project_root = Path(__file__).resolve().parents[4]
    with tempfile.TemporaryDirectory(prefix="cc-f2-schematic-", dir=project_root) as temporary:
        temp_root = Path(temporary)
        source_path = temp_root / Path(filename).name
        if source_path.suffix != ".kicad_sch":
            source_path = source_path.with_suffix(".kicad_sch")
        source_path.write_bytes(raw)
        result = export_schematic(
            source_path, project_root=temp_root, cache_root=temp_root / "svg-cache",
        )
        if result.status != "EXPORTED":
            raise ValueError(result.message)
        pages = []
        for svg_path in result.svg_paths:
            view_box = read_svg_view_box(svg_path)
            if view_box is None:
                continue
            pages.append((svg_path.name, svg_path.read_text(encoding="utf-8"), view_box))
        return pages
