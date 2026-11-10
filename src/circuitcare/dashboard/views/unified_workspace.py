"""PCB-first F2 workbench shell with real intake images and an evidence-neutral panel."""

from __future__ import annotations

import hashlib
import base64
import io
from dataclasses import replace
from pathlib import Path
import tempfile
from typing import Any

from circuitcare.frontend.models import ProjectContext
from circuitcare.symptom_diagnosis.diagnosis_state import (
    component_inventory_version, current_overlay_candidates,
    current_verified_component_history,
)
from circuitcare.frontend.component_overlay_adapter import (
    adapt_component_map_components, adapt_original_photo_components,
)
from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.frontend.diagnosis_adapter import (
    DiagnosisStatus, adapt_fault_overlay, assess_diagnosis_readiness,
    current_diagnosis_context, is_diagnosis_result_current,
)
from circuitcare.frontend.measurement_models import measurement_set_sha256
from circuitcare.frontend.measurement_state import diagnosis_state, get_measurements
from circuitcare.frontend.component_map_adapter import (
    COMPONENT_MAP_STATE_KEY, LOGGER, REGISTRATION_LABELS,
    clear_component_map_state,
    component_map_cache_key, component_map_entry_is_current, decode_canonical_image,
    map_backend_photo_ids, optional_backend_provider, photo_input_fingerprint,
    validate_component_map,
)
from circuitcare.frontend.photo_adapter import adapt_component_map_input
from circuitcare.dashboard.components.pcb_workspace import (
    OVERLAPPING_SELECTION_KEY, SELECTED_MAP_KEY, render_component_viewer, scale_overlay_to_display,
)
from circuitcare.dashboard.views.inspection_panel import (
    COMPONENT_EDIT_KEY, MAP_VERSION_KEY, render_inspection_panel,
    _diagnosis_status_label,
)
from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.vision.component_recognition_panel import (
    ensure_component_detection, render_component_recognition_panel,
    review_keys, review_scope as component_review_scope,
)
from circuitcare.vision.box_reconciliation import board_visible_review_rows, exclude_review_row
from circuitcare.frontend.state import (
    INTAKE_IMAGES_KEY, INTAKE_PROJECT_KEY, clear_intake_draft,
    reset_intake_for_new_project,
)
from circuitcare.frontend.workbench_state import (
    ACTIVE_PHOTO_KEY, MODE_KEY, SCREEN_KEY, SELECTION_KEY, VIEW_ZOOM_KEY,
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
    [data-testid="stVerticalBlockBorderWrapper"]:has(.cc-inspection-marker) { position:sticky; top:1.25rem; align-self:start; }
    [class*="st-key-cc_viewer_canvas"] { overflow:auto; max-height:calc(100vh - 15rem); background:#101820; border:1px solid #334252; border-radius:8px; padding:.5rem; }
    div[role="radiogroup"] label { border:1px solid #334252; border-radius:6px; padding:.35rem .8rem; }
    @media (max-width:900px) { [data-testid="stMainBlockContainer"] { padding:.8rem; } }
    @media (max-width:900px) {
      .st-key-cc_workbench_body [data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
      .st-key-cc_workbench_body [data-testid="column"] { flex:1 1 100%; width:100%; min-width:100%; }
      [data-testid="stVerticalBlockBorderWrapper"]:has(.cc-inspection-marker) { position:static; }
    }
    @media (max-width:700px) { [class*="st-key-cc_viewer_canvas"] { max-height:60vh; } }
    </style>""", unsafe_allow_html=True)


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
    selection = st.session_state.get(SELECTION_KEY)
    if getattr(selection, "selected_component", None) and st.session_state.get(SELECTED_MAP_KEY) != cache_key:
        update_selection(st.session_state, project=project, selected_component=None)
        st.session_state.pop(SELECTED_MAP_KEY, None)

    if provider is None:
        st.caption("백엔드 미연결 · 원본 사진 보기")
    elif not ready_inputs:
        st.caption("정합 가능한 사진 없음 · 원본 사진 보기")
    else:
        if st.button("사진 정합 실행", key="circuitcare_v3_build_map"):
            try:
                if getattr(st.session_state.get(SELECTION_KEY), "selected_component", None):
                    update_selection(st.session_state, project=project, selected_component=None)
                    st.session_state.pop(SELECTED_MAP_KEY, None)
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
    selection_context = st.session_state.get(SELECTION_KEY)
    live_context = current_diagnosis_context(
        st.session_state, project_id=project.project_id, board_id=project.board_id or "",
        analysis_context_id=getattr(selection_context, "analysis_context_id", None),
        component_map_version=st.session_state.get(MAP_VERSION_KEY),
    )
    scoped_diagnosis = diagnosis_state(
        st.session_state, project_id=project.project_id, board_id=project.board_id or "",
    )
    board_measurements = get_measurements(
        st.session_state, project_id=project.project_id, board_id=project.board_id or "",
    )
    current_measurement_set = measurement_set_sha256(
        board_measurements, project_id=project.project_id, board_id=project.board_id or "",
    )
    live_readiness = assess_diagnosis_readiness(
        context=live_context, measurements=board_measurements,
        provider_available=st.session_state.get("circuitcare_v5_diagnosis_provider") is not None,
    )
    result_is_current = (
        scoped_diagnosis.get("status") in {DiagnosisStatus.DIAGNOSIS_COMPLETE, DiagnosisStatus.MODEL_MISMATCH}
        and is_diagnosis_result_current(
            stored_result=scoped_diagnosis, current_context=live_context,
            current_measurement_set=current_measurement_set,
        )
    )
    if result_is_current:
        result_status = scoped_diagnosis.get("status")
        display_status = "MODEL_MISMATCH" if result_status == DiagnosisStatus.MODEL_MISMATCH else "DIAGNOSIS_RESULT_NONE"
    elif scoped_diagnosis.get("result") is not None:
        display_status = "DIAGNOSIS_UNAVAILABLE"
    elif live_readiness.status not in {DiagnosisStatus.NO_MEASUREMENT, DiagnosisStatus.DIAGNOSIS_READY}:
        display_status = "DIAGNOSIS_UNAVAILABLE"
    else:
        display_status = "DIAGNOSIS_NOT_RUN"
    component_models = tuple(replace(item, diagnosis_display_status=display_status) for item in component_models)
    if mode == "통합 이미지" and component_map and side_enum and canonical and decoded_canonical:
        stored_diagnosis = scoped_diagnosis.get("result")
        if (result_is_current and isinstance(stored_diagnosis, dict)):
            overlay_rows, overlay_error = adapt_fault_overlay(
                {
                    "diagnosis": stored_diagnosis,
                    "fault_overlay": stored_diagnosis.get("fault_overlay"),
                    "canonical_frame_id": stored_diagnosis.get("canonical_frame_id"),
                    "component_map_version": stored_diagnosis.get("component_map_version"),
                    "confirmed_component_ids": stored_diagnosis.get("confirmed_component_ids", []),
                },
                context=live_context,
                measurement_set_sha256=current_measurement_set,
                board_side=side_enum, canonical_frame_id=decoded_canonical.frame_id,
                component_ids={item.component_id for item in component_models},
            )
            if overlay_error:
                st.caption("진단 위치가 현재 이미지에 확인되지 않아 고장 Overlay를 표시하지 않았습니다.")
            elif overlay_rows:
                component_models = tuple(replace(
                    item, fault_candidate_status=overlay_rows[item.component_id]["fault_candidate_status"],
                    diagnosis_display_status=(
                        "MODEL_MISMATCH" if scoped_diagnosis.get("status") == DiagnosisStatus.MODEL_MISMATCH
                        else overlay_rows[item.component_id]["fault_candidate_status"]
                    ),
                ) if item.component_id in overlay_rows else item for item in component_models)
    if mode == "원본 사진" and detector_image_bytes is not None:
        review_hash = hashlib.sha256(detector_image_bytes).hexdigest()[:16]
        review_scope = component_review_scope(
            detector_image_bytes, project.project_id, selected,
        )
        result_key, records_key, _error_key = review_keys(review_scope)
        st.session_state["cc_active_component_review_scope"] = review_scope
        st.session_state["cc_active_component_review_image"] = detector_image_bytes
        review_rows = st.session_state.get(records_key, [])
        visible_review_rows = [(index, row) for index, row in enumerate(board_visible_review_rows(review_rows))]
        auto_models = tuple(ComponentOverlayViewModel(
            project_id=project.project_id,
            board_id=project.board_id or "",
            board_side=side_enum or active_side,
            component_id=str(row.get("component_id") or f"auto-{review_hash}-{index + 1}"),
            component_type=str(row.get("소자 종류", "unknown")),
            display_label=(f"AI 제안 · {row.get('소자 종류', 'unknown')}" if row.get("source") == "qwen_suggestion"
                           else f"YOLO 2차 · {row.get('소자 종류', 'unknown')}" if row.get("source") == "yolo_second_pass"
                           else f"수동 · {row.get('소자 종류', 'unknown')}" if row.get("source") == "manual"
                           else f"{index + 1} · {row.get('소자 종류', 'unknown')}"),
            canonical_frame_id=selected,
            coordinate_frame="SOURCE_PHOTO_PIXEL",
            bbox_xyxy=(float(row["x1"]), float(row["y1"]),
                       float(row["x2"]), float(row["y2"])),
            polygon_xy=(),
            source_backend_photo_ids=(selected,),
            observation_ids=(),
            detection_confidence=float(row.get("신뢰도", 0.0)),
            identification_status="USER_CONFIRMED" if row.get("상태") == "확인" else "SUGGESTED_CLASS_ONLY",
            identity_match_status=None,
            # Nearby board silkscreen is reference text, not a marking on the body.
            ocr_marking=row.get("body_marking"),
            possible_part_number=None,
            evidence=("Qwen 미확인 박스 제안" if row.get("source") == "qwen_suggestion"
                      else "YOLO 저신뢰 2차 탐색 후보" if row.get("source") == "yolo_second_pass"
                      else "사용자 추가 박스" if row.get("source") == "manual"
                      else "YOLO component proposal; awaiting user review",),
            overlay_status="EXCLUDED" if row.get("상태") == "제외" else "CANDIDATE",
            diagnosis_display_status="DIAGNOSIS_NOT_RUN",
            location_confidence=float(row.get("location_confidence", row.get("신뢰도", 0.0))),
            type_confidence=row.get("type_confidence"), match_score=row.get("match_score"),
            component_type_status="USER_CONFIRMED" if row.get("상태") == "확인" else "UNCONFIRMED",
            ocr_status=row.get("ocr_association_status"),
            ocr_candidates=tuple(row.get("ocr_candidates", [])),
            ocr_class_hints=tuple(row.get("ocr_class_hints", [])),
            reference_designators=tuple(row.get("reference_designators", [])),
            overlay_source=row.get("source", "yolo"),
        ) for index, row in visible_review_rows)
        component_models = tuple(component_models) + auto_models
        excluded_component_ids = {str(row.get("component_id")) for row in review_rows
                                  if row.get("상태") == "제외" and row.get("component_id")}
        component_models = tuple(item for item in component_models
                                 if item.component_id not in excluded_component_ids)
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
                component_type = edit.get("component_type", model.component_type)
                model = replace(
                    model,
                    component_type=component_type,
                    display_label=f"{model.component_id.rsplit('-', 1)[-1]} · {component_type}",
                    ocr_marking=edit.get("ocr_marking", model.ocr_marking) or None,
                    possible_part_number=(
                        edit.get("ocr_marking", model.possible_part_number)
                        if component_type.casefold() in {"ic", "integrated_circuit"}
                        else model.possible_part_number
                    ),
                    component_info=edit.get("component_info", model.component_info) or None,
                )
            updated_models.append(model)
        component_models = tuple(updated_models)
    candidate_state_by_project = st.session_state.get("circuitcare_v4_diagnosis_candidates_by_project", {})
    candidate_state = current_overlay_candidates(
        candidate_state_by_project.get(project.project_id) if isinstance(candidate_state_by_project, dict) else {},
        project_id=project.project_id, board_id=project.board_id,
        component_map_version=st.session_state.get(MAP_VERSION_KEY),
        input_fingerprint=st.session_state.get(f"cc_symptom_diagnosis_{project.project_id}", {}).get("input_fingerprint"),
    )
    verified_history = current_verified_component_history(
        candidate_state_by_project.get(project.project_id) if isinstance(candidate_state_by_project, dict) else {},
        project_id=project.project_id, board_id=project.board_id,
        component_map_version=st.session_state.get(MAP_VERSION_KEY),
        input_fingerprint=st.session_state.get(f"cc_symptom_diagnosis_{project.project_id}", {}).get("input_fingerprint"),
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
                    fault_candidate_status=status_map.get(candidate_state[model.component_id].get("visual_status"),
                                                          ("INCONCLUSIVE", "판정 불가"))[0],
                    diagnosis_display_status=status_map.get(candidate_state[model.component_id].get("visual_status"),
                                                            ("INCONCLUSIVE", "판정 불가"))[1],
                    fault_hypothesis=candidate_state[model.component_id].get("fault_hypothesis"),
                    fault_role_status=candidate_state[model.component_id].get("role_status"),
                    measurement_status=candidate_state[model.component_id].get("measurement_status"),
                    solder_status=candidate_state[model.component_id].get("solder_status"),
                    fault_score=candidate_state[model.component_id].get("score"),
                    fault_evidence=tuple(candidate_state[model.component_id].get("evidence", [])),
                    recommended_tests=tuple(candidate_state[model.component_id].get("recommended_tests", [])),
                    fault_history=tuple(history_by_id.get(model.component_id, [])))
            if model.component_id in candidate_state else model
            for model in component_models
        )
    if history_by_id:
        component_models = tuple(
            replace(
                model,
                fault_history=tuple(history_by_id[model.component_id]),
                fault_candidate_status=(
                    history_by_id[model.component_id][-1].get("visual_status")
                    if model.component_id not in candidate_state else model.fault_candidate_status
                ),
                diagnosis_display_status=(
                    history_by_id[model.component_id][-1].get("visual_status")
                    if model.component_id not in candidate_state else model.diagnosis_display_status
                ),
            )
            if model.component_id in history_by_id and model.component_id not in candidate_state else model
            for model in component_models
        )
    st.session_state["circuitcare_v4_current_components"] = tuple(component_models)
    st.session_state["circuitcare_v4_detection_status"] = detection_status
    current_selection = st.session_state.get(SELECTION_KEY)
    selected_component_id = getattr(current_selection, "selected_component", None)
    if selected_component_id and selected_component_id not in {item.component_id for item in component_models}:
        update_selection(st.session_state, project=project, selected_component=None)
        st.session_state.pop(SELECTED_MAP_KEY, None)
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
        from circuitcare.vision.component_recognition_panel import invalidate_qwen_review
        invalidate_qwen_review(st.session_state, review_scope)
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
    if candidate_state:
        st.caption("진단 색상 · 노랑: 검사 후보 · 빨강: 해당 가설 측정 이상 · 초록: 해당 검사 정상 · 회색: 판정 불가. "
                   "초록은 소자 전체의 정상 판정을 뜻하지 않습니다.")
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
    if overlay_result and overlay_result.status == "READY":
        st.caption(f"검출 결과 · {len(overlay_result.components)}개 · 후보 · 고장 진단 미실행")
    elif auto_detection_error:
        st.error(f"자동 소자 탐지에 실패했습니다 · {auto_detection_error}")
    elif auto_detection_result is not None and not auto_detection_result.get("detections"):
        st.info("이 사진에서 선택할 탐지 박스를 찾지 못했습니다.")
    elif auto_detection_result is not None:
        st.caption(f"탐지 박스 · {len(auto_detection_result.get('detections', []))}개")
    elif detection_status == "DETECTOR_NOT_CONFIGURED":
        st.caption("검출 모델 미설정 · 원본 사진 보기 가능")
    elif detection_status == "BLOCKED_MODEL_UNAVAILABLE":
        st.caption("검출 모델을 사용할 수 없습니다 · 원본 사진 보기 가능")
    elif component_map and not side_result.get("components"):
        st.caption("검출 결과 없음 · 고장 진단 미실행")
    if mode == "원본 사진":
        mapped = entry.get("photo_id_map", []) if entry else []
        photo_mapping = next((row for row in mapped if row.frontend_photo_id == selected and row.project_id == project.project_id), None)
        st.caption(f"정합 상태 · {REGISTRATION_LABELS.get(photo_mapping.registration_status, '상태 미확인') if photo_mapping else '정합 결과 없음'}")
        with st.expander("소자 탐지 · 박스 검수", expanded=True):
            render_component_recognition_panel(
                st, image_bytes=detector_image_bytes or image_data,
                project_id=project.project_id, photo_id=selected,
                auto_result=auto_detection_result, auto_error=auto_detection_error,
            )


def _photo_label(record: Any, number: int) -> str:
    role = {
        "full_view": "전체", "close_up": "확대", "oblique": "비스듬히",
        "marking_detail": "부품 표기", "unknown": "미지정",
    }[record.capture_type]
    return f"사진 {number} · {role}"


def _exclude_selected_review_box(rows: list[dict[str, Any]], target: ComponentOverlayViewModel) -> dict[str, Any]:
    """Persist an excluded visible box through the existing component review record."""
    row = next((item for item in rows if str(item.get("component_id")) == target.component_id), None)
    if row is None:
        x1, y1, x2, y2 = target.bbox_xyxy
        row = {"component_id": target.component_id, "소자 종류": target.component_type,
               "상태": "검수 필요", "신뢰도": float(target.detection_confidence or 0.0),
               "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
               "source": target.overlay_source or "component_map"}
        rows.append(row)
    exclude_review_row(row, reason="user_box_exclusion")
    return row


def _render_selected_box_actions(st: Any, project: ProjectContext,
                                 components: tuple[ComponentOverlayViewModel, ...]) -> None:
    """Expose direct box exclusion and overlap selection at the top of inspection."""
    selection = st.session_state.get(SELECTION_KEY)
    selected_id = getattr(selection, "selected_component", None)
    overlap_key = st.session_state.get("cc_active_component_overlap_key")
    overlap_ids = st.session_state.get(overlap_key, []) if overlap_key else []
    by_id = {item.component_id: item for item in components}
    hits = [component_id for component_id in overlap_ids if component_id in by_id]
    if selected_id in by_id and selected_id not in hits:
        hits.insert(0, selected_id)

    if len(hits) > 1:
        labels = {
            f"{component_id} · {by_id[component_id].component_type} · "
            f"{(by_id[component_id].bbox_xyxy[2] - by_id[component_id].bbox_xyxy[0]) * (by_id[component_id].bbox_xyxy[3] - by_id[component_id].bbox_xyxy[1]):.0f}px²": component_id
            for component_id in hits
        }
        label_by_id = {component_id: label for label, component_id in labels.items()}
        current = selected_id if selected_id in hits else hits[0]
        choice = st.selectbox("겹친 박스 선택", list(labels),
                              index=list(labels).index(label_by_id[current]),
                              key=f"cc_box_overlap_select_{st.session_state.get('cc_active_component_review_scope', 'none')}")
        chosen_id = labels[choice]
        if chosen_id != selected_id:
            update_selection(st.session_state, project=project, selected_component=chosen_id)
            st.rerun()

    target = by_id.get(selected_id)
    review_scope = st.session_state.get("cc_active_component_review_scope")
    if target is None or not review_scope:
        return
    if not st.button("박스 제외", key=f"cc_selected_box_exclude_{selected_id}", use_container_width=True):
        return
    _result_key, records_key, _error_key = review_keys(review_scope)
    rows = list(st.session_state.get(records_key, []))
    _exclude_selected_review_box(rows, target)
    st.session_state[records_key] = rows
    from circuitcare.vision.component_recognition_panel import invalidate_qwen_review
    invalidate_qwen_review(st.session_state, review_scope)
    update_selection(st.session_state, project=project, selected_component=None)
    st.rerun()


def render_workbench_shell(st: Any, project: ProjectContext) -> None:
    """Render PCB/schematic tabs and a fixed desktop inspection column."""
    _style(st)
    selection = st.session_state.get(SELECTION_KEY)
    if getattr(selection, "project_id", None) != project.project_id:
        enter_workbench(
            st.session_state, project,
            preview=st.session_state.get(MODE_KEY) == "PREVIEW_ONLY",
        )

    title, add_photo, edit, new = st.columns([5, 1, 1, 1])
    with title:
        st.markdown('<div class="cc-wb-header">CircuitCare AI</div>', unsafe_allow_html=True)
        st.caption(f"{project.project_id[:8]} · 사진 {len(project.image_records)}장 · {_diagnosis_status_label(st, project)}")
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
        left, right = st.columns([3.2, 1.55], gap="medium")
        with left:
            st.session_state.pop("circuitcare_v2_active_tab", None)
            update_selection(
                st.session_state, project=project, active_tab="pcb",
            )
            _render_pcb(st, project, st.session_state.get(INTAKE_IMAGES_KEY, []))
        with right:
            current_components = tuple(st.session_state.get("circuitcare_v4_current_components", ()))
            selection = st.session_state.get(SELECTION_KEY)
            selected_id = getattr(selection, "selected_component", None)
            selected_component = next((item for item in current_components if item.component_id == selected_id), None)
            _render_selected_box_actions(st, project, current_components)
            render_inspection_panel(
                st, project, preview=preview, components=current_components,
                selected_component=selected_component,
                detection_status=st.session_state.get("circuitcare_v4_detection_status", "UNKNOWN"),
                selected_marking_context=st.session_state.get("circuitcare_v4_selected_marking_context"),
                component_map_version=st.session_state.get(MAP_VERSION_KEY),
                inventory_scope=st.session_state.get("circuitcare_v4_component_inventory_scope"),
                review_image_bytes=st.session_state.get("cc_active_component_review_image"),
                review_scope=st.session_state.get("cc_active_component_review_scope"),
            )


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
