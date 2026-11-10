"""PCB inspection summary and selected-component details."""

from __future__ import annotations

import io
import hashlib
import logging
from dataclasses import replace
from typing import Any

from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel, display_identity_status
from circuitcare.frontend.models import ProjectContext
from circuitcare.vision.component_recognition_panel import (
    CLASS_OPTIONS, LAST_EXCLUDED_KEY, STATUS_OPTIONS, ensure_review_row_for_component,
    invalidate_component_review_state, record_box_exclusion, restore_review_row,
    review_keys, undo_last_box_exclusion, update_review_row,
)
from circuitcare.frontend.workbench_state import ACTIVE_PHOTO_KEY, SELECTION_KEY, update_selection

MAP_VERSION_KEY = "circuitcare_v4_selection_map_key"
COMPONENT_EDIT_KEY = "circuitcare_v4_component_edits"
COMPONENT_INVENTORY_KEY = "circuitcare_v4_component_inventory_by_map"
ROLE_OPTIONS = ("power", "control", "control_input", "control_timing", "driver", "output",
                "input", "input_output", "protection", "thermal", "buzzer", "relay")
LOGGER = logging.getLogger(__name__)
def render_inspection_panel(
    st: Any, project: ProjectContext, *, preview: bool = False,
    components: tuple[ComponentOverlayViewModel, ...] = (),
    selected_component: ComponentOverlayViewModel | None = None,
    detection_status: str = "UNKNOWN",
    selected_marking_context: dict[str, Any] | None = None,
    component_map_version: str | None = None,
    inventory_scope: str | None = None,
    review_image_bytes: bytes | None = None,
    review_scope: str | None = None,
) -> None:
    owner = f"{project.project_id}|{project.board_id or ''}|{inventory_scope or component_map_version or ''}"
    inventories = st.session_state.setdefault(COMPONENT_INVENTORY_KEY, {})
    inventory = inventories.setdefault(owner, {})
    with st.container(border=True):
        st.markdown('<span class="cc-inspection-marker"></span>', unsafe_allow_html=True)
        overlap_key = st.session_state.get("cc_active_component_overlap_key")
        overlap_ids = st.session_state.get(overlap_key, []) if overlap_key else []
        by_id = {item.component_id: item for item in components}
        overlap_ids = list(dict.fromkeys(item for item in overlap_ids if item in by_id))
        selection = st.session_state.get(SELECTION_KEY)
        selected_id = getattr(selection, "selected_component", None)
        LOGGER.info("pcb_selection stage=panel_render project=%s selected=%s component_found=%s component_count=%s overlaps=%s",
                    project.project_id, selected_id,
                    selected_component.component_id if selected_component is not None else None,
                    len(components), overlap_ids)
        if len(overlap_ids) > 1:
            labels = {
                f"{item} · {by_id[item].component_type} · "
                f"{(by_id[item].bbox_xyxy[2] - by_id[item].bbox_xyxy[0]) * (by_id[item].bbox_xyxy[3] - by_id[item].bbox_xyxy[1]):.0f}px²": item
                for item in overlap_ids
            }
            selected_option = next((label for label, item in labels.items() if item == selected_id), next(iter(labels)))
            overlap_signature = hashlib.sha1("|".join(overlap_ids).encode()).hexdigest()[:8]
            # Streamlit keeps widget values across reruns. Include the server-side
            # click selection in the key so a prior overlap choice cannot replace
            # the most recent box clicked in the SVG.
            viewer_key = overlap_key.removeprefix("circuitcare_v4_overlapping_box_ids_")
            event_seq = st.session_state.get(f"{viewer_key}_last_seq", 0)
            choice = st.selectbox("겹친 박스", list(labels), index=list(labels).index(selected_option),
                                  key=(f"cc_box_overlap_select_{st.session_state.get('cc_active_component_review_scope', 'none')}"
                                       f"_{overlap_signature}_{selected_id or 'none'}_{event_seq}"),
                                  help="겹친 박스 중 편집하거나 제외할 대상을 선택합니다. Viewer 강조도 함께 바뀝니다.")
            if labels[choice] != selected_id:
                update_selection(st.session_state, project=project, selected_component=labels[choice])
                st.rerun()
        if preview:
            st.markdown("**부품** · 인식 결과 없음")
        elif detection_status in {"DETECTOR_NOT_CONFIGURED", "BLOCKED_MODEL_UNAVAILABLE"}:
            st.markdown("**부품 검출** · 검출 모델 미설정")
        elif detection_status not in {"INFERENCE_COMPLETE", "PARTIAL_MAP_READY"}:
            st.markdown("**부품 검출** · 인식 결과 없음")
        elif not components:
            st.markdown("**인식된 부품** · 없음")
        else:
            st.markdown(f"**인식된 부품** · {len(components)}개")
        active_scope = st.session_state.get("cc_active_component_review_scope") or review_scope
        active_records_key = review_keys(active_scope)[1] if active_scope else None
        active_rows = list(st.session_state.get(active_records_key, [])) if active_records_key else []
        if active_scope:
            undo_key = f"{LAST_EXCLUDED_KEY}_{active_scope}"
            last_excluded = st.session_state.get(undo_key)
            undo_col, restore_col = st.columns(2)
            undo_id = last_excluded.get("component_id") if isinstance(last_excluded, dict) else None
            if undo_col.button("직전 제외 취소", key=f"cc_component_undo_exclusion_{active_scope}",
                               disabled=not bool(undo_id), use_container_width=True,
                               help="가장 최근에 제외한 박스를 복원합니다."):
                restored_id = undo_last_box_exclusion(st.session_state, scope=active_scope, rows=active_rows)
                if restored_id:
                    st.session_state[active_records_key] = active_rows
                    invalidate_component_review_state(st.session_state, scope=active_scope,
                        project_id=project.project_id, board_id=project.board_id)
                    update_selection(st.session_state, project=project, selected_component=restored_id)
                    st.rerun()
            excluded_rows = [row for row in active_rows if row.get("상태") == "제외"]
            if excluded_rows:
                with restore_col.popover(f"제외 복원 · {len(excluded_rows)}",
                                         help="이전에 제외한 박스를 선택해 다시 표시합니다."):
                    for index, row in enumerate(excluded_rows):
                        st.caption(f"{row.get('component_id')} · {row.get('소자 종류', 'unknown')}")
                        if st.button("복원", key=f"cc_restore_box_{active_scope}_{row.get('component_id')}_{index}",
                                     help="이 박스를 다시 활성 상태로 돌리고 Viewer와 진단 목록에 반영합니다."):
                            restore_review_row(row)
                            st.session_state[active_records_key] = active_rows
                            if isinstance(last_excluded, dict) and last_excluded.get("component_id") == row.get("component_id"):
                                st.session_state.pop(undo_key, None)
                            invalidate_component_review_state(st.session_state, scope=active_scope,
                                project_id=project.project_id, board_id=project.board_id)
                            update_selection(st.session_state, project=project,
                                             selected_component=str(row.get("component_id")))
                            st.rerun()
        st.markdown("### 선택 부품")
        if selected_component is None:
            st.caption("PCB에서 박스를 누르면 이곳에 인식 정보가 표시됩니다.")
        else:
            edits = st.session_state.setdefault(COMPONENT_EDIT_KEY, {})
            edit = edits.get(selected_component.component_id, {})
            current_type = selected_component.component_type
            current_marking = edit.get("ocr_marking", selected_component.ocr_marking or "")
            current_value = edit.get("component_info", selected_component.component_info or "")
            st.caption(f"부품 ID · {selected_component.component_id}")
            inventory_item = inventory.setdefault(selected_component.component_id, {
                "presence_status": "UNCONFIRMED", "component_type_status": "UNCONFIRMED",
                "part_identity_status": "UNCONFIRMED", "confirmed_functional_roles": [],
            })
            review_row = None
            records_key = review_keys(review_scope)[1] if review_scope else None
            if review_image_bytes is not None and records_key:
                review_row = next((row for row in st.session_state.get(records_key, [])
                                   if row.get("component_id") == selected_component.component_id), None)
            current_status = review_row.get("상태", "검수 필요") if review_row else (
                "확인" if selected_component.presence_status == "CONFIRMED" else "검수 필요")
            inventory_item["presence_status"] = (
                "CONFIRMED" if current_status == "확인" else "EXCLUDED" if current_status == "제외" else "UNCONFIRMED")
            inventory_item["component_type_status"] = "USER_CONFIRMED" if current_status == "확인" else "UNCONFIRMED"
            st.caption(f"상태 · {current_status} · 검수 레코드와 동기화됨")
            if review_row is not None and review_row.get("상태") == "제외":
                st.warning("제외된 박스입니다. 복원 전까지 진단과 기본 Viewer에서 제외됩니다.")
            elif review_scope and records_key:
                if st.button("박스 제외", key=f"cc_selected_box_exclude_{selected_component.component_id}",
                             type="primary", use_container_width=True,
                             help="선택한 박스를 기본 Viewer와 진단 대상에서 숨깁니다. 검수 기록은 보존되며 직전 제외 취소 또는 제외 복원으로 되돌릴 수 있습니다."):
                    rows = list(st.session_state.get(records_key, []))
                    review_row = ensure_review_row_for_component(rows, selected_component)
                    record_box_exclusion(st.session_state, scope=review_scope, row=review_row)
                    st.session_state[records_key] = rows
                    invalidate_component_review_state(st.session_state, scope=review_scope,
                        project_id=project.project_id, board_id=project.board_id)
                    update_selection(st.session_state, project=project, selected_component=None)
                    st.rerun()
            if review_scope and records_key and review_image_bytes is not None:
                box = review_row or {"x1": selected_component.bbox_xyxy[0], "y1": selected_component.bbox_xyxy[1],
                                     "x2": selected_component.bbox_xyxy[2], "y2": selected_component.bbox_xyxy[3]}
                row_signature = (current_type, current_status,
                                 int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"]))
                form_version = hashlib.sha1(repr(row_signature).encode()).hexdigest()[:8]
                form_key = f"cc_review_box_editor_{review_scope}_{selected_component.component_id}_{form_version}"
                with st.form(form_key):
                    st.selectbox("소자 종류", CLASS_OPTIONS,
                                 index=CLASS_OPTIONS.index(current_type if current_type in CLASS_OPTIONS else "unknown"),
                                 key=f"{form_key}_class",
                                 help="검수 결과의 소자 종류를 바꾸면 Viewer와 진단 소자 목록에도 반영됩니다.")
                    editable_statuses = [value for value in STATUS_OPTIONS if value != "제외"]
                    status_value = current_status if current_status in editable_statuses else "검수 필요"
                    st.selectbox("검수 상태", editable_statuses, index=editable_statuses.index(status_value),
                                 key=f"{form_key}_status",
                                 help="확인은 사용자 확인 기록으로 저장됩니다. 제외는 별도 제외 버튼으로 처리합니다.")
                    with st.expander("박스 위치·크기"):
                        cols = st.columns(2)
                        x1 = cols[0].number_input("왼쪽 x", min_value=0, value=int(box["x1"]), key=f"{form_key}_x1",
                                                  help="원본 사진의 픽셀 좌표입니다.")
                        y1 = cols[1].number_input("위쪽 y", min_value=0, value=int(box["y1"]), key=f"{form_key}_y1",
                                                  help="원본 사진의 픽셀 좌표입니다.")
                        x2 = cols[0].number_input("오른쪽 x", min_value=0, value=int(box["x2"]), key=f"{form_key}_x2",
                                                  help="원본 사진의 픽셀 좌표입니다.")
                        y2 = cols[1].number_input("아래쪽 y", min_value=0, value=int(box["y2"]), key=f"{form_key}_y2",
                                                  help="원본 사진의 픽셀 좌표입니다.")
                    apply = st.form_submit_button("변경 적용", use_container_width=True,
                                                   help="종류·상태·좌표를 검수 기록에 저장하고 Viewer와 진단 목록에 반영합니다.")
                if apply:
                    from PIL import Image
                    with Image.open(io.BytesIO(review_image_bytes)) as source_image:
                        width, height = source_image.size
                    coords = [max(0, min(width, int(x1))), max(0, min(height, int(y1))),
                              max(0, min(width, int(x2))), max(0, min(height, int(y2)))]
                    try:
                        rows = list(st.session_state.get(records_key, []))
                        review_row = ensure_review_row_for_component(rows, selected_component)
                        next_status = st.session_state[f"{form_key}_status"]
                        update_review_row(review_row,
                            component_type=st.session_state[f"{form_key}_class"],
                            status=next_status, bbox_xyxy=coords)
                        st.session_state[records_key] = rows
                        inventory_item["presence_status"] = "CONFIRMED" if review_row["상태"] == "확인" else "UNCONFIRMED"
                        inventory_item["component_type_status"] = "USER_CONFIRMED" if review_row["상태"] == "확인" else "UNCONFIRMED"
                        st.session_state[COMPONENT_INVENTORY_KEY] = inventories
                        invalidate_component_review_state(st.session_state, scope=review_scope,
                            project_id=project.project_id, board_id=project.board_id)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
            type_status_options = ("UNCONFIRMED", "USER_CONFIRMED")
            with st.expander("상세 식별·기능 확인", expanded=False):
                identity_key = f"cc_part_identity_status_{hash(owner)}_{selected_component.component_id}"
                if st.session_state.get(identity_key) not in type_status_options:
                    st.session_state[identity_key] = inventory_item.get("part_identity_status", "UNCONFIRMED")
                part_identity_status = st.radio("부품 모델/부품번호 확인", type_status_options,
                    format_func=lambda value: "모델 식별됨" if value == "USER_CONFIRMED" else "모델 미확인",
                    horizontal=True, key=identity_key,
                    on_change=_sync_inventory_widget, args=(st, owner, selected_component.component_id,
                        "part_identity_status", identity_key, project.project_id, project.board_id,
                        str(st.session_state.get(ACTIVE_PHOTO_KEY) or "target:default")))
                inventory_item["part_identity_status"] = part_identity_status
                role_key = f"cc_component_roles_{hash(owner)}_{selected_component.component_id}"
                current_roles = inventory_item.get("confirmed_functional_roles", [])
                if role_key not in st.session_state:
                    st.session_state[role_key] = [role for role in current_roles if role in ROLE_OPTIONS]
                confirmed_roles = st.multiselect(
                    "이 부품에서 확인된 기능 역할", ROLE_OPTIONS, key=role_key,
                    help="직접 확인한 역할만 선택합니다. 선택하지 않은 배치/OCR 추정은 미확정 후보로 유지됩니다.",
                    on_change=_sync_inventory_widget, args=(st, owner, selected_component.component_id,
                        "confirmed_functional_roles", role_key, project.project_id, project.board_id,
                        str(st.session_state.get(ACTIVE_PHOTO_KEY) or "target:default")),
                )
                inventory_item["confirmed_functional_roles"] = list(confirmed_roles)
            inventories[owner] = inventory
            st.session_state[COMPONENT_INVENTORY_KEY] = inventories
            selected_component = replace(
                selected_component,
                component_type=current_type,
                presence_status=inventory_item["presence_status"],
                confirmed_functional_roles=tuple(confirmed_roles),
            )
            if review_image_bytes is not None and review_scope:
                if review_row is not None:
                    marking_result_key = f"cc_component_marking_ocr_{review_scope}_{selected_component.component_id}"
                    with st.expander("선택 부품 각인 OCR", expanded=False):
                        st.caption("각인 인식값은 미확인 후보입니다. OCR만으로 부품 모델이나 종류를 확정하지 않습니다.")
                        if st.button("선택 부품에서 각인 읽기", key=f"cc_component_marking_run_{review_scope}_{selected_component.component_id}"):
                            try:
                                from PIL import Image
                                from circuitcare.vision.ocr_backend import run_component_marking_ocr

                                with Image.open(io.BytesIO(review_image_bytes)) as source_image:
                                    result = run_component_marking_ocr(
                                        source_image.convert("RGB"),
                                        [review_row["x1"], review_row["y1"], review_row["x2"], review_row["y2"]],
                                        current_type,
                                    )
                                st.session_state[marking_result_key] = result
                            except Exception as exc:
                                st.warning(f"각인 OCR을 실행하지 못했습니다 · {exc}")
                        marking_result = st.session_state.get(marking_result_key)
                        if isinstance(marking_result, dict):
                            _render_selected_marking(st, {
                                "image_bytes": review_image_bytes,
                                "bbox_xyxy": [review_row["x1"], review_row["y1"], review_row["x2"], review_row["y2"]],
                                "result": marking_result,
                            })
                            choices = [item.get("text", "") for item in marking_result.get("candidates", []) if item.get("text")]
                            if choices:
                                choice_key = f"cc_component_marking_choice_{review_scope}_{selected_component.component_id}"
                                choice = st.selectbox("OCR 원문 후보 선택", choices, key=choice_key)
                                if st.button("선택한 원문을 편집 후보로 사용", key=f"cc_component_marking_use_{review_scope}_{selected_component.component_id}"):
                                    edits = st.session_state.setdefault(COMPONENT_EDIT_KEY, {})
                                    edits[selected_component.component_id] = {
                                        **edits.get(selected_component.component_id, {}),
                                        "ocr_marking": choice,
                                        "component_info": marking_result.get("inferred_value") or current_value,
                                    }
                                    st.session_state[COMPONENT_EDIT_KEY] = edits
                                    invalidate_component_review_state(st.session_state, scope=review_scope,
                                        project_id=project.project_id, board_id=project.board_id)
                                    st.rerun()
            if selected_component.fault_hypothesis:
                st.caption(f"진단 후보 · {selected_component.diagnosis_display_status or '미확정'} · "
                           f"가설 {selected_component.fault_hypothesis}")
                if selected_component.fault_role_status == "INFERRED":
                    st.caption("기능 역할 추정 · 미확정 후보이며 추가 역할 확인이 필요합니다.")
                elif selected_component.fault_role_status == "CONFIRMED":
                    st.caption("기능 역할 확인됨")
                if selected_component.fault_score is not None:
                    st.caption(f"상대 점수 · {selected_component.fault_score:g} · 확률이 아닌 후보 정렬 점수")
                if selected_component.measurement_status:
                    st.caption(f"측정 판정 · {selected_component.measurement_status} · "
                               "이 검사 가설에만 적용")
                if selected_component.fault_evidence:
                    st.caption("후보 근거 · " + " · ".join(selected_component.fault_evidence[:5]))
                if selected_component.recommended_tests:
                    st.caption("권장 확인 · " + " · ".join(selected_component.recommended_tests[:3]))
            for history_item in selected_component.fault_history[-3:]:
                st.caption(
                    f"검사 이력 · {history_item.get('fault_title', history_item.get('fault_hypothesis', '고장 가설'))} · "
                    f"{history_item.get('measurement_verdict', '판정 기록')}"
                )
            marking_key = f"cc_v4_edit_marking_{selected_component.component_id}"
            value_key = f"cc_v4_edit_value_{selected_component.component_id}"
            if marking_key not in st.session_state:
                st.session_state[marking_key] = current_marking
            if value_key not in st.session_state:
                st.session_state[value_key] = current_value
            with st.expander("인식 정보 편집", expanded=False):
                st.text_input("OCR 인식값 · 각인/부품번호", key=marking_key,
                              placeholder="예: 103, LM358")
                st.text_input("소자 값 · OCR 유추값 또는 직접 입력", key=value_key,
                              placeholder="예: 10 kΩ, 100 nF, 3.3 V")
                if st.button("인식 정보 저장", type="primary", key=f"cc_v4_save_component_{selected_component.component_id}"):
                    edits[selected_component.component_id] = {
                        "ocr_marking": st.session_state[marking_key].strip(),
                        "component_info": st.session_state[value_key].strip(),
                    }
                    st.session_state[COMPONENT_EDIT_KEY] = edits
                    if review_scope:
                        invalidate_component_review_state(st.session_state, scope=review_scope,
                            project_id=project.project_id, board_id=project.board_id)
                    st.rerun()
            if current_marking or current_value:
                st.caption(f"자동 판독 상태 · {display_identity_status(selected_component.identification_status)}")
            if selected_component.reference_designators:
                st.caption("주변 실크스크린 Reference 후보 · " + " · ".join(selected_component.reference_designators))
            if selected_component.ocr_marking:
                st.caption("소자 본체 각인 후보 · " + selected_component.ocr_marking)
            if selected_marking_context and selected_component.component_id.startswith("auto-"):
                _render_selected_marking(st, selected_marking_context)
            if selected_component.identity_match_status:
                st.caption(f"이전 식별 연속성 · {display_identity_status(selected_component.identity_match_status)}")
            st.caption(f"관측 사진 · {len(selected_component.source_backend_photo_ids)}장")
            st.caption("정합 · 후보 · 미검증")
            if selected_component.evidence:
                with st.expander("검출 근거"):
                    st.write(" · ".join(selected_component.evidence))
            if review_row is not None and review_image_bytes is not None and review_scope:
                rows = st.session_state.get(review_keys(review_scope)[1], [])
                excluded_rows = [row for row in rows if row.get("상태") == "제외"]
                if excluded_rows:
                    st.caption(f"제외된 박스 {len(excluded_rows)}개 · 상단의 복원 메뉴에서 되살릴 수 있습니다.")
        # Fault analysis lives in its own full-width workspace. This panel is
        # intentionally limited to PCB component review and editing.


def _diagnosis_status_label(st: Any, project: ProjectContext, *, analysis_snapshot: dict[str, Any] | None = None) -> str:
    if analysis_snapshot is None:
        from circuitcare.dashboard.views.symptom_diagnosis_panel import current_analysis_snapshot

        components = tuple(st.session_state.get("circuitcare_v4_current_components", ()))
        result_scope = st.session_state.get("circuitcare_v4_component_inventory_scope")
        owner = f"{project.project_id}|{project.board_id or ''}|{result_scope or st.session_state.get(MAP_VERSION_KEY) or ''}"
        inventories = st.session_state.get(COMPONENT_INVENTORY_KEY, {})
        inventory = inventories.get(owner, {}) if isinstance(inventories, dict) else {}
        analysis_snapshot = current_analysis_snapshot(
            st, project, components=components, component_inventory=inventory,
        )
    result = analysis_snapshot.get("result")
    if not isinstance(result, dict):
        status = analysis_snapshot.get("record", {}).get("status")
        if status == "RUNNING":
            return "진단 중"
        if status == "STALE":
            return "입력 변경 · 재분석 필요"
        if status == "FAILED":
            return "진단 실패 · 재시도 필요"
        return "진단 미실행"
    status = result.get("status")
    if status == "CANDIDATES_AVAILABLE":
        return f"진단 완료 · 후보 {len(result.get('component_candidates', []))}개"
    if status == "ABSTAIN_INSUFFICIENT_EVIDENCE":
        return "추론 보류 · 추가 정보 필요"
    return "진단 결과 확인 필요"


def _sync_inventory_widget(st: Any, owner: str, component_id: str, field: str, widget_key: str,
                           project_id: str | None = None, board_id: str | None = None,
                           review_scope: str | None = None) -> None:
    inventories = st.session_state.setdefault(COMPONENT_INVENTORY_KEY, {})
    inventory = inventories.setdefault(owner, {})
    item = inventory.setdefault(component_id, {})
    item[field] = st.session_state.get(widget_key)
    st.session_state[COMPONENT_INVENTORY_KEY] = inventories
    if project_id:
        invalidate_component_review_state(st.session_state, scope=review_scope or "target:default",
                                          project_id=project_id, board_id=board_id)


def _render_selected_marking(st: Any, context: dict[str, Any]) -> None:
    """Show the enlarged selected-part crop and the OCR candidates in the inspector."""
    from PIL import Image, ImageEnhance, ImageOps

    raw_image = context.get("image_bytes")
    bbox = context.get("bbox_xyxy")
    result = context.get("result") or {}
    if raw_image and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            image = Image.open(io.BytesIO(raw_image)).convert("RGB")
            x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
            crop = image.crop((max(0, min(x1, x2)), max(0, min(y1, y2)),
                               min(image.width, max(x1, x2)), min(image.height, max(y1, y2))))
            if crop.width and crop.height:
                scale = max(2, min(6, 480 // max(crop.size)))
                enlarged = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
                enlarged = ImageEnhance.Contrast(ImageOps.grayscale(enlarged)).enhance(1.25)
                st.image(enlarged, caption="선택 부품 각인 확대 · OCR 입력", use_container_width=True)
        except (OSError, ValueError, TypeError):
            st.caption("선택 부품 확대 이미지를 만들 수 없습니다.")

    status = result.get("status")
    if status == "OCR_UNAVAILABLE":
        st.warning(f"각인 OCR을 사용할 수 없습니다 · {result.get('error', '엔진 오류')}")
        st.caption("OCR 의존성을 설치한 뒤 다시 선택하면 재시도합니다.")
    elif status == "OCR_NO_TEXT":
        st.info("확대 이미지에서 읽을 수 있는 각인을 찾지 못했습니다. 원본 소자 종류는 유지됩니다.")
    elif status == "OCR_COMPLETE":
        candidates = result.get("candidates", [])
        if len(candidates) > 1:
            st.caption("다른 방향에서 읽힌 후보 · " + " · ".join(item["text"] for item in candidates[1:4]))
        st.caption("OCR 인식값과 계산값은 검수 전 후보이며, 원본 부품 라벨은 바뀌지 않습니다.")
