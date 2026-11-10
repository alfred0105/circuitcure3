"""PCB inspection summary and selected-component details."""

from __future__ import annotations

import io
from dataclasses import replace
from typing import Any

from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel, display_identity_status
from circuitcare.frontend.measurement_state import get_measurements
from circuitcare.frontend.models import ProjectContext
from circuitcare.dashboard.views.diagnosis_panel import render_diagnosis_panel
from circuitcare.dashboard.views.measurement_panel import render_measurement_panel
from circuitcare.dashboard.views.symptom_diagnosis_panel import render_symptom_diagnosis_panel
from circuitcare.vision.component_recognition_panel import render_selected_component_editor

MAP_VERSION_KEY = "circuitcare_v4_selection_map_key"
COMPONENT_EDIT_KEY = "circuitcare_v4_component_edits"
COMPONENT_INVENTORY_KEY = "circuitcare_v4_component_inventory_by_map"
ROLE_OPTIONS = ("power", "control", "control_input", "control_timing", "driver", "output",
                "input", "input_output", "protection", "thermal", "buzzer", "relay")
COMPONENT_TYPE_OPTIONS = (
    "unknown", "resistor", "inductor", "capacitor", "connector", "jumper", "IC",
    "switch", "relay", "transistor", "potentiometer", "battery", "buzzer", "diode",
    "display", "emi_filter", "ferrite_bead", "fuse", "LED", "clock", "oscillator",
    "voltage_regulator", "MOV", "transformer", "heatsink",
)


def _component_label(value: str) -> str:
    types = {
        "capacitor": "커패시터", "resistor": "저항", "inductor": "인덕터",
        "diode": "다이오드", "transistor": "트랜지스터", "integrated_circuit": "IC",
        "ic": "IC", "connector": "커넥터", "unknown_component": "부품 후보",
    }
    return types.get(value.lower(), value)


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
        render_symptom_diagnosis_panel(st, project, components=components, preview=preview,
                                       component_inventory=inventory)
        st.divider()
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
        st.markdown("### 선택 부품")
        if selected_component is None:
            st.caption("PCB에서 박스를 누르면 이곳에 인식 정보가 표시됩니다.")
        else:
            edits = st.session_state.setdefault(COMPONENT_EDIT_KEY, {})
            edit = edits.get(selected_component.component_id, {})
            current_type = edit.get("component_type", selected_component.component_type)
            current_marking = edit.get("ocr_marking", selected_component.ocr_marking or "")
            current_value = edit.get("component_info", selected_component.component_info or "")
            st.caption(f"부품 ID · {selected_component.component_id}")
            inventory_item = inventory.setdefault(selected_component.component_id, {
                "presence_status": "UNCONFIRMED", "component_type_status": "UNCONFIRMED",
                "part_identity_status": "UNCONFIRMED", "confirmed_functional_roles": [],
            })
            review_row = None
            if review_image_bytes is not None and review_scope:
                from circuitcare.vision.component_recognition_panel import review_keys

                _result_key, records_key, _error_key = review_keys(review_scope)
                review_row = next((row for row in st.session_state.get(records_key, [])
                                   if row.get("component_id") == selected_component.component_id), None)
                if review_row and review_row.get("상태") == "제외":
                    inventory_item["presence_status"] = "EXCLUDED"
            presence_options = ("UNCONFIRMED", "CONFIRMED", "EXCLUDED")
            presence_labels = {"UNCONFIRMED": "미확인", "CONFIRMED": "실제 부품 확인", "EXCLUDED": "오검출 · 제외"}
            presence_key = f"cc_component_presence_{hash(owner)}_{selected_component.component_id}"
            current_presence = inventory_item.get("presence_status", "UNCONFIRMED")
            if review_row and review_row.get("상태") == "제외":
                st.session_state[presence_key] = "EXCLUDED"
            elif st.session_state.get(presence_key) not in presence_options:
                st.session_state[presence_key] = current_presence
            presence_status = st.radio(
                "PCB에서 선택 부품 상태", presence_options,
                format_func=presence_labels.get, horizontal=True, key=presence_key,
                on_change=_sync_inventory_widget, args=(st, owner, selected_component.component_id,
                                                        "presence_status", presence_key),
            )
            inventory_item["presence_status"] = presence_status
            st.caption("확인한 소자만 진단 후보로 사용합니다. 오검출은 제외하고, 아직 판단이 안 되면 미확인으로 두세요.")
            type_status_key = f"cc_component_type_status_{hash(owner)}_{selected_component.component_id}"
            type_status_options = ("UNCONFIRMED", "USER_CONFIRMED")
            if st.session_state.get(type_status_key) not in type_status_options:
                st.session_state[type_status_key] = inventory_item.get("component_type_status", "UNCONFIRMED")
            component_type_status = st.radio("부품 종류 확인", type_status_options,
                format_func=lambda value: "종류 확인됨" if value == "USER_CONFIRMED" else "종류 미확인",
                horizontal=True, key=type_status_key,
                on_change=_sync_inventory_widget, args=(st, owner, selected_component.component_id,
                                                        "component_type_status", type_status_key))
            inventory_item["component_type_status"] = component_type_status
            with st.expander("상세 식별·기능 확인", expanded=False):
                identity_key = f"cc_part_identity_status_{hash(owner)}_{selected_component.component_id}"
                if st.session_state.get(identity_key) not in type_status_options:
                    st.session_state[identity_key] = inventory_item.get("part_identity_status", "UNCONFIRMED")
                part_identity_status = st.radio("부품 모델/부품번호 확인", type_status_options,
                    format_func=lambda value: "모델 식별됨" if value == "USER_CONFIRMED" else "모델 미확인",
                    horizontal=True, key=identity_key,
                    on_change=_sync_inventory_widget, args=(st, owner, selected_component.component_id,
                                                            "part_identity_status", identity_key))
                inventory_item["part_identity_status"] = part_identity_status
                role_key = f"cc_component_roles_{hash(owner)}_{selected_component.component_id}"
                current_roles = inventory_item.get("confirmed_functional_roles", [])
                if role_key not in st.session_state:
                    st.session_state[role_key] = [role for role in current_roles if role in ROLE_OPTIONS]
                confirmed_roles = st.multiselect(
                    "이 부품에서 확인된 기능 역할", ROLE_OPTIONS, key=role_key,
                    help="직접 확인한 역할만 선택합니다. 선택하지 않은 배치/OCR 추정은 미확정 후보로 유지됩니다.",
                    on_change=_sync_inventory_widget, args=(st, owner, selected_component.component_id,
                                                            "confirmed_functional_roles", role_key),
                )
                inventory_item["confirmed_functional_roles"] = list(confirmed_roles)
            inventories[owner] = inventory
            st.session_state[COMPONENT_INVENTORY_KEY] = inventories
            selected_component = replace(
                selected_component,
                component_type=current_type,
                presence_status=presence_status,
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
                                        "component_type": current_type,
                                        "ocr_marking": choice,
                                        "component_info": marking_result.get("inferred_value") or current_value,
                                    }
                                    st.session_state[COMPONENT_EDIT_KEY] = edits
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
            type_key = f"cc_v4_edit_type_{selected_component.component_id}"
            marking_key = f"cc_v4_edit_marking_{selected_component.component_id}"
            value_key = f"cc_v4_edit_value_{selected_component.component_id}"
            if st.session_state.get(type_key) not in COMPONENT_TYPE_OPTIONS:
                st.session_state[type_key] = current_type if current_type in COMPONENT_TYPE_OPTIONS else "unknown"
            if marking_key not in st.session_state:
                st.session_state[marking_key] = current_marking
            if value_key not in st.session_state:
                st.session_state[value_key] = current_value
            with st.expander("인식 정보 편집", expanded=False):
                st.selectbox("인식 부품 종류", COMPONENT_TYPE_OPTIONS,
                             format_func=_component_label, key=type_key)
                st.text_input("OCR 인식값 · 각인/부품번호", key=marking_key,
                              placeholder="예: 103, LM358")
                st.text_input("소자 값 · OCR 유추값 또는 직접 입력", key=value_key,
                              placeholder="예: 10 kΩ, 100 nF, 3.3 V")
                if st.button("인식 정보 저장", type="primary", key=f"cc_v4_save_component_{selected_component.component_id}"):
                    edits[selected_component.component_id] = {
                        "component_type": st.session_state[type_key],
                        "ocr_marking": st.session_state[marking_key].strip(),
                        "component_info": st.session_state[value_key].strip(),
                    }
                    st.session_state[COMPONENT_EDIT_KEY] = edits
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
                render_selected_component_editor(
                    st, image_bytes=review_image_bytes, scope=review_scope,
                    component_id=selected_component.component_id,
                )
        measurements = get_measurements(
            st.session_state, project_id=project.project_id, board_id=project.board_id or "",
        )
        map_version = st.session_state.get(MAP_VERSION_KEY)
        with st.expander("측정 및 회로 모델 상세", expanded=False):
            render_measurement_panel(
                st, project, selected_component=selected_component, preview=preview,
                component_map_version=map_version,
            )
            selection = st.session_state.get("circuitcare_v2_selection")
            render_diagnosis_panel(
                st, project, measurements=measurements,
                analysis_context_id=getattr(selection, "analysis_context_id", None),
                component_map_version=map_version,
            )


def _diagnosis_status_label(st: Any, project: ProjectContext) -> str:
    state = st.session_state.get(f"cc_symptom_diagnosis_{project.project_id}", {})
    result = state.get("result") if isinstance(state, dict) else None
    if not isinstance(result, dict):
        return "진단 미실행"
    status = result.get("status")
    if status == "CANDIDATES_AVAILABLE":
        return f"진단 완료 · 후보 {len(result.get('component_candidates', []))}개"
    if status == "ABSTAIN_INSUFFICIENT_EVIDENCE":
        return "추론 보류 · 추가 정보 필요"
    return "진단 결과 확인 필요"


def _sync_inventory_widget(st: Any, owner: str, component_id: str, field: str, widget_key: str) -> None:
    inventories = st.session_state.setdefault(COMPONENT_INVENTORY_KEY, {})
    inventory = inventories.setdefault(owner, {})
    item = inventory.setdefault(component_id, {})
    item[field] = st.session_state.get(widget_key)
    st.session_state[COMPONENT_INVENTORY_KEY] = inventories


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
