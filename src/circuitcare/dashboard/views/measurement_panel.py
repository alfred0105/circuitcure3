"""Inline measurement entry and history for the selected visual component."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.frontend.board_component_registry import resolve_measurement_board_component
from circuitcare.frontend.measurement_adapter import adapt_measurement_input
from circuitcare.frontend.measurement_models import (
    BASE_UNITS, MEASUREMENT_TYPE_LABELS, MEASUREMENT_UNITS, SOURCE_LABELS,
    MeasurementRecord, parse_finite_number,
)
from circuitcare.frontend.measurement_state import (
    delete_measurement, get_measurements, save_measurement, update_measurement,
)
from circuitcare.frontend.models import ProjectContext
from circuitcare.symptom_diagnosis import load_knowledge

SOURCE_OPTIONS = (
    "실제 측정 · 사용자 입력", "시뮬레이션", "가상 데모", "가져온 측정값", "출처 미확인",
)
SOURCE_CODES = {label: code for code, label in SOURCE_LABELS.items()}
TYPE_OPTIONS = tuple(MEASUREMENT_TYPE_LABELS)


def _measurement_verification_gaps(record: MeasurementRecord) -> list[str]:
    """Return UI-facing missing conditions; recording remains allowed without them."""
    gaps = []
    if not record.measurement_target:
        gaps.append("측정 대상 미지정")
    if not record.board_component_id or not str(record.identity_resolution_status).startswith(("USER_SELECTED_", "VERIFIED_")):
        gaps.append("보드 소자 연결 미확인")
    if record.mapping_status != "MAPPED" or not record.probe_id or not record.node_id:
        gaps.append("전기적 측정 지점 연결 미확인")
    if record.uncertainty is None or record.uncertainty <= 0:
        gaps.append("측정 불확실도 미입력")
    if record.source != "user_measurement" or record.verification_status != "USER_REPORTED":
        gaps.append("실측 출처 미확인")
    if not record.test_id:
        gaps.append("검사 프로파일 미선택")
    if (record.reference_min is None or record.reference_max is None
            or record.reference_verification_status not in {"DATASHEET_VERIFIED", "BOARD_SPEC_VERIFIED"}
            or not record.reference_source):
        gaps.append("검증된 정상 범위·출처 없음")
    if (not record.operating_conditions or not record.reference_conditions
            or record.operating_conditions.strip() != record.reference_conditions.strip()):
        gaps.append("측정 조건과 기준 조건이 일치하지 않음")
    if not record.location_verified:
        gaps.append("측정 지점 확인 전")
    return gaps


def _sigma_for_input_unit(value: Any, measurement_type: str, unit: str) -> str | None:
    """UI sigma is entered in the measurement type's base unit; adapters expect the selected unit."""
    if value is None or not str(value).strip():
        return None
    sigma_base = parse_finite_number(value, label="불확실도")
    return str(sigma_base / MEASUREMENT_UNITS[measurement_type][unit])


def render_measurement_panel(
    st: Any, project: ProjectContext, *, selected_component: ComponentOverlayViewModel | None,
    preview: bool = False, component_map_version: str | None = None,
    board_registry: dict[str, Any] | None = None,
) -> None:
    if selected_component is None:
        st.info("PCB Viewer에서 소자를 선택하면 해당 소자의 측정값을 기록할 수 있습니다.")
        return
    if not project.board_id:
        st.warning("보드 ID가 없어 측정 기록을 연결할 수 없습니다.")
        return
    board_component_id = selected_component.board_component_id
    owner = f"{project.project_id}-{project.board_id}-{board_component_id or selected_component.component_id}"
    all_records = get_measurements(st.session_state, project_id=project.project_id,
                                   board_id=project.board_id)
    current_records: list[MeasurementRecord] = []
    stale_records: list[MeasurementRecord] = []
    for row in all_records:
        resolved_id = row.board_component_id
        resolution = row.identity_resolution_status
        if isinstance(board_registry, dict):
            if resolved_id:
                source_photo = row.source_photo_id
                source_local = row.source_component_id or row.component_id
                if source_photo:
                    actual, resolution = resolve_measurement_board_component(
                        board_registry, photo_id=source_photo, component_id=source_local,
                        photo_fingerprint=row.source_photo_fingerprint,
                        component_map_version=(row.component_map_version
                                               if not row.source_photo_fingerprint else None))
                    if actual is None:
                        resolved_id, resolution = None, "UNRESOLVED"
                    else:
                        resolved_id = actual
                elif (not source_photo or not source_local or not any(item.get("board_component_id") == resolved_id
                             for item in board_registry.get("components", []))):
                    resolved_id, resolution = None, "UNRESOLVED"
            else:
                resolved_id, resolution = resolve_measurement_board_component(
                    board_registry, photo_id=row.source_photo_id,
                    component_id=row.source_component_id or row.component_id,
                    photo_fingerprint=row.source_photo_fingerprint,
                    component_map_version=row.component_map_version)
        if (board_component_id and resolved_id == board_component_id
                and str(resolution).startswith("VERIFIED_")):
            projected = replace(
                row, board_component_id=str(resolved_id),
                identity_resolution_status=resolution,
            )
            current_records.append(projected)
        elif row.component_id == selected_component.component_id and board_registry is None:
            if (component_map_version is not None and row.component_map_version == component_map_version
                    and row.component_identity_status not in {"IDENTITY_AMBIGUOUS", "IDENTITY_CONFLICT"}):
                current_records.append(row)
            else:
                stale_records.append(row)
        elif (not str(resolution).startswith("VERIFIED_")
              and (row.component_id == selected_component.component_id
                   or row.source_component_id == selected_component.component_id)):
            stale_records.append(row)
    st.caption(f"측정 대상 · {board_component_id or selected_component.component_id} · {selected_component.component_type}")
    st.caption("측정값을 먼저 기록하세요. 검증 조건이 부족한 기록은 보존되지만 검증 결과로 판정하지 않습니다.")
    if stale_records:
        st.warning(f"이전 Component Map의 측정 기록 {len(stale_records)}개는 현재 부품에 자동 연결하지 않았습니다. 부품 연결을 확인해 주세요.")
    if preview:
        st.info("미리보기에서는 측정 기록을 저장할 수 없습니다.")
        return

    type_key = f"cc_v5_type_{owner}"
    measurement_type = st.selectbox(
        "측정 종류", TYPE_OPTIONS, format_func=MEASUREMENT_TYPE_LABELS.get,
        key=type_key,
    )
    units = tuple(MEASUREMENT_UNITS[measurement_type])
    knowledge = load_knowledge()
    fault_modes = knowledge.get("fault_modes", {}).get("fault_modes", {})
    normalized_type = selected_component.component_type.casefold()
    type_profile = knowledge.get("component_test_profiles", {}).get("profiles", {}).get(normalized_type, {})
    test_profiles = type_profile.get("tests", {})
    test_options = [""] + list(test_profiles)
    unit_key = f"cc_v5_unit_{owner}"
    if st.session_state.get(unit_key) not in units:
        st.session_state[unit_key] = units[0]
    with st.form(f"cc_v5_add_measurement_{owner}"):
        value = st.text_input("측정값", key=f"cc_v5_value_{owner}", placeholder="예: 0, -0.25, 3.3")
        unit = st.selectbox("단위", units, key=unit_key)
        target = st.text_input("측정 대상/단자", placeholder="예: BZ1 양단 전압, U1 VCC-GND")
        with st.expander("검증 정보 · 정상 범위 · 출처", expanded=False):
            uncertainty = st.text_input(
                f"측정 불확실도 σ ({BASE_UNITS[measurement_type]})",
                key=f"cc_v5_sigma_{owner}", placeholder="계측기 정확도 사양에 따른 양수",
            )
            uncertainty_source = st.text_input("불확실도 출처", placeholder="예: 멀티미터 모델·정확도 사양")
            test_id = st.selectbox(
                "부품별 검사 항목", test_options,
                format_func=lambda value: "프로파일 검사 선택 안 함" if not value else value,
                help="등록된 검사 항목만 가설 검증에 사용할 수 있습니다.",
            )
            relations = test_profiles.get(test_id, {}).get("fault_relations", {})
            fault_options = [""] + [fault_id for fault_id in fault_modes if fault_id in relations]
            fault_hypothesis = st.selectbox(
                "검사할 고장 가설", fault_options,
                format_func=lambda code: "선택 안 함" if not code else fault_modes[code]["title"],
            )
            range_col_a, range_col_b = st.columns(2)
            with range_col_a:
                reference_min = st.text_input(f"정상 범위 하한 ({unit})")
            with range_col_b:
                reference_max = st.text_input(f"정상 범위 상한 ({unit})")
            reference_source = st.text_input("정상 범위 출처", placeholder="예: 제조사 데이터시트, 보드 사양")
            reference_source_id = st.text_input("기준 출처 ID", placeholder="예: 데이터시트 문서번호·보드 사양 항목")
            reference_status = st.selectbox("기준 출처", ("UNVERIFIED", "DATASHEET_VERIFIED", "BOARD_SPEC_VERIFIED"),
                                            format_func=lambda value: {
                                                "UNVERIFIED": "미검증",
                                                "DATASHEET_VERIFIED": "데이터시트 확인",
                                                "BOARD_SPEC_VERIFIED": "보드 사양 확인",
                                            }[value])
            operating_conditions = st.text_input("측정 조건", placeholder="예: 전원 5 V, 무부하")
            reference_conditions = st.text_input("기준 조건", placeholder="실제 측정 조건과 동일하게")
            location_verified = st.checkbox("측정 지점이 지정 단자임을 확인함")
            source_label = st.selectbox("측정 출처", SOURCE_OPTIONS, key=f"cc_v5_source_{owner}")
        submitted = st.form_submit_button("측정 기록 저장", type="primary")
    if submitted:
        try:
            record = adapt_measurement_input(
                project_id=project.project_id, board_id=project.board_id,
                component_id=selected_component.component_id,
                component_identity_status=selected_component.identification_status,
                measurement_type=measurement_type, value=value, unit=unit,
                uncertainty=_sigma_for_input_unit(uncertainty, measurement_type, unit),
                source=SOURCE_CODES[source_label],
                component_map_version=component_map_version,
                component_type=selected_component.component_type,
                board_component_id=board_component_id,
                source_photo_id=selected_component.canonical_frame_id,
                source_photo_fingerprint=next((item.get("image_fingerprint")
                                                for item in (board_registry or {}).get("photos", [])
                                                if item.get("photo_id") == selected_component.canonical_frame_id), None),
                source_component_id=selected_component.component_id,
                measurement_location=target,
                identity_resolution_status=("USER_SELECTED_VERIFIED" if board_component_id else "UNRESOLVED"),
                measurement_target=target, fault_hypothesis=fault_hypothesis or None,
                test_id=test_id or None,
                reference_min=reference_min, reference_max=reference_max,
                reference_source=reference_source,
                reference_source_id=reference_source_id,
                reference_verification_status=reference_status,
                operating_conditions=operating_conditions,
                reference_conditions=reference_conditions,
                location_verified=location_verified,
                uncertainty_source=uncertainty_source,
            )
            save_measurement(st.session_state, record)
            st.success("측정 기록을 저장했습니다. 저장은 측정 검증 완료를 뜻하지 않습니다.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.caption(f"현재 Component Map 측정 기록 · {len(current_records)}개")
    if not current_records:
        st.caption("저장된 측정값 없음")
    for record in reversed(current_records):
        _render_record_editor(st, project=project, record=record, owner=owner)
    if stale_records:
        with st.expander(f"이전 Map · 연결 확인 필요 · {len(stale_records)}개"):
            for record in reversed(stale_records):
                st.caption(
                    f"{MEASUREMENT_TYPE_LABELS[record.measurement_type]} · "
                    f"{record.entered_value:g} {record.entered_unit} · "
                    f"{record.timestamp[:16]} · Map {record.component_map_version or '없음'} · "
                    "현재 부품과 연결 미확인"
                )


def _render_record_editor(st: Any, *, project: ProjectContext, record: MeasurementRecord, owner: str) -> None:
    title = (f"{MEASUREMENT_TYPE_LABELS[record.measurement_type]} · "
             f"{record.entered_value:g} {record.entered_unit} · {record.timestamp[:16]}")
    with st.expander(title):
        check_reasons = _measurement_verification_gaps(record)
        if check_reasons:
            st.warning("기록됨 · 검증 대기 · " + " / ".join(check_reasons))
        else:
            st.success("필수 정보 입력 완료 · 진단 결과와 대조 대기")
        st.caption(f"출처 · {SOURCE_LABELS[record.source]} · 기록 상태 {record.verification_status}")
        st.caption(f"Component ID · {record.component_id} · 전기적 Mapping 미확인")
        if record.measurement_target or record.fault_hypothesis:
            st.caption(f"검사 대상 · {record.measurement_target or '미지정'} · 가설 {record.fault_hypothesis or '미지정'}")
        if record.reference_min is not None and record.reference_max is not None:
            st.caption(f"등록 기준 · {record.reference_min:g}–{record.reference_max:g} {record.unit} · "
                       f"{record.reference_source or '출처 없음'} · 조건 {record.operating_conditions or '미지정'}")
        if record.uncertainty is not None:
            st.caption(f"불확실도 σ · {record.uncertainty:g} {record.unit}")
        st.caption(f"Measurement Set · {(record.measurement_set_id or '')[:12]}")
        type_key = f"cc_v5_edit_type_{record.measurement_id}"
        options = tuple(MEASUREMENT_UNITS)
        if st.session_state.get(type_key) not in options:
            st.session_state[type_key] = record.measurement_type
        callback_key = f"cc_v5_edit_type_previous_{record.measurement_id}"
        if callback_key not in st.session_state:
            st.session_state[callback_key] = record.measurement_type

        def reset_changed_kind(*, state: Any, measurement_id: str, key: str, previous_key: str) -> None:
            new_type = state.get(key)
            if new_type != state.get(previous_key):
                state[f"cc_v5_edit_unit_{measurement_id}"] = tuple(MEASUREMENT_UNITS[new_type])[0]
                state[f"cc_v5_edit_sigma_{measurement_id}"] = ""
                state[previous_key] = new_type

        edit_type = st.selectbox("수정 측정 종류", options, format_func=MEASUREMENT_TYPE_LABELS.get,
                                 key=type_key,
                                 on_change=reset_changed_kind,
                                 kwargs={"state": st.session_state, "measurement_id": record.measurement_id,
                                         "key": type_key, "previous_key": callback_key})
        units = tuple(MEASUREMENT_UNITS[edit_type])
        unit_key = f"cc_v5_edit_unit_{record.measurement_id}"
        if st.session_state.get(unit_key) not in units:
            st.session_state[unit_key] = record.entered_unit if record.entered_unit in units else units[0]
        knowledge = load_knowledge()
        fault_modes = knowledge.get("fault_modes", {}).get("fault_modes", {})
        type_profile = knowledge.get("component_test_profiles", {}).get("profiles", {}).get(
            record.component_type.casefold(), {}
        )
        test_profiles = type_profile.get("tests", {})
        test_options = [""] + list(test_profiles)
        selected_test = record.test_id if record.test_id in test_profiles else ""
        with st.form(f"cc_v5_edit_form_{record.measurement_id}"):
            value = st.text_input("수정 측정값", value=f"{record.entered_value:g}",
                                  key=f"cc_v5_edit_value_{record.measurement_id}")
            unit = st.selectbox("수정 단위", units, key=unit_key)
            target = st.text_input("측정 대상/단자", value=record.measurement_target or "",
                                   key=f"cc_v5_edit_target_{record.measurement_id}")
            source_label = st.selectbox(
                "측정 출처", SOURCE_OPTIONS, index=SOURCE_OPTIONS.index(SOURCE_LABELS[record.source]),
                key=f"cc_v5_edit_source_{record.measurement_id}",
            )
            sigma_default = ""
            if edit_type == record.measurement_type and record.uncertainty is not None:
                sigma_default = f"{record.uncertainty:g}"
            with st.expander("고급 검증 조건 수정", expanded=False):
                uncertainty = st.text_input(
                    f"불확실도 σ ({BASE_UNITS[edit_type]})", value=sigma_default,
                    key=f"cc_v5_edit_sigma_{record.measurement_id}",
                )
                uncertainty_source = st.text_input(
                    "불확실도 출처", value=record.uncertainty_source or "",
                    key=f"cc_v5_edit_sigma_source_{record.measurement_id}",
                )
                test_id = st.selectbox(
                    "부품별 검사 항목", test_options,
                    index=test_options.index(selected_test),
                    format_func=lambda item: "프로파일 검사 선택 안 함" if not item else item,
                    key=f"cc_v5_edit_test_{record.measurement_id}",
                )
                relations = test_profiles.get(test_id, {}).get("fault_relations", {})
                fault_options = [""] + [item for item in fault_modes if item in relations]
                fault_default = record.fault_hypothesis if record.fault_hypothesis in fault_options else ""
                fault_hypothesis = st.selectbox(
                    "검사할 고장 가설", fault_options,
                    index=fault_options.index(fault_default),
                    format_func=lambda code: "선택 안 함" if not code else fault_modes[code]["title"],
                    key=f"cc_v5_edit_fault_{record.measurement_id}",
                )
                multiplier = MEASUREMENT_UNITS[edit_type][unit]
                reference_min_default = ""
                reference_max_default = ""
                if edit_type == record.measurement_type:
                    if record.reference_min is not None:
                        reference_min_default = f"{record.reference_min / multiplier:g}"
                    if record.reference_max is not None:
                        reference_max_default = f"{record.reference_max / multiplier:g}"
                range_left, range_right = st.columns(2)
                with range_left:
                    reference_min = st.text_input(
                        f"정상 범위 하한 ({unit})", value=reference_min_default,
                        key=f"cc_v5_edit_ref_min_{record.measurement_id}",
                    )
                with range_right:
                    reference_max = st.text_input(
                        f"정상 범위 상한 ({unit})", value=reference_max_default,
                        key=f"cc_v5_edit_ref_max_{record.measurement_id}",
                    )
                reference_source = st.text_input(
                    "정상 범위 출처", value=record.reference_source or "",
                    key=f"cc_v5_edit_ref_source_{record.measurement_id}",
                )
                reference_source_id = st.text_input(
                    "기준 출처 ID", value=record.reference_source_id or "",
                    key=f"cc_v5_edit_ref_source_id_{record.measurement_id}",
                )
                reference_status_options = ("UNVERIFIED", "DATASHEET_VERIFIED", "BOARD_SPEC_VERIFIED")
                current_reference_status = record.reference_verification_status
                if current_reference_status not in reference_status_options:
                    current_reference_status = "UNVERIFIED"
                reference_status = st.selectbox(
                    "기준 출처 검증", reference_status_options,
                    index=reference_status_options.index(current_reference_status),
                    format_func=lambda item: {"UNVERIFIED": "미검증", "DATASHEET_VERIFIED": "데이터시트 확인", "BOARD_SPEC_VERIFIED": "보드 사양 확인"}[item],
                    key=f"cc_v5_edit_ref_status_{record.measurement_id}",
                )
                operating_conditions = st.text_input(
                    "측정 조건", value=record.operating_conditions or "",
                    key=f"cc_v5_edit_operating_{record.measurement_id}",
                )
                reference_conditions = st.text_input(
                    "기준 조건", value=record.reference_conditions or "",
                    key=f"cc_v5_edit_reference_conditions_{record.measurement_id}",
                )
                location_verified = st.checkbox(
                    "측정 지점이 지정 단자임을 확인함", value=record.location_verified,
                    key=f"cc_v5_edit_location_{record.measurement_id}",
                )
            edit = st.form_submit_button("변경 저장")
            delete = st.form_submit_button("측정값 삭제")
        if edit:
            try:
                updated = adapt_measurement_input(
                    project_id=record.project_id, board_id=record.board_id,
                    component_id=record.component_id,
                    component_identity_status=record.component_identity_status,
                    measurement_type=edit_type, value=value, unit=unit,
                    uncertainty=_sigma_for_input_unit(uncertainty, edit_type, unit),
                    source=SOURCE_CODES[source_label],
                    component_map_version=record.component_map_version,
                    component_type=record.component_type,
                    measurement_id=record.measurement_id, timestamp=record.timestamp,
                    measurement_target=target,
                    fault_hypothesis=fault_hypothesis or None,
                    test_id=test_id or None,
                    reference_min=reference_min,
                    reference_max=reference_max,
                    reference_source=reference_source,
                    reference_source_id=reference_source_id,
                    reference_verification_status=reference_status,
                    operating_conditions=operating_conditions,
                    reference_conditions=reference_conditions,
                    location_verified=location_verified,
                    uncertainty_source=uncertainty_source,
                )
                updated = replace(
                    updated, board_component_id=record.board_component_id,
                    source_photo_id=record.source_photo_id,
                    source_photo_fingerprint=record.source_photo_fingerprint,
                    source_component_id=record.source_component_id,
                    measurement_location=target or None,
                    identity_resolution_status=record.identity_resolution_status,
                    component_type=record.component_type,
                )
                updated = replace(updated, changed_at=datetime.now(timezone.utc).isoformat())
                update_measurement(st.session_state, updated)
                st.success("측정 기록을 수정했습니다. 기존 진단 결과는 무효화되었습니다.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if delete:
            delete_measurement(st.session_state, project_id=project.project_id,
                               board_id=project.board_id or "", measurement_id=record.measurement_id)
            st.success("측정 기록을 삭제했습니다. 기존 진단 결과는 무효화되었습니다.")
            st.rerun()
