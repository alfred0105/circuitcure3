"""F1 diagnosis intake screen for local PCB photos and symptom context."""

from __future__ import annotations

import hashlib
import inspect
from typing import Any

from circuitcare.frontend.intake import (
    CAPTURE_TYPES,
    MAX_PHOTOS,
    MAX_SYMPTOM_CHARS,
    PHOTO_SIDES,
    SYMPTOM_CATEGORIES,
    add_image_batch,
    create_project_context,
    normalize_symptom_description,
    validate_circuit_upload,
)
from circuitcare.frontend.photo_adapter import normalize_photo_metadata
from circuitcare.frontend.component_map_adapter import clear_component_map_state
from circuitcare.frontend.state import (
    INTAKE_IMAGES_KEY,
    INTAKE_PROJECT_KEY,
    INTAKE_UPLOAD_BATCH_KEY,
    clear_intake_draft,
    draft_is_current,
)
from circuitcare.frontend.workbench_state import SCREEN_KEY, enter_workbench
from circuitcare.vision.image_loader import load_board_image


def _style(st: Any) -> None:
    st.markdown(
        """<style>
        [data-testid="stMainBlockContainer"] { max-width: 940px; padding-top: 2rem; }
        .cc-entry-brand { text-align:center; font-size:1.05rem; font-weight:700; color:#e8eef5; }
        .cc-entry-title { text-align:center; font-size:1.65rem; font-weight:700; color:#f4f7fa; margin:.5rem 0 1.5rem; }
        .cc-entry-section { margin-top:1.2rem; color:#e4ebf2; font-weight:650; }
        .cc-entry-muted { color:#9aa9b8; font-size:.82rem; }
        [data-testid="stFileUploader"] { border:1px dashed #546474; border-radius:8px; padding:.55rem; }
        @media (max-width:700px) {
          [data-testid="stMainBlockContainer"] { padding:1rem .7rem; }
          .cc-entry-title { font-size:1.35rem; }
        }
        </style>""",
        unsafe_allow_html=True,
    )


def _upload_limit_kwargs(st: Any) -> dict[str, int]:
    return {"max_upload_size": 20} if "max_upload_size" in inspect.signature(st.file_uploader).parameters else {}


def _current_fingerprint(state: Any, category: str, description: str | None) -> str | None:
    images = state.get(INTAKE_IMAGES_KEY, [])
    if not images:
        return None
    try:
        return create_project_context(
            images=images, symptom_category=category, symptom_description=description,
            circuit_source=state.get("circuitcare_v2_circuit_source"),
        ).input_fingerprint
    except ValueError:
        return None


def _normalize_photo_state(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Migrate old session roles while preserving bytes, IDs, and source hashes."""
    normalized_images = []
    for existing in images:
        normalized, _ = normalize_photo_metadata(existing)
        item = dict(existing)
        item.update({
            "image_id": normalized.image_id,
            "side": normalized.side,
            "capture_type": normalized.capture_type,
            "sha256": normalized.content_sha256,
            "image_dimensions": normalized.image_dimensions,
            "image_format": normalized.image_format,
        })
        item.pop("role", None)
        normalized_images.append(item)
    return normalized_images


def render_diagnosis_entry(st: Any) -> None:
    """Render photo and symptom intake for the PCB diagnosis workbench."""
    _style(st)
    state = st.session_state
    images = _normalize_photo_state(state.setdefault(INTAKE_IMAGES_KEY, []))
    state[INTAKE_IMAGES_KEY] = images
    state.setdefault(INTAKE_UPLOAD_BATCH_KEY, 0)
    category = state.get("circuitcare_v2_symptom_category", "unknown")
    description_raw = state.get("circuitcare_v2_symptom_description", "")
    try:
        description = normalize_symptom_description(description_raw)
    except ValueError:
        description = None
    fingerprint = _current_fingerprint(state, category, description)
    project = state.get(INTAKE_PROJECT_KEY)
    if project and draft_is_current(state, fingerprint or ""):
        enter_workbench(state, project)
        st.rerun()
    if project:
        clear_intake_draft(state)

    _, center, _ = st.columns([1, 2.2, 1])
    with center:
        st.markdown('<div class="cc-entry-brand">CircuitCare AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="cc-entry-title">새 PCB 진단</div>', unsafe_allow_html=True)

        st.markdown('<div class="cc-entry-section">PCB 사진 추가</div>', unsafe_allow_html=True)
        st.caption(f"앞면 · 뒷면 · 확대 사진 · 최대 {MAX_PHOTOS}장, 사진당 20 MB")
        batch_key = f"cc_v2_photo_upload_{state[INTAKE_UPLOAD_BATCH_KEY]}"
        uploaded = st.file_uploader(
            "PNG · JPG · JPEG · WebP", type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True, key=batch_key,
            help="사진은 이 Streamlit 세션에서만 보관합니다.",
            **_upload_limit_kwargs(st),
        )
        if uploaded:
            new_images, messages = add_image_batch(
                images, ((item.name, item.getvalue()) for item in uploaded),
            )
            state[INTAKE_IMAGES_KEY] = new_images
            if new_images != images:
                clear_component_map_state(state)
            state["circuitcare_v2_photo_messages"] = messages
            state[INTAKE_UPLOAD_BATCH_KEY] += 1
            state.pop(INTAKE_PROJECT_KEY, None)
            st.rerun()

        for message in state.pop("circuitcare_v2_photo_messages", []):
            st.warning(message)

        images = state.get(INTAKE_IMAGES_KEY, [])
        if images:
            for start in range(0, len(images), 4):
                row = images[start:start + 4]
                cols = st.columns(len(row))
                for offset, image_record in enumerate(row):
                    index = start + offset
                    with cols[offset]:
                        preview = load_board_image(image_record["data"], max_edge=640).image
                        st.image(preview, use_container_width=True)
                        st.caption(image_record["filename"][:36])
                        side = st.selectbox(
                            "촬영면", list(PHOTO_SIDES),
                            index=list(PHOTO_SIDES).index(image_record["side"]),
                            format_func=lambda value: PHOTO_SIDES[value],
                            key=f"cc_v2_photo_side_{image_record['image_id']}",
                            label_visibility="collapsed",
                        )
                        capture_type = st.selectbox(
                            "촬영 유형", list(CAPTURE_TYPES),
                            index=list(CAPTURE_TYPES).index(image_record["capture_type"]),
                            format_func=lambda value: CAPTURE_TYPES[value],
                            key=f"cc_v2_photo_capture_{image_record['image_id']}",
                            label_visibility="collapsed",
                        )
                        if side != image_record["side"] or capture_type != image_record["capture_type"]:
                            clear_component_map_state(state)
                            image_record["side"] = side
                            image_record["capture_type"] = capture_type
                            state.pop(INTAKE_PROJECT_KEY, None)
                            state.pop("circuitcare_v2_analysis_status", None)
                        if st.button("사진 제거", key=f"cc_v2_remove_{image_record['sha256']}", use_container_width=True):
                            clear_component_map_state(state)
                            state[INTAKE_IMAGES_KEY] = [item for item in images if item["sha256"] != image_record["sha256"]]
                            state.pop(INTAKE_PROJECT_KEY, None)
                            state.pop("circuitcare_v2_analysis_status", None)
                            st.rerun()
        else:
            st.info("PCB 사진을 한 장 이상 추가하세요.")

        st.markdown('<div class="cc-entry-section">고장 증상</div>', unsafe_allow_html=True)
        category_code = st.radio(
            "고장 증상", list(SYMPTOM_CATEGORIES),
            index=list(SYMPTOM_CATEGORIES).index(category),
            format_func=lambda value: SYMPTOM_CATEGORIES[value],
            key="circuitcare_v2_symptom_category", horizontal=True,
        )
        description_text = st.text_area(
            "상세 증상", placeholder="증상을 입력하세요", max_chars=MAX_SYMPTOM_CHARS,
            key="circuitcare_v2_symptom_description", height=86,
        )
        try:
            description = normalize_symptom_description(description_text)
            description_error = None
        except ValueError as exc:
            description = None
            description_error = str(exc)

        st.markdown('<div class="cc-entry-section">추가 자료 (선택)</div>', unsafe_allow_html=True)
        st.caption("지원 형식 · KiCad XML Netlist, KiCad 회로도")
        circuit_upload = st.file_uploader(
            ".xml 또는 .kicad_sch", type=["xml", "kicad_sch"], key="cc_v2_circuit_upload",
            **_upload_limit_kwargs(st),
        )
        if circuit_upload:
            raw_circuit = circuit_upload.getvalue()
            digest = hashlib.sha256(raw_circuit).hexdigest()
            if state.get("circuitcare_v2_ignored_circuit_digest") == digest:
                pass
            elif state.get("circuitcare_v2_circuit_source", {}).get("content_sha256") != digest:
                try:
                    source = validate_circuit_upload(circuit_upload.name, raw_circuit)
                    state["circuitcare_v2_circuit_source"] = source
                    state["circuitcare_v2_circuit_bytes"] = raw_circuit
                    state.pop("circuitcare_v2_ignored_circuit_digest", None)
                    state.pop(INTAKE_PROJECT_KEY, None)
                    state.pop("circuitcare_v2_circuit_error", None)
                except ValueError as exc:
                    state["circuitcare_v2_circuit_error"] = str(exc)
                    state.pop("circuitcare_v2_circuit_source", None)
                    state.pop("circuitcare_v2_circuit_bytes", None)
            if state.get("circuitcare_v2_circuit_source"):
                st.caption(f"자료 확인 · {state['circuitcare_v2_circuit_source']['filename']} · 진단 미실행")
            elif state.get("circuitcare_v2_circuit_error"):
                st.error(state["circuitcare_v2_circuit_error"])
        if state.get("circuitcare_v2_circuit_source"):
            source = state["circuitcare_v2_circuit_source"]
            if not circuit_upload:
                st.caption(f"자료 확인 · {source['filename']} · 진단 미실행")
            if st.button("설계 자료 제거", key="cc_v2_remove_circuit"):
                if circuit_upload:
                    state["circuitcare_v2_ignored_circuit_digest"] = source["content_sha256"]
                state.pop("circuitcare_v2_circuit_source", None)
                state.pop("circuitcare_v2_circuit_bytes", None)
                state.pop(INTAKE_PROJECT_KEY, None)
                st.rerun()

        if description_error:
            st.error(description_error)
        can_start = bool(images) and description_error is None
        if st.button("진단 시작", type="primary", use_container_width=True, disabled=not can_start):
            try:
                project = create_project_context(
                    images=state[INTAKE_IMAGES_KEY], symptom_category=category_code,
                    symptom_description=description,
                    circuit_source=state.get("circuitcare_v2_circuit_source"),
                    project_id=state.get("circuitcare_v2_edit_project_id"),
                    board_id=state.get("circuitcare_v2_edit_board_id"),
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                state[INTAKE_PROJECT_KEY] = project
                state["circuitcare_v2_analysis_status"] = "NOT_RUN"
                state.pop("circuitcare_v2_edit_project_id", None)
                state.pop("circuitcare_v2_edit_board_id", None)
                enter_workbench(state, project)
                st.rerun()

        st.caption("사진과 증상은 진단 보조 입력입니다. 이 단계에서는 AI 진단을 실행하지 않습니다.")
