"""Connection-optional, evidence-ranked fault diagnosis UI."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
import time
from pathlib import Path
from uuid import uuid4

from circuitcare.frontend.models import ProjectContext
from circuitcare.frontend.measurement_state import get_measurements
from circuitcare.local_llm import LocalLLMError, OllamaClient
from circuitcare.symptom_diagnosis import diagnose, load_knowledge
from circuitcare.symptom_diagnosis.llm_assist import (
    explain_candidates, explain_component_candidates, suggest_unregistered_causes,
)
from circuitcare.symptom_diagnosis.observation_store import ObservationStore
from circuitcare.symptom_diagnosis.diagnosis_state import build_overlay_state
from circuitcare.symptom_diagnosis.schemas import ComponentRecord, Observation
from circuitcare.symptom_diagnosis.symptom_parser import parse_user_report
from circuitcare.frontend.workbench_state import ACTIVE_PHOTO_KEY, update_selection
from circuitcare.analysis.models import AnalysisScope
from circuitcare.analysis.fingerprints import (
    build_analysis_fingerprints, build_measurement_fingerprint,
    build_symptom_input_fingerprint,
)
from circuitcare.analysis.state_manager import (
    begin_analysis, commit_analysis_result, fail_analysis, get_current_result,
    get_current_state, migrate_legacy_result,
)


def _project_key(project: ProjectContext) -> str:
    return f"cc_symptom_diagnosis_{project.project_id}"


def _diagnosis_fingerprint(
    project: ProjectContext, components: tuple[Any, ...], symptoms: list[str],
    symptom_text: str, observations: dict[str, Any], answers: dict[str, Any],
    measurements: list[Any], silkscreen_labels: list[dict[str, Any]], state: Any,
    confirmed_component_ids: list[str] | None = None,
    inventory_complete: bool = False,
    component_inventory: dict[str, dict[str, Any]] | None = None,
    analysis_target_id: str | None = None,
) -> str:
    target_id = str(analysis_target_id or state.get(ACTIVE_PHOTO_KEY) or "target:default")
    analysis_scope = AnalysisScope(project.project_id, project.board_id or "", target_id)
    photo_snapshots = state.get("circuitcare_v6_photo_component_snapshots", {})
    photo_snapshot = photo_snapshots.get(analysis_scope.key) if isinstance(photo_snapshots, dict) else None
    return build_symptom_input_fingerprint(
        project=project, components=components, symptoms=symptoms, symptom_text=symptom_text,
        observations=observations, answers=answers, silkscreen_labels=silkscreen_labels,
        component_inventory=component_inventory, confirmed_component_ids=confirmed_component_ids,
        inventory_complete=inventory_complete,
        target_id=target_id,
        target_photo_sha256=(photo_snapshot.get("image_fingerprint")
                             if isinstance(photo_snapshot, dict) else None),
    )


def _analysis_scope(project: ProjectContext, state: Any, analysis_target_id: str | None = None) -> AnalysisScope:
    return AnalysisScope(project.project_id, project.board_id or "",
                         str(analysis_target_id or state.get("circuitcare_v7_analysis_target_id")
                             or state.get(ACTIVE_PHOTO_KEY) or "target:default"))


def current_analysis_snapshot(
    st: Any, project: ProjectContext, *, components: tuple[Any, ...] = (),
    component_inventory: dict[str, dict[str, Any]] | None = None,
    symptom_text: str | None = None, selected_codes: list[str] | None = None,
    observations: dict[str, Any] | None = None, inventory_complete: bool | None = None,
    analysis_target_id: str | None = None,
    component_id_aliases: dict[str, str | None] | None = None,
    board_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild and validate a photo-scoped or board-scoped analysis version."""
    key = _project_key(project)
    state = st.session_state.get(key, {})
    if not isinstance(state, dict):
        state = {}
    kb = load_knowledge()
    symptom_items = kb["symptom_ontology"]["symptoms"]
    code_to_label = {code: item["label"] for code, item in symptom_items.items()}
    if selected_codes is None:
        labels = st.session_state.get(f"{key}_symptoms", [
            code_to_label[code] for code in state.get("symptoms", []) if code in code_to_label
        ])
        selected_codes = [code for code, label in code_to_label.items() if label in labels]
    if symptom_text is None:
        symptom_text = st.session_state.get(f"{key}_description", project.symptom_description or "")
    if inventory_complete is None:
        inventory_complete = bool(st.session_state.get(
            f"{key}_inventory_complete", state.get("inventory_complete", False)))
    if observations is None:
        observations = dict(state.get("manual_observations", state.get("observations", {})))
        for fact in ("POWER_SOURCE_VERIFIED", "BUZZER_VERIFIED_FAILED", "BUZZER_VERIFIED_WORKING",
                     "BUZZER_DRIVE_ABNORMAL", "BUZZER_DRIVE_VERIFIED_NORMAL", "VISIBLE_SOLDER_DAMAGE"):
            widget_value = st.session_state.get(f"{key}_{fact}")
            widget_value = {"예": True, "아니오": False, "모름": None}.get(widget_value, widget_value)
            if widget_value in {True, False, None}:
                observations[fact] = widget_value
        if st.session_state.get(f"{key}_power_off", False):
            observations["POWER_OFF_CONFIRMED"] = True
        else:
            observations.pop("POWER_OFF_CONFIRMED", None)
    inventory = component_inventory or {}
    rows = _component_rows(components, inventory)
    confirmed_ids = sorted(row["component_id"] for row in rows if row.get("presence_status") == "CONFIRMED")
    measurements = get_measurements(
        st.session_state, project_id=project.project_id, board_id=project.board_id or "",
    )
    if isinstance(board_registry, dict):
        from circuitcare.frontend.board_component_registry import project_measurements_to_registry
        projected, measurements = project_measurements_to_registry(measurements, board_registry)
        # Unresolved/retired identities stay in the measurement archive and UI,
        # but are not evidence for the current diagnosis.
    elif component_id_aliases is not None:
        measurements = [replace(
            row, component_id=component_id_aliases.get(row.component_id) or "",
            component_identity_status=(row.component_identity_status
                                       if component_id_aliases.get(row.component_id) else "UNKNOWN"),
        ) for row in measurements]
    silkscreen = st.session_state.get("circuitcare_v4_silkscreen_labels", [])
    if not isinstance(silkscreen, list):
        silkscreen = []
    input_fingerprint = _diagnosis_fingerprint(
        project, components, selected_codes, symptom_text, observations, state.get("answers", {}),
        measurements, silkscreen, st.session_state, confirmed_ids,
        bool(inventory_complete), inventory, analysis_target_id,
    )
    llm_enabled = bool(st.session_state.get(f"{key}_llm_enabled", False))
    selected_model = st.session_state.get(f"{key}_model")
    fingerprints = build_analysis_fingerprints(
        input_payload={"input_sha256": input_fingerprint}, measurements=measurements,
        extra_rules_payload={"llm_enabled": llm_enabled,
                             "llm_model": selected_model if llm_enabled else None},
    )
    scope = _analysis_scope(project, st.session_state, analysis_target_id)
    record = get_current_state(st.session_state, scope)
    legacy_matches = (
        state.get("input_fingerprint") == input_fingerprint
        and state.get("measurement_fingerprint") == _measurement_fingerprint(project, measurements)
        and state.get("rules_sha256") == fingerprints.rules_sha256
    )
    if state.get("result") and not record.get("results") and not record.get("fingerprints") and legacy_matches:
        migrate_legacy_result(
            st.session_state, scope=scope, fingerprints=fingerprints,
            result=state["result"], legacy_fingerprint=state.get("input_fingerprint"),
        )
    record = get_current_state(st.session_state, scope, fingerprints)
    return {
        "scope": scope, "fingerprints": fingerprints, "record": record,
        "result": get_current_result(st.session_state, scope=scope, fingerprints=fingerprints),
        "input_fingerprint": input_fingerprint,
        "measurement_fingerprint": _measurement_fingerprint(project, measurements),
        "measurements": measurements, "symptoms": selected_codes,
        "symptom_text": symptom_text, "observations": observations,
        "inventory_complete": bool(inventory_complete), "components": components,
        "component_inventory": inventory, "llm_enabled": llm_enabled,
        "selected_model": selected_model,
        "component_id_aliases": component_id_aliases, "board_registry": board_registry,
    }


def get_current_diagnosis_result(
    st: Any, project: ProjectContext, *, components: tuple[Any, ...] = (),
    component_inventory: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    return current_analysis_snapshot(
        st, project, components=components, component_inventory=component_inventory,
    )["result"]


def _measurement_fingerprint(project: ProjectContext, measurements: list[Any]) -> str:
    return build_measurement_fingerprint(measurements)


def _sync_component_candidates(st: Any, project: ProjectContext, result: dict[str, Any]) -> None:
    projects = dict(st.session_state.get("circuitcare_v6_diagnosis_candidates_by_scope", {}))
    diagnosis_state = st.session_state.get(_project_key(project), {})
    scope = _analysis_scope(project, st.session_state)
    projects[scope.key] = build_overlay_state(
        project_id=project.project_id, board_id=project.board_id,
        component_map_version=st.session_state.get("circuitcare_v4_selection_map_key"),
        result={**result, "input_fingerprint": diagnosis_state.get("input_fingerprint")},
        previous=projects.get(scope.key),
    )
    st.session_state["circuitcare_v6_diagnosis_candidates_by_scope"] = projects


def _focus_board_component(st: Any, project: ProjectContext, registry: dict[str, Any] | None,
                           board_component_id: str) -> bool:
    rows = registry.get("components", []) if isinstance(registry, dict) else []
    row = next((item for item in rows if item.get("board_component_id") == board_component_id), None)
    sources = row.get("source_components", []) if row else []
    if not sources:
        return False
    source = sources[0]
    photo_id, local_id = source.get("photo_id"), source.get("component_id")
    st.session_state["circuitcare_v2_photo_choice"] = photo_id
    update_selection(st.session_state, project=project,
                     active_photo_id=photo_id, selected_component=local_id)
    st.session_state[f"cc_analysis_viewer_open_{project.project_id}"] = True
    return True


def _component_rows(components: tuple[Any, ...],
                    component_inventory: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    inventory = component_inventory or {}
    result = []
    for item in components:
        review_owned = getattr(item, "overlay_source", None) in {"yolo", "manual", "qwen_suggestion", "board_registry"}
        presence_status = ("CONFIRMED" if item.identification_status == "USER_CONFIRMED" else "UNCONFIRMED") \
            if review_owned else inventory.get(item.component_id, {}).get("presence_status", "UNCONFIRMED")
        type_status = ("USER_CONFIRMED" if item.identification_status == "USER_CONFIRMED" else "UNCONFIRMED") \
            if review_owned else inventory.get(item.component_id, {}).get(
                "component_type_status", getattr(item, "component_type_status", "UNCONFIRMED"))
        roles = (list(getattr(item, "confirmed_functional_roles", ()) or ())
                 if review_owned else inventory.get(item.component_id, {}).get("confirmed_functional_roles", []))
        result.append({"component_id": item.component_id, "component_type": item.component_type,
             "bbox_xyxy": (list(item.bbox_xyxy) if getattr(item, "bbox_xyxy", None) else None),
             "ocr_marking": item.ocr_marking,
             "component_info": getattr(item, "component_info", None),
             "board_component_id": getattr(item, "board_component_id", None),
             "source_photo_ids": list(getattr(item, "source_photo_ids", ()) or ()),
             "identity_status": getattr(item, "identity_status", None),
             "presence_status": presence_status,
             "component_type_status": type_status,
             "part_identity_status": inventory.get(item.component_id, {}).get("part_identity_status", "UNCONFIRMED"),
             "confirmed_functional_roles": roles,
             "role_status": "CONFIRMED" if roles else "UNKNOWN",
             "role_evidence": ("사용자가 선택한 기능 역할",) if roles else ()})
    return result


def _current_analysis_context(project: ProjectContext, state: Any,
                              component_inventory: dict[str, dict[str, Any]]) -> dict[str, Any]:
    registry = state.get("circuitcare_v7_active_board_registry", {})
    statuses = {
        str(row.get("board_component_id")): "USER_CONFIRMED"
        for row in registry.get("components", [])
        if isinstance(registry, dict) and row.get("presence_status") == "CONFIRMED"
    } if isinstance(registry, dict) else {
        component_id: "USER_CONFIRMED"
        for component_id, row in component_inventory.items()
        if isinstance(row, dict) and row.get("presence_status") == "CONFIRMED"
    }
    return {
        "project_id": project.project_id,
        "board_id": project.board_id or "",
        "component_map_version": (registry.get("version") if isinstance(registry, dict)
                                  else state.get("circuitcare_v4_selection_map_key")),
        "component_identity_statuses": statuses,
    }


def _combined_observations(state: dict[str, Any], answers: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = list(state.get("parsed_observations", []))
    rows.extend({"code": code, "value": value, "status": "observed", "source": "user_report"}
                for code, value in state.get("manual_observations", {}).items())
    rows.extend({"code": code, "value": value, "status": "observed", "source": "follow_up_answer"}
                for code, value in (answers if answers is not None else state.get("answers", {})).items())
    return rows


def _persist_diagnosis(project: ProjectContext, state: dict[str, Any], result: dict[str, Any],
                       components: tuple[Any, ...], observations: list[dict[str, Any]]) -> None:
    target_id = str(state.get("analysis_target_id") or "target:default")
    ids_by_target = state.setdefault("diagnosis_ids_by_target", {})
    legacy_id = state.get("diagnosis_id") if state.get("analysis_target_id") in (None, target_id) else None
    diagnosis_id = ids_by_target.setdefault(target_id, legacy_id or str(uuid4()))
    state["diagnosis_id"] = diagnosis_id  # legacy readers still use the most recent ID
    result["diagnosis_id"] = diagnosis_id
    component_records = [ComponentRecord.from_mapping({
        "component_id": item.component_id, "label": item.component_type,
        "bbox_xyxy": (list(item.bbox_xyxy) if getattr(item, "bbox_xyxy", None) else None),
        "detection_confidence": item.detection_confidence,
        "marking": item.ocr_marking,
    }).to_dict() for item in components]
    normalized_observations = [Observation.from_dict(row).to_dict() for row in observations]
    result["component_snapshot"] = component_records
    db_path = Path(__file__).resolve().parents[4] / "data" / "private" / "diagnosis" / "diagnosis.sqlite3"
    ObservationStore(db_path).save(
        diagnosis_id=diagnosis_id, project_id=project.project_id, board_id=project.board_id,
        status=result.get("status", "UNKNOWN"), result=result, observations=normalized_observations,
        input_fingerprint=state.get("input_fingerprint"),
    )


def render_symptom_diagnosis_panel(
    st: Any, project: ProjectContext, *, components: tuple[Any, ...] = (), preview: bool = False,
    component_inventory: dict[str, dict[str, Any]] | None = None,
    analysis_target_id: str | None = None,
    component_id_aliases: dict[str, str | None] | None = None,
    board_registry: dict[str, Any] | None = None,
) -> None:
    """Collect symptoms, run local rules, and render candidates and safe next questions."""
    kb = load_knowledge()
    symptom_items = kb["symptom_ontology"]["symptoms"]
    key = _project_key(project)
    state = st.session_state.setdefault(key, {"manual_observations": {}, "result": None})
    state.setdefault("manual_observations", state.pop("observations", {}))
    if state.pop("measurement_refresh_notice", False):
        st.info("측정 기록이 변경되어 기존 결과를 무효화하고 새 입력으로 원인 후보를 다시 계산했습니다. 측정 근거와 판정 상태를 확인하세요.")
    if state.get("invalidated_reason"):
        st.warning(f"입력이 변경되어 이전 분석 결과를 무효화했습니다 · {state['invalidated_reason']} · 다시 분석해 주세요.")
    st.markdown("#### 증상 입력 및 원인 분석")
    st.caption("자연어로 증상을 설명하세요. 원인 후보는 검사 전 가설이며 실제 고장 확률로 해석하지 않습니다.")
    description_key = f"{key}_description"
    if description_key not in st.session_state:
        st.session_state[description_key] = project.symptom_description or ""
    symptom_text = st.text_area("증상 설명", key=description_key,
                                 placeholder="예: 전원은 켜지지만 LED가 들어오지 않고 버저가 울리지 않음", height=82)
    code_to_label = {code: item["label"] for code, item in symptom_items.items()}
    symptom_widget_key = f"{key}_symptoms"
    if symptom_widget_key not in st.session_state:
        st.session_state[symptom_widget_key] = [
            code_to_label[code] for code in state.get("symptoms", []) if code in code_to_label
        ]
    selected_labels = st.session_state.get(symptom_widget_key, [])
    selected_codes = [code for code, label in code_to_label.items() if label in selected_labels]
    component_inventory = component_inventory or {}
    component_rows = _component_rows(components, component_inventory)
    confirmed_component_ids = sorted(
        row["component_id"] for row in component_rows if row.get("presence_status") == "CONFIRMED"
    )
    st.caption(f"PCB 검수 목록 · 확인 {len(confirmed_component_ids)} / 표시 {len(component_rows)}개")
    with st.expander("추가 관측·고급 분석 설정", expanded=False):
        st.multiselect("구조화된 증상", list(code_to_label.values()), key=symptom_widget_key,
                       help="자연어 설명 외에 분류된 증상을 추가합니다.")
        inventory_complete = st.checkbox(
            "PCB 전체 부품 목록을 확인 완료했습니다",
            value=bool(state.get("inventory_complete", False)),
            key=f"{key}_inventory_complete",
            help="누락 여부 분석에만 사용됩니다. 확실하지 않으면 선택하지 마세요.",
        )
        yes_no_unknown = ("모름", "예", "아니오")
        evidence_fields = {
            "POWER_SOURCE_VERIFIED": "외부 전원/배터리 입력이 정상임",
            "BUZZER_VERIFIED_FAILED": "버저 자체가 고장임을 별도 확인함",
            "BUZZER_VERIFIED_WORKING": "버저 자체는 정상임을 별도 확인함",
            "BUZZER_DRIVE_ABNORMAL": "버저 구동 신호 이상을 확인함",
            "BUZZER_DRIVE_VERIFIED_NORMAL": "버저 구동 신호 정상임을 확인함",
            "VISIBLE_SOLDER_DAMAGE": "납땜 손상을 사진/육안으로 확인함",
        }
        observations = dict(state.get("manual_observations", {}))
        for fact, label in evidence_fields.items():
            current = observations.get(fact)
            default = "모름" if current is None else ("예" if current is True else "아니오")
            answer = st.selectbox(label, yes_no_unknown, index=yes_no_unknown.index(default), key=f"{key}_{fact}")
            observations[fact] = True if answer == "예" else False if answer == "아니오" else None
        power_off = st.checkbox("전원을 차단했고 기판이 식은 상태임", key=f"{key}_power_off")
        if power_off:
            observations["POWER_OFF_CONFIRMED"] = True
        else:
            observations.pop("POWER_OFF_CONFIRMED", None)
        model_cache = state.get("ollama_models")
        checked_at = float(state.get("ollama_checked_at", 0))
        if time.time() - checked_at > 30:
            try:
                local_client = OllamaClient(timeout_seconds=2)
                model_cache = [str(item.get("name")) for item in local_client.list_models() if item.get("name")]
                state["ollama_error"] = None
            except (LocalLLMError, ValueError) as exc:
                model_cache = []
                state["ollama_error"] = str(exc)
            state["ollama_models"] = model_cache
            state["ollama_checked_at"] = time.time()
            st.session_state[key] = state
        if model_cache:
            selected_model = st.selectbox("로컬 LLM", model_cache, key=f"{key}_model")
            llm_key = f"{key}_llm_enabled"
            if llm_key not in st.session_state:
                st.session_state[llm_key] = True
            llm_enabled = st.checkbox("LLM 설명 보조 사용", key=llm_key)
        else:
            st.caption("Ollama를 사용할 수 없어 규칙 기반 분석으로 실행합니다.")
    snapshot = current_analysis_snapshot(
        st, project, components=components, component_inventory=component_inventory,
        symptom_text=symptom_text, selected_codes=selected_codes,
        observations=observations, inventory_complete=inventory_complete,
        analysis_target_id=analysis_target_id, component_id_aliases=component_id_aliases,
        board_registry=board_registry,
    )
    measurements = snapshot["measurements"]
    silkscreen_labels = st.session_state.get("circuitcare_v4_silkscreen_labels", [])
    if not isinstance(silkscreen_labels, list):
        silkscreen_labels = []
    fingerprint = snapshot["input_fingerprint"]
    measurement_fingerprint = snapshot["measurement_fingerprint"]
    llm_enabled = snapshot["llm_enabled"]
    selected_model = snapshot["selected_model"]
    analysis_scope = snapshot["scope"]
    analysis_fingerprints = snapshot["fingerprints"]
    confirmed_component_ids = sorted(
        row["component_id"] for row in _component_rows(components, component_inventory)
        if row.get("presence_status") == "CONFIRMED"
    )
    central_state = get_current_state(st.session_state, analysis_scope)
    legacy_result_matches_inputs = (
        state.get("input_fingerprint") == fingerprint
        and state.get("measurement_fingerprint") == measurement_fingerprint
        and state.get("rules_sha256") == analysis_fingerprints.rules_sha256
    )
    if state.get("result") and not central_state.get("results"):
        migrated = False
        if legacy_result_matches_inputs:
            migrated = migrate_legacy_result(
                st.session_state, scope=analysis_scope, fingerprints=analysis_fingerprints,
                result=state["result"], legacy_fingerprint=state.get("input_fingerprint"),
            )
        if not migrated:
            legacy_history = list(state.get("history", []))
            legacy_history.append({"result": state["result"],
                                   "reason": "중앙 상태 버전 전환 전의 진단 기록"})
            state["history"] = legacy_history[-100:]
            state["result"] = None
    central_state = get_current_state(st.session_state, analysis_scope, analysis_fingerprints)
    current_result = get_current_result(
        st.session_state, scope=analysis_scope, fingerprints=analysis_fingerprints,
    )
    if state.get("input_fingerprint") != fingerprint or state.get("measurement_fingerprint") != measurement_fingerprint:
        if state.get("result") and not current_result:
            state["invalidated_reason"] = central_state.get("reason") or "분석 입력 변경"
        state["input_fingerprint"] = fingerprint
        state["measurement_fingerprint"] = measurement_fingerprint
        state["result"] = current_result
        state["analysis_status"] = central_state.get("status", "NOT_RUN")
        st.session_state[key] = state
    elif current_result is not None:
        state["result"] = current_result
        state["analysis_status"] = "CURRENT"
    else:
        state["result"] = None
        state["analysis_status"] = central_state.get("status", "NOT_RUN")
    if st.button("원인 후보 분석" if not current_result else "다시 분석", type="primary", disabled=preview,
                 key=f"{key}_run"):
        state["symptoms"] = selected_codes
        state["confirmed_component_ids"] = confirmed_component_ids
        state["inventory_complete"] = inventory_complete
        state["manual_observations"] = {fact: value for fact, value in observations.items() if value is not None}
        run_id = begin_analysis(
            st.session_state, scope=analysis_scope, fingerprints=analysis_fingerprints,
            result_kind="symptom",
        )
        component_rows = _component_rows(components, component_inventory)
        parsed = {"observations": [], "uncertain_facts": [], "parser_status": "SKIPPED"}
        if llm_enabled and selected_model and symptom_text.strip():
            try:
                with st.spinner(f"{selected_model}이 증상을 구조화하는 중…"):
                    parsed = parse_user_report(OllamaClient(timeout_seconds=120), selected_model, symptom_text)
            except (LocalLLMError, ValueError, TypeError, OSError) as exc:
                parsed = {"observations": [], "uncertain_facts": [], "parser_status": "FAILED", "error": str(exc)}
        state["parsed_observations"] = parsed.get("observations", [])
        combined_observations = list(state["parsed_observations"])
        combined_observations.extend({"code": fact, "value": value, "status": "observed", "source": "user_report"}
                                     for fact, value in state["manual_observations"].items())
        combined_observations.extend({"code": fact, "value": value, "status": "observed", "source": "follow_up_answer"}
                                     for fact, value in state.get("answers", {}).items())
        parsed_symptoms = [row["code"] for row in state["parsed_observations"]
                           if row.get("status") in {"observed", "measured"} and row.get("value") is True
                           and row.get("code") in symptom_items]
        try:
            result = diagnose(
                symptoms=list(dict.fromkeys([*selected_codes, *parsed_symptoms])),
                symptom_text=symptom_text,
                observations=combined_observations, components=component_rows,
                confirmed_component_ids=confirmed_component_ids,
                inventory_complete=inventory_complete,
                measurements=[row.to_dict() for row in measurements],
                current_context=_current_analysis_context(project, st.session_state, component_inventory),
                silkscreen_labels=silkscreen_labels,
                connectivity_status="none",
            )
        except Exception as exc:
            fail_analysis(st.session_state, scope=analysis_scope, run_id=run_id, error=str(exc))
            state["result"] = None
            state["analysis_status"] = "FAILED"
            st.session_state[key] = state
            st.error(f"규칙 기반 분석을 완료하지 못했습니다: {exc}")
            return
        result["llm_parser"] = parsed
        if len(result.get("hypotheses", [])) < 3 and llm_enabled and selected_model:
            try:
                with st.spinner(f"{selected_model}이 미등록 원인 가능성을 별도 검토하는 중…"):
                    result["llm_unverified_suggestions"] = suggest_unregistered_causes(
                        OllamaClient(timeout_seconds=120), selected_model,
                        symptom_text=symptom_text, diagnosis=result,
                    )
            except (LocalLLMError, ValueError, TypeError, OSError) as exc:
                result["llm_suggestion_error"] = str(exc)
        if llm_enabled and selected_model and result.get("hypotheses"):
            try:
                with st.spinner(f"{selected_model}이 근거를 정리하는 중…"):
                    result["llm_assist"] = explain_candidates(
                        OllamaClient(timeout_seconds=120), selected_model,
                        symptom_text=symptom_text, diagnosis=result,
                    )
            except (LocalLLMError, ValueError, TypeError, OSError) as exc:
                result["llm_error"] = str(exc)
        if llm_enabled and selected_model and result.get("component_candidates"):
            try:
                with st.spinner(f"{selected_model}이 소자별 검사 이유를 짧게 정리하는 중…"):
                    result["llm_component_explanations"] = explain_component_candidates(
                        OllamaClient(timeout_seconds=120), selected_model,
                        candidates=result["component_candidates"],
                    )
            except (LocalLLMError, ValueError, TypeError, OSError) as exc:
                result["llm_component_error"] = str(exc)
        live_snapshot = current_analysis_snapshot(
            st, project, components=components, component_inventory=component_inventory,
            symptom_text=symptom_text, selected_codes=selected_codes,
            observations=observations, inventory_complete=inventory_complete,
            analysis_target_id=analysis_target_id, component_id_aliases=component_id_aliases,
            board_registry=board_registry,
        )
        committed = commit_analysis_result(
            st.session_state, scope=analysis_scope, run_id=run_id,
            fingerprints=analysis_fingerprints, current_fingerprints=live_snapshot["fingerprints"],
            current_scope=live_snapshot["scope"],
            result=result, result_kind="symptom",
        )
        if not committed:
            state["result"] = None
            state["analysis_status"] = "STALE"
            state["invalidated_reason"] = "분석 중 입력이 바뀌어 이전 버전 결과를 저장하지 않았습니다."
            st.session_state[key] = state
            st.warning(state["invalidated_reason"])
            return
        state["analysis_target_id"] = analysis_scope.target_id
        st.session_state["circuitcare_v7_analysis_target_id"] = analysis_scope.target_id
        _persist_diagnosis(project, state, result, components, combined_observations)
        _sync_component_candidates(st, project, result)
        state["result"] = result
        state["analysis_status"] = "CURRENT"
        state.pop("invalidated_reason", None)
        state["input_fingerprint"] = fingerprint
        state["measurement_fingerprint"] = measurement_fingerprint
        st.session_state[key] = state
        st.rerun()
    result = current_result
    if not result:
        st.info("증상을 입력하고 **원인 후보 분석**을 실행하세요.")
        return
    if result.get("evidence_conflicts"):
        st.warning("서로 다른 입력이 있어 해당 사실은 확정하지 않았습니다. 증상/측정값을 확인해 주세요.")
        for conflict in result["evidence_conflicts"]:
            st.caption(f"충돌 · {conflict['code']} · {conflict['values']} · {conflict['resolution']}")
    if result["status"] == "NEEDS_SYMPTOMS":
        st.info(result["message"])
        return
    inventory = result.get("component_inventory", {})
    if not inventory.get("confirmed_count"):
        st.info("실제 존재한다고 확인한 소자가 없습니다. 소자 내부 고장 후보는 생성되지 않았습니다.")
    if not inventory.get("complete"):
        st.caption("전체 부품 목록은 아직 미확정입니다. 미검출 소자가 실제로 없는지 판단하지 않으며, 선택한 실재 확인 ID만 부품 고장 후보에 사용합니다.")
    st.markdown("#### 원인 후보")
    st.success(result["message"])
    component_candidates = result.get("component_candidates", [])[:3]
    if component_candidates:
        st.markdown("**우선 확인할 실제 검출 소자 · 최대 3개**")
        explanations = (result.get("llm_component_explanations", {}).get("explanations", {})
                        if isinstance(result.get("llm_component_explanations"), dict) else {})
        status_labels = {
            "SUSPECTED": "검사 후보 · 아직 미확정",
            "FAULT_SUPPORTED": "검증된 검사 · 해당 고장 가설 지지",
            "FAULT_REFUTED": "검증된 검사 · 해당 고장 가설 반박",
            "INCONCLUSIVE": "측정 판정 불가",
            "CONFLICTED": "측정 결과 충돌 · 판단 보류",
        }
        for candidate in component_candidates:
            label = status_labels.get(candidate.get("visual_status"), "판정 불가")
            st.markdown(f"**{candidate['candidate_rank']}. {candidate['component_id']}** · "
                        f"{candidate['component_type']} · {candidate['fault_title']} · {label}")
            reason = explanations.get(candidate["component_id"])
            if reason:
                st.caption(reason)
            with st.expander("이 소자 후보의 근거·판정", expanded=False):
                supporting = candidate.get("supporting_evidence") or candidate.get("evidence", [])
                opposing = candidate.get("opposing_evidence", [])
                st.caption("뒷받침 · " + (" · ".join(supporting[:5]) if supporting else "직접 근거 없음"))
                st.caption("반대 근거 · " + (" · ".join(opposing[:5]) if opposing else "확인된 반대 근거 없음"))
                if candidate.get("measurement_reason"):
                    st.caption("측정 판정 · " + str(candidate["measurement_reason"]))
                elif candidate.get("recommended_tests"):
                    st.caption("다음 검사 · " + " · ".join(candidate["recommended_tests"][:3]))
                elif not supporting:
                    st.caption("부품 종류와 증상으로 좁힌 미확정 검사 후보입니다.")
            if any(item.component_id == candidate.get("component_id") for item in components):
                if st.button("PCB에서 위치 보기", key=f"cc_symptom_candidate_focus_{project.project_id}_{candidate.get('component_id')}"):
                    focused = (_focus_board_component(st, project, board_registry,
                               str(candidate.get("component_id"))) if board_registry else False)
                    if not focused:
                        update_selection(st.session_state, project=project,
                                         selected_component=candidate.get("component_id"))
                        st.session_state[f"cc_analysis_viewer_open_{project.project_id}"] = True
                    st.rerun()
        if result.get("llm_component_explanations"):
            st.caption(f"후보별 설명 · {result['llm_component_explanations'].get('model', '')} · 후보 추가/확정 없음")
        elif result.get("llm_component_error"):
            st.caption(f"LLM 후보 설명을 만들지 못해 규칙 근거를 표시합니다 · {result['llm_component_error']}")
    else:
        if inventory.get("unconfirmed_detection_ids"):
            st.caption("확인되지 않은 자동 검출은 고장 후보에서 제외했습니다. 실제 존재하는 부품만 위 목록에서 선택해 다시 추론하세요.")
        else:
            st.caption("현재 근거로 개별 부품을 좁힐 수 없습니다. 소자를 임의로 강조하지 않았습니다.")
    if result.get("llm_assist"):
        assist = result["llm_assist"]
        st.markdown("**로컬 LLM 설명 보조**")
        st.write(assist["summary"])
        st.caption(f"모델 · {assist['model']} · 후보 추가/확정 없음")
        for note in assist.get("candidate_notes", []):
            st.caption(f"{note['fault_id']} · {note['explanation']}")
            st.caption("사용 근거 · " + " · ".join(note.get("evidence_used", [])))
    elif result.get("llm_error"):
        st.warning(f"로컬 LLM 설명 보조 실패 · {result['llm_error']} · 규칙 기반 후보는 유지됩니다.")
    parsed = result.get("llm_parser", {})
    if parsed.get("observations"):
        with st.expander("LLM이 구조화한 사용자 관측값", expanded=False):
            for row in parsed["observations"]:
                label = symptom_items.get(row["code"], kb["symptom_ontology"].get("observations", {}).get(row["code"], {})).get("label", row["code"])
                st.caption(f"{label} · {'예' if row['value'] else '아니오'} · 원문: {row.get('evidence_quote', '')}")
            for item in parsed.get("uncertain_facts", []):
                st.caption(f"미분류/불확실 · {item}")
    if parsed.get("error"):
        st.caption(f"LLM 증상 파싱 실패 · {parsed['error']} · 키워드 규칙으로 처리했습니다.")
    if result.get("llm_unverified_suggestions"):
        with st.expander("LLM 미등록 원인 제안 · 순위 미반영", expanded=False):
            for suggestion in result["llm_unverified_suggestions"]:
                st.markdown(f"**{suggestion['title']}** · 미검증 제안")
                st.caption(suggestion["reason"])
                if suggestion["evidence_used"]:
                    st.caption("참조 관측 · " + " · ".join(suggestion["evidence_used"]))
    if result.get("matched_rules"):
        st.caption("적용 규칙 · " + " · ".join(row["name"] for row in result["matched_rules"]))
    groups = result.get("hypothesis_groups", {})
    for group_key, group_label in (("component", "소자 내부·기능 블록 원인"),
                                   ("board", "기판·납땜 원인"),
                                   ("system", "외부·시스템 원인")):
        rows = groups.get(group_key, []) if isinstance(groups, dict) else []
        if not rows:
            continue
        st.markdown(f"**{group_label}**")
        for index, row in enumerate(rows[:5], start=1):
            title = f"{index}. {row['title']} · 미확정 후보"
            with st.expander(title, expanded=(group_key == "component" and index == 1)):
                if row["supporting_evidence"]:
                    st.markdown("**뒷받침한 사실** · " + " · ".join(row["supporting_evidence"]))
                else:
                    st.caption("직접 뒷받침하는 사실은 아직 없습니다.")
                if row["opposing_evidence"]:
                    st.markdown("**반대 사실** · " + " · ".join(row["opposing_evidence"]))
                if row.get("related_component_ids"):
                    st.caption("확인된 관련 부품 · " + " · ".join(row["related_component_ids"]))
                st.caption("상태 · 검증 전 후보. 이 결과만으로 부품 고장을 확정하지 않습니다.")
    if result.get("functional_blocks"):
        with st.expander("부품 기능 역할 추정", expanded=False):
            for block in result["functional_blocks"]:
                st.caption(f"{block['component_id']} · {block['component_type']} → "
                           f"{', '.join(block['candidate_blocks']) or '역할 미분류'} · 배치/OCR 기반 미확정 추정")
                identity = block.get("ic_identity")
                if identity:
                    st.caption(f"IC 각인 {identity.get('marking') or '없음'} · "
                               f"{', '.join(identity.get('role_candidates', [])) or '기능 미분류'} · "
                               f"{identity.get('status')} · 데이터시트 검증 전")
    if result.get("measurement_evidence"):
        with st.expander("진단에 연결된 측정 기록", expanded=False):
            for row in result["measurement_evidence"]:
                st.caption(f"{row['component_id']} · {row['measurement_type']} {row['value']} {row['unit']} · "
                           f"{row['verification_status']}")
            st.caption("빨강/초록은 검사 프로파일과 출처가 검증된 측정이 고장 가설을 직접 지지/반박할 때만 표시됩니다.")
    if result.get("joint_candidates"):
        with st.expander("소자 내부 고장과 분리한 납땜·기판 후보", expanded=False):
            for candidate in result["joint_candidates"]:
                st.caption(f"{candidate['title']} · 미확정 · " + " · ".join(candidate.get("evidence", [])))
    questions = [item for item in result.get("next_tests", []) if item.get("type") == "question"]
    if questions:
        st.markdown("**다음 확인 질문**")
        answers: dict[str, Any] = {}
        for question in questions[:3]:
            test_id = question["test_id"]
            answer = st.selectbox(question["question"], ("미답변", "예", "아니오"),
                                  key=f"{key}_answer_{test_id}")
            if answer in {"예", "아니오"}:
                outcome = question.get("results", {}).get("yes" if answer == "예" else "no", {})
                answers.update(outcome.get("observations", {}))
        if st.button("답변 반영하고 다시 추론", key=f"{key}_apply_answers", disabled=preview):
            state["answers"] = answers
            answer_input_fingerprint = _diagnosis_fingerprint(
                project, components, selected_codes, symptom_text,
                observations, answers, measurements,
                silkscreen_labels, st.session_state, confirmed_component_ids, inventory_complete,
                component_inventory, analysis_target_id,
            )
            answer_fingerprints = build_analysis_fingerprints(
                input_payload={"input_sha256": answer_input_fingerprint}, measurements=measurements,
                extra_rules_payload={"llm_enabled": llm_enabled,
                                     "llm_model": selected_model if llm_enabled else None},
            )
            followup_run_id = begin_analysis(
                st.session_state, scope=analysis_scope, fingerprints=answer_fingerprints,
                result_kind="symptom",
            )
            combined_observations = _combined_observations(state, answers)
            try:
                state["result"] = diagnose(
                    symptoms=state.get("symptoms", []), symptom_text="",
                    observations=combined_observations,
                    components=_component_rows(components, component_inventory),
                    confirmed_component_ids=confirmed_component_ids,
                    inventory_complete=inventory_complete,
                    measurements=[row.to_dict() for row in measurements],
                    current_context=_current_analysis_context(project, st.session_state, component_inventory),
                    silkscreen_labels=silkscreen_labels, connectivity_status="none",
                )
            except Exception as exc:
                fail_analysis(st.session_state, scope=analysis_scope, run_id=followup_run_id, error=str(exc))
                state["result"] = None
                state["analysis_status"] = "FAILED"
                st.session_state[key] = state
                st.error(f"추가 답변 분석을 완료하지 못했습니다: {exc}")
                return
            if llm_enabled and selected_model and state["result"].get("hypotheses"):
                try:
                    with st.spinner(f"{selected_model}이 갱신된 근거를 정리하는 중…"):
                        state["result"]["llm_assist"] = explain_candidates(
                            OllamaClient(timeout_seconds=120), selected_model,
                            symptom_text=symptom_text, diagnosis=state["result"],
                        )
                except (LocalLLMError, ValueError, TypeError, OSError) as exc:
                    state["result"]["llm_error"] = str(exc)
            if llm_enabled and selected_model and state["result"].get("component_candidates"):
                try:
                    state["result"]["llm_component_explanations"] = explain_component_candidates(
                        OllamaClient(timeout_seconds=120), selected_model,
                        candidates=state["result"]["component_candidates"],
                    )
                except (LocalLLMError, ValueError, TypeError, OSError) as exc:
                    state["result"]["llm_component_error"] = str(exc)
            answer_result = state["result"]
            live_snapshot = current_analysis_snapshot(
                st, project, components=components, component_inventory=component_inventory,
                symptom_text=symptom_text, selected_codes=selected_codes,
                observations=observations, inventory_complete=inventory_complete,
                analysis_target_id=analysis_target_id, component_id_aliases=component_id_aliases,
                board_registry=board_registry,
            )
            committed = commit_analysis_result(
                st.session_state, scope=analysis_scope, run_id=followup_run_id,
                fingerprints=answer_fingerprints, current_fingerprints=live_snapshot["fingerprints"],
                current_scope=live_snapshot["scope"],
                result=answer_result, result_kind="symptom",
            )
            if not committed:
                state["result"] = None
                state["analysis_status"] = "STALE"
                state["invalidated_reason"] = "추가 답변 처리 중 분석 입력이 변경되어 결과를 저장하지 않았습니다."
                st.session_state[key] = state
                st.warning(state["invalidated_reason"])
                return
            state["input_fingerprint"] = answer_input_fingerprint
            state["measurement_fingerprint"] = measurement_fingerprint
            state["analysis_status"] = "CURRENT"
            state["analysis_target_id"] = analysis_scope.target_id
            st.session_state["circuitcare_v7_analysis_target_id"] = analysis_scope.target_id
            _persist_diagnosis(project, state, answer_result, components, combined_observations)
            _sync_component_candidates(st, project, answer_result)
            st.session_state[key] = state
            st.rerun()
    visuals = [item for item in result.get("next_tests", []) if item.get("type") == "visual_inspection"]
    if visuals:
        st.caption("육안 점검은 전원 차단 및 냉각을 확인한 뒤에만 표시됩니다.")
        for item in visuals:
            st.write(f"**{item['name']}** · {item['question']}")
