from types import SimpleNamespace
from pathlib import Path

import pytest

from circuitcare.dashboard.views import unified_workspace as workspace
from circuitcare.dashboard.views import symptom_diagnosis_panel as symptoms
from circuitcare.dashboard.views import inspection_panel as inspection
from circuitcare.frontend.models import ProjectContext
from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.frontend.workbench_state import SELECTION_KEY, enter_workbench
from circuitcare.dashboard.views.symptom_diagnosis_panel import candidate_focus_widget_key
from circuitcare.dashboard.views.symptom_diagnosis_panel import (
    _queue_measurement_change_rerun, _consume_measurement_change_rerun,
)
from circuitcare.analysis.models import AnalysisStatus


class _Context:
    def button(self, *_args, **_kwargs):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _StreamlitStub:
    def __init__(self, *, state=None, clicked=()):
        self.session_state = state or {}
        self.clicked = set(clicked)
        self.radio_calls = []

    def container(self, **_kwargs):
        return _Context()

    def columns(self, specs, **_kwargs):
        return [_Context() for _ in specs]

    def expander(self, *_args, **_kwargs):
        return _Context()

    def radio(self, label, options, *, key=None, **_kwargs):
        self.radio_calls.append((label, tuple(options), key))
        return self.session_state.get(key, options[0])

    def button(self, _label, *, key=None, **_kwargs):
        return key in self.clicked

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


def test_analysis_workspace_only_renders_selected_panel_instead_of_hidden_tabs(monkeypatch):
    project = ProjectContext(project_id="project-1", board_id="board-1")
    state = {}
    enter_workbench(state, project)
    st = _StreamlitStub(state=state)
    calls = []
    monkeypatch.setattr(workspace, "render_symptom_diagnosis_panel", lambda *_args, **_kwargs: calls.append("symptom"))
    workspace._render_fault_analysis_workspace(st, project, preview=False, mode="증상")

    assert calls == ["symptom"]


def test_cause_candidate_selection_opens_same_pcb_viewer_and_updates_shared_selection():
    project = ProjectContext(project_id="project-2", board_id="board-2")
    state = {"circuitcare_v4_current_components": (SimpleNamespace(component_id="U3"),)}
    enter_workbench(state, project)
    button_key = candidate_focus_widget_key("analysis_dashboard", "project-2", "board-2", "U3", "")
    st = _StreamlitStub(state=state, clicked={button_key})
    result = {"component_candidates": [{
        "component_id": "U3", "component_type": "ic", "fault_title": "전원 입력 이상",
        "candidate_rank": 1, "supporting_evidence": ["출력이 없음"],
        "opposing_evidence": [],
    }]}

    workspace._render_cause_dashboard(st, project, result)

    assert state[SELECTION_KEY].selected_component == "U3"


def test_board_candidate_focus_changes_to_its_source_photo_and_photo_local_box():
    project = ProjectContext("project-board", "board-board")
    state = {}
    enter_workbench(state, project)
    registry = {"components": [{
        "board_component_id": "bc-1",
        "source_components": [{"photo_id": "close-2", "component_id": "det-17"}],
    }]}
    st = _StreamlitStub(state=state)

    assert workspace._focus_board_registry_component(st, project, registry, "bc-1")
    assert state["circuitcare_v2_photo_choice"] == "close-2"
    assert state[SELECTION_KEY].active_photo_id == "close-2"
    assert state[SELECTION_KEY].selected_component == "det-17"


def test_measurement_target_changes_shared_selection_only_when_callback_runs():
    project = ProjectContext("project-measure", "board-measure")
    state = {}
    enter_workbench(state, project)
    registry = {"components": [{
        "board_component_id": "bc-2",
        "source_components": [{"photo_id": "back-photo", "component_id": "R8"}],
    }]}
    st = _StreamlitStub(state=state)
    target_key = "measurement-target"
    state[target_key] = "bc-2"

    workspace._handle_measurement_target_change(state, project, registry, target_key)

    assert state[workspace.BOARD_COMPONENT_SELECTION_KEY] == "bc-2"
    assert state["circuitcare_v2_photo_choice"] == "back-photo"
    assert state[SELECTION_KEY].selected_component == "R8"


def test_fixed_viewer_shell_keeps_pcb_visible_and_renders_single_diagnosis_flow(monkeypatch):
    project = ProjectContext("project-shell", "board-shell")
    registry = {"project_id": project.project_id, "board_id": project.board_id,
                "version": "board-registry:v1", "components": []}
    monkeypatch.setattr(workspace, "_style", lambda _st: None)
    monkeypatch.setattr(workspace, "_prepare_board_registry_snapshot", lambda *_args, **_kwargs: (registry, {}))
    monkeypatch.setattr(workspace, "current_analysis_snapshot", lambda *_args, **_kwargs: {
        "result": None, "record": {"status": "NOT_RUN"}})
    monkeypatch.setattr(workspace, "_diagnosis_status_label", lambda *_args, **_kwargs: "분석 전")
    calls = []
    monkeypatch.setattr(workspace, "_render_pcb", lambda *_args, **_kwargs: calls.append("viewer"))
    monkeypatch.setattr(workspace, "_render_component_inspector", lambda *_args, **_kwargs: calls.append("review"))
    monkeypatch.setattr(workspace, "_render_fault_analysis_workspace",
                        lambda _st, _project, _preview, **kwargs: calls.append(kwargs["mode"]))

    state = {}
    enter_workbench(state, project)
    state[f"cc_workbench_panel_{project.project_id}"] = "원인"  # ignored legacy navigation state
    workspace.render_workbench_shell(_StreamlitStub(state=state), project)
    assert calls == ["viewer", "review", "통합"]


def test_simple_candidate_list_aggregates_faults_by_physical_component():
    result = {"candidate_fusion": {"top_candidates": [
        {"board_component_id": "BC-1", "fault_id": "F-A", "score": 8},
        {"board_component_id": "BC-1", "fault_id": "F-B", "score": 7},
        {"board_component_id": "BC-2", "fault_id": "F-C", "score": 6},
        {"board_component_id": "BC-3", "fault_id": "F-D", "score": 5},
        {"board_component_id": "BC-4", "fault_id": "F-E", "score": 4},
    ]}}

    candidates = symptoms._simple_candidate_rows(result)

    assert [row["board_component_id"] for row in candidates] == ["BC-1", "BC-2", "BC-3"]
    assert [row["fault_id"] for row in candidates[0]["fault_hypotheses"]] == ["F-A", "F-B"]


def test_auto_qwen_prefers_installed_qwen35_9b_and_caches_lookup(monkeypatch):
    class OllamaStub:
        calls = 0

        def __init__(self, *, timeout_seconds):
            assert timeout_seconds == 2

        def list_models(self):
            OllamaStub.calls += 1
            return [{"name": "qwen2.5:14b"}, {"name": "qwen3.5:9b"}, {"name": "gemma4:12b"}]

    monkeypatch.setattr(symptoms, "OllamaClient", OllamaStub)
    state = {}

    assert symptoms._auto_qwen_model(state) == "qwen3.5:9b"
    assert symptoms._auto_qwen_model(state) == "qwen3.5:9b"
    assert OllamaStub.calls == 1


def test_compact_inspector_renders_selected_component_without_scope_error():
    class InspectorStub(_StreamlitStub):
        def selectbox(self, _label, options, *, key=None, **_kwargs):
            return self.session_state.get(key, options[0])

    project = ProjectContext("compact-p", "compact-b")
    component = SimpleNamespace(
        component_id="R1", board_component_id="BC-R1", component_type="resistor",
        reference_designators=("R1",), ocr_marking="103", bbox_xyxy=(1, 2, 8, 9),
        source_backend_photo_ids=("photo-1",),
    )
    state = {SELECTION_KEY: SimpleNamespace(project_id=project.project_id, selected_component="R1")}

    inspection.render_inspection_panel(
        InspectorStub(state=state), project, components=(component,),
        selected_component=component, compact=True,
    )

    assert state[inspection.USER_SUSPECTS_KEY][f"{project.project_id}|{project.board_id}"] == {}


def test_integrated_workspace_uses_candidate_rows_and_guides_measurement(monkeypatch):
    project = ProjectContext("integrated-p", "integrated-b")
    component = SimpleNamespace(component_id="R1", board_component_id="BC-R1", component_type="resistor")
    result = {"candidate_fusion": {"candidates": [
        {"board_component_id": "BC-R1", "component_id": "R1", "fault_id": "RESISTOR_OPEN"},
    ]}}
    snapshot = {"result": result, "record": {"status": "CURRENT"}}
    monkeypatch.setattr(workspace, "render_symptom_diagnosis_panel", lambda *_a, **_k: None)
    monkeypatch.setattr(workspace, "current_analysis_snapshot", lambda *_a, **_k: snapshot)
    measured = []
    monkeypatch.setattr(workspace, "render_measurement_panel",
                        lambda _st, _project, **kwargs: measured.append(kwargs["selected_component"]))

    workspace._render_fault_analysis_workspace(
        _StreamlitStub(state={"circuitcare_v4_current_components": (component,)}), project,
        preview=False, mode="통합", analysis_snapshot=snapshot,
        photo_snapshot={"components": (component,)}, board_components=(component,),
    )

    # A board-level diagnosis DTO has no photo-local geometry and must not be
    # passed directly into a form that expects a viewer/measurement model.
    assert measured == [None]


def test_integrated_measurement_never_falls_back_to_unrelated_viewer_selection(monkeypatch):
    project = ProjectContext("integrated-target-p", "integrated-target-b")

    def component(local_id, board_id, kind):
        return ComponentOverlayViewModel(
            project_id=project.project_id, board_id=project.board_id, board_side="TOP",
            component_id=local_id, board_component_id=board_id, component_type=kind,
            display_label=local_id, canonical_frame_id="photo-1",
            coordinate_frame="SOURCE_PHOTO_PIXEL", bbox_xyxy=(1, 2, 8, 9), polygon_xy=(),
            source_backend_photo_ids=("photo-1",), observation_ids=(), detection_confidence=.9,
            identification_status="USER_CONFIRMED", identity_match_status="PHOTO_LOCAL",
            ocr_marking=None, possible_part_number=None, evidence=(), overlay_status="CANDIDATE",
        )

    candidate = component("R1", "BC-R1", "resistor")
    unrelated_selection = component("C2", "BC-C2", "capacitor")
    result = {"candidate_fusion": {"candidates": [
        {"board_component_id": "BC-R1", "component_id": "R1", "fault_id": "RESISTOR_OPEN"},
    ]}}
    snapshot = {"result": result, "record": {"status": "CURRENT"}}
    monkeypatch.setattr(workspace, "render_symptom_diagnosis_panel", lambda *_a, **_k: None)
    monkeypatch.setattr(workspace, "current_analysis_snapshot", lambda *_a, **_k: snapshot)
    measured = []
    monkeypatch.setattr(workspace, "render_measurement_panel",
                        lambda _st, _project, **kwargs: measured.append(kwargs["selected_component"]))
    state = {
        "circuitcare_v4_current_components": (candidate, unrelated_selection),
        SELECTION_KEY: SimpleNamespace(selected_component="C2"),
        workspace.BOARD_COMPONENT_SELECTION_KEY: "BC-C2",
    }
    workspace._render_fault_analysis_workspace(
        _StreamlitStub(state=state), project, preview=False, mode="통합", analysis_snapshot=snapshot,
        photo_snapshot={"components": (candidate, unrelated_selection)},
        board_components=(candidate, unrelated_selection),
    )

    assert measured == [candidate]


def test_measurement_change_queues_exactly_one_reanalysis_for_current_fingerprint():
    state = {"measurement_fingerprint": "m-v1", "run_after_measurement_change": None}
    _queue_measurement_change_rerun(
        state, previous_fingerprint="m-v1", current_fingerprint="m-v2",
        prior_analysis_exists=True, current_result=None, central_status=AnalysisStatus.STALE.value,
    )
    assert state["run_after_measurement_change"] == "m-v2"
    assert _consume_measurement_change_rerun(state, "m-v2") is True
    assert state.get("run_after_measurement_change") is None
    assert _consume_measurement_change_rerun(state, "m-v2") is False


def test_measurement_change_does_not_run_diagnosis_for_stale_or_obsolete_fingerprint():
    state = {}
    _queue_measurement_change_rerun(
        state, previous_fingerprint="m-v1", current_fingerprint="m-v2",
        prior_analysis_exists=True, current_result=None, central_status=AnalysisStatus.STALE.value,
    )
    assert _consume_measurement_change_rerun(state, "m-v3") is False
    assert state.get("run_after_measurement_change") is None
    _queue_measurement_change_rerun(
        state, previous_fingerprint="m-v1", current_fingerprint="m-v2",
        prior_analysis_exists=True, current_result={"status": "CURRENT"},
        central_status=AnalysisStatus.CURRENT.value,
    )
    assert state.get("run_after_measurement_change") is None


def test_streamlit_measurement_submission_automatically_reexecutes_diagnosis(monkeypatch):
    app_testing = pytest.importorskip("streamlit.testing.v1")
    calls = []

    def diagnose_stub(**kwargs):
        calls.append(list(kwargs.get("measurements", [])))
        return {
            "status": "OK", "message": "mock result", "symptom_codes": ["NO_POWER"],
            "component_candidates": [{
                "component_id": "R1", "board_component_id": "BC-R1",
                "component_type": "resistor", "fault_id": "RESISTOR_FAILURE",
                "candidate_rank": 1, "supporting_evidence": ["입력 증상"],
                "opposing_evidence": [], "recommended_tests": ["isolated_resistance"],
            }],
            "hypotheses": [], "next_tests": [],
        }

    class OllamaStub:
        def __init__(self, *, timeout_seconds):
            pass

        def list_models(self):
            return []

    monkeypatch.setattr(symptoms, "diagnose", diagnose_stub)
    monkeypatch.setattr(symptoms, "OllamaClient", OllamaStub)
    app_path = Path(__file__).parent / "e2e" / "f5_measurement_reanalysis_app.py"
    app = app_testing.AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    next(button for button in app.button if button.label == "AI 진단 시작").click().run()
    assert not app.exception
    assert len(calls) == 1

    app.text_input(key="cc_v5_value_f5-reanalysis-apptest-f5-reanalysis-board-BC-R1-RESISTOR_FAILURE").set_value("10")
    next(item for item in app.text_input if item.label == "측정 대상/단자").set_value("R1 양단")
    app.button(key="FormSubmitter:cc_v5_add_measurement_f5-reanalysis-apptest-f5-reanalysis-board-BC-R1-RESISTOR_FAILURE-측정 기록 저장").click().run()

    assert not app.exception
    assert len(calls) == 2
    assert calls[0] == []
    assert len(calls[1]) == 1
    assert calls[1][0]["component_id"] == "R1"
    app.run()
    assert not app.exception
    assert len(calls) == 2


def test_real_streamlit_app_test_renders_workbench_without_runtime_errors():
    app_testing = pytest.importorskip("streamlit.testing.v1")
    from circuitcare.frontend.state import INTAKE_IMAGES_KEY, INTAKE_PROJECT_KEY
    from circuitcare.frontend.workbench_state import SCREEN_KEY

    app = app_testing.AppTest.from_file(str(Path(__file__).parents[1] / "src/circuitcare/ui.py"),
                                        default_timeout=30)
    project = ProjectContext("apptest-project", "apptest-board")
    app.session_state[INTAKE_PROJECT_KEY] = project
    app.session_state[INTAKE_IMAGES_KEY] = []
    app.session_state[SCREEN_KEY] = "workbench"
    app.session_state[f"cc_workbench_panel_{project.project_id}"] = "원인"
    app.run()

    assert not app.exception
    assert len(app.radio) == 1  # symptom prompt only; no workbench-mode selector
    assert not any(item.label == "작업 패널" for item in app.radio)
    assert not any(item.label == "후보 표시 전략" for item in app.selectbox)
    assert "진단 미실행" in " ".join(item.value for item in app.caption)
