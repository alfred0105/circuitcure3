from types import SimpleNamespace
from pathlib import Path

import pytest

from circuitcare.dashboard.views import unified_workspace as workspace
from circuitcare.frontend.models import ProjectContext
from circuitcare.frontend.workbench_state import SELECTION_KEY, enter_workbench
from circuitcare.dashboard.views.symptom_diagnosis_panel import candidate_focus_widget_key


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


def test_fixed_viewer_shell_keeps_pcb_visible_and_only_renders_selected_task(monkeypatch):
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

    for mode, expected in (("검수", "review"), ("증상", "증상"), ("원인", "원인"), ("측정", "측정")):
        state = {}
        enter_workbench(state, project)
        state[f"cc_workbench_panel_{project.project_id}"] = mode
        calls.clear()
        workspace.render_workbench_shell(_StreamlitStub(state=state), project)
        assert calls == ["viewer", expected]


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
    assert len(app.radio) == 1
    assert "진단 미실행" in " ".join(item.value for item in app.caption)
