from types import SimpleNamespace

from circuitcare.dashboard.views import unified_workspace as workspace
from circuitcare.frontend.models import ProjectContext
from circuitcare.frontend.workbench_state import SELECTION_KEY, enter_workbench


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _StreamlitStub:
    def __init__(self, *, state=None, clicked=()):
        self.session_state = state or {}
        self.clicked = set(clicked)
        self.tab_labels = ()

    def container(self, **_kwargs):
        return _Context()

    def expander(self, *_args, **_kwargs):
        return _Context()

    def tabs(self, labels):
        self.tab_labels = tuple(labels)
        return tuple(_Context() for _ in labels)

    def button(self, _label, *, key=None, **_kwargs):
        return key in self.clicked

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


def test_analysis_workspace_has_three_stages_and_keeps_viewer_available(monkeypatch):
    project = ProjectContext(project_id="project-1", board_id="board-1")
    state = {}
    enter_workbench(state, project)
    st = _StreamlitStub(state=state)
    calls = []
    monkeypatch.setattr(workspace, "render_symptom_diagnosis_panel", lambda *_args, **_kwargs: calls.append("symptom"))
    monkeypatch.setattr(workspace, "render_measurement_panel", lambda *_args, **_kwargs: calls.append("measurement"))
    monkeypatch.setattr(workspace, "render_diagnosis_panel", lambda *_args, **_kwargs: calls.append("diagnosis"))
    monkeypatch.setattr(workspace, "_render_pcb", lambda *_args, **_kwargs: calls.append("viewer"))
    monkeypatch.setattr(workspace, "_render_selected_component_summary", lambda *_args, **_kwargs: calls.append("selection"))

    workspace._render_fault_analysis_workspace(st, project, preview=False)

    assert st.tab_labels == ("1 · 증상 입력", "2 · 원인 분석", "3 · 측정 검증")
    assert calls == ["symptom", "measurement", "diagnosis", "viewer", "selection"]


def test_cause_candidate_selection_opens_same_pcb_viewer_and_updates_shared_selection():
    project = ProjectContext(project_id="project-2", board_id="board-2")
    state = {"circuitcare_v4_current_components": (SimpleNamespace(component_id="U3"),)}
    enter_workbench(state, project)
    st = _StreamlitStub(state=state, clicked={"cc_analysis_dashboard_focus_project-2_U3"})
    result = {"component_candidates": [{
        "component_id": "U3", "component_type": "ic", "fault_title": "전원 입력 이상",
        "candidate_rank": 1, "supporting_evidence": ["출력이 없음"],
        "opposing_evidence": [],
    }]}

    workspace._render_cause_dashboard(st, project, result)

    assert state[SELECTION_KEY].selected_component == "U3"
    assert state["cc_analysis_viewer_open_project-2"] is True


def test_review_navigation_callback_sets_radio_state_before_next_render():
    state = {"cc_workbench_view_project-3": "고장 분석"}

    workspace._set_workbench_view(state, project_id="project-3", view="PCB 소자 검수")

    assert state["cc_workbench_view_project-3"] == "PCB 소자 검수"
