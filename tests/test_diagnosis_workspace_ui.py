from types import SimpleNamespace
from pathlib import Path

import pytest

from circuitcare.dashboard.views import unified_workspace as workspace
from circuitcare.dashboard.views import symptom_diagnosis_panel as symptoms
from circuitcare.dashboard.views import inspection_panel as inspection
from circuitcare.frontend.models import ProjectContext
from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.frontend.workbench_state import (
    SELECTION_KEY, WORKFLOW_REVIEW, WORKFLOW_QUESTIONS, WORKFLOW_INSPECTION,
    get_workflow_state, transition_workflow, enter_workbench,
)
from circuitcare.dashboard.views.symptom_diagnosis_panel import candidate_focus_widget_key
from circuitcare.dashboard.views.symptom_diagnosis_panel import (
    _queue_measurement_change_rerun, _consume_measurement_change_rerun,
)
from circuitcare.analysis.models import AnalysisStatus
from circuitcare.frontend.board_component_registry import build_board_component_registry, source_component_key
from test_board_component_registry import _snapshots


class _Context:
    def button(self, *_args, **_kwargs):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def markdown(self, *_args, **_kwargs):
        pass

    def caption(self, *_args, **_kwargs):
        pass


class _StreamlitStub:
    def __init__(self, *, state=None, clicked=()):
        self.session_state = state or {}
        self.clicked = set(clicked)
        self.radio_calls = []

    def container(self, **_kwargs):
        return _Context()

    def columns(self, specs, **_kwargs):
        count = specs if isinstance(specs, int) else len(specs)
        return [_Context() for _ in range(count)]

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


def test_workflow_state_is_board_scoped_and_only_accepts_explicit_valid_transitions():
    first = ProjectContext("wf-project", "front-board")
    second = ProjectContext("wf-project", "other-board")
    state = {}
    workflow = get_workflow_state(state, first)
    assert workflow["stage"] == WORKFLOW_REVIEW
    with pytest.raises(ValueError, match="허용되지 않은"):
        transition_workflow(state, first, WORKFLOW_INSPECTION)
    transition_workflow(state, first, WORKFLOW_QUESTIONS, event="review_complete")
    assert get_workflow_state(state, first)["stage"] == WORKFLOW_QUESTIONS
    assert get_workflow_state(state, second)["stage"] == WORKFLOW_REVIEW
    assert workflow["stage_history"][-1]["event"] == "review_complete"


def test_inspection_targets_use_real_registry_ids_order_evidence_and_never_invent_boxes():
    project = ProjectContext("workflow-targets", "workflow-target-board")
    registry = build_board_component_registry(
        {}, project_id=project.project_id, board_id=project.board_id,
        photo_snapshots=_snapshots(),
    )
    front_id = registry["source_to_board_id"][source_component_key("front", "local-1")]
    close_id = registry["source_to_board_id"][source_component_key("close", "close-1")]
    suspects = [{"board_component_id": close_id}]
    selected = workspace._inspection_targets(
        registry,
        [{"board_component_id": front_id, "component_type": "resistor",
          "fault_hypotheses": [{"fault_id": "RESISTOR_OPEN"}]}],
        suspects,
    )
    assert [row["board_component_id"] for row in selected[:2]] == [
        front_id, close_id,
    ]
    assert len(selected) <= 3
    assert all(row["board_component_id"] in {item["board_component_id"] for item in registry["components"]}
               for row in selected)

    excluded_only = build_board_component_registry(
        {}, project_id=project.project_id, board_id=project.board_id,
        photo_snapshots={"front": {"components": (
            SimpleNamespace(component_id="X1", component_type="resistor", bbox_xyxy=(1, 2, 8, 9),
                            coordinate_frame="SOURCE_PHOTO_PIXEL", identification_status="EXCLUDED",
                            presence_status="EXCLUDED", component_type_status="UNCONFIRMED",
                            ocr_marking=None, possible_part_number=None, confirmed_functional_roles=(),
                            evidence=(), overlay_source="test"),),
            "image_fingerprint": "excluded", "review_version": "excluded"}},
    )
    assert workspace._inspection_targets(excluded_only, [], []) == []


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
        "presence_status": "UNCONFIRMED",
        "source_components": [{"photo_id": "back-photo", "component_id": "R8",
                               "coordinate_frame": "SOURCE_PHOTO_PIXEL", "bbox_xyxy": [1, 2, 8, 9],
                               "component_type": "resistor", "review_status": "검수 필요"}],
    }]}
    st = _StreamlitStub(state=state)
    target_key = "measurement-target"
    state[target_key] = "bc-2"

    workspace._handle_measurement_target_change(state, project, registry, target_key)

    assert state[workspace.MEASUREMENT_TARGET_BOARD_COMPONENT_KEY] == "bc-2"
    assert workspace.BOARD_COMPONENT_SELECTION_KEY not in state
    assert "circuitcare_v2_photo_choice" not in state
    assert state[SELECTION_KEY].selected_component is None


def test_fixed_viewer_shell_starts_at_component_review_with_one_current_stage(monkeypatch):
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

    state = {}
    enter_workbench(state, project)
    state[f"cc_workbench_panel_{project.project_id}"] = "원인"  # ignored legacy navigation state
    workspace.render_workbench_shell(_StreamlitStub(state=state), project)
    assert calls == ["viewer", "review"]


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


def test_dismissed_board_suspect_is_not_restored_by_legacy_scope_migration():
    project = ProjectContext("suspect-p", "suspect-b")
    owner = f"{project.project_id}|{project.board_id}"
    old_owner = f"{owner}|photo-1"
    state = {
        inspection.USER_SUSPECTS_KEY: {
            old_owner: {"BC-R1": {"board_component_id": "BC-R1", "component_id": "R1"}},
        },
        inspection.USER_SUSPECTS_MIGRATION_KEY: {},
    }

    inspection.dismiss_board_suspect(state, project, "BC-R1")
    assert inspection.board_suspects(state, project) == []
    assert "BC-R1" in state[inspection.USER_SUSPECTS_DISMISSED_KEY][owner]


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
    assert state[SELECTION_KEY].selected_component == "C2"
    assert state[workspace.BOARD_COMPONENT_SELECTION_KEY] == "BC-C2"
    assert state[workspace.MEASUREMENT_TARGET_BOARD_COMPONENT_KEY] == "BC-R1"


def test_excluded_registry_component_cannot_be_focused_as_measurement_or_viewer_target():
    project = ProjectContext("excluded-focus-p", "excluded-focus-b")
    registry = {"components": [{
        "board_component_id": "bc-x", "presence_status": "EXCLUDED",
        "source_components": [{"photo_id": "p1", "component_id": "R1",
                               "coordinate_frame": "SOURCE_PHOTO_PIXEL", "bbox_xyxy": [1, 2, 8, 9]}],
    }]}
    state = {}
    enter_workbench(state, project)
    st = _StreamlitStub(state=state)
    assert workspace._focus_board_registry_component(st, project, registry, "bc-x") is False
    state["target"] = "bc-x"
    workspace._handle_measurement_target_change(state, project, registry, "target")
    assert workspace.MEASUREMENT_TARGET_BOARD_COMPONENT_KEY not in state
    assert workspace.BOARD_COMPONENT_SELECTION_KEY not in state


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
    assert len(app.radio) == 0  # first workflow stage is component review
    assert not any(item.label == "작업 패널" for item in app.radio)
    assert not any(item.label == "후보 표시 전략" for item in app.selectbox)
    assert "진단 미실행" in " ".join(item.value for item in app.caption)


def test_apptest_review_next_button_transitions_only_right_panel_to_questions(monkeypatch):
    app_testing = pytest.importorskip("streamlit.testing.v1")
    from circuitcare.frontend.state import INTAKE_IMAGES_KEY, INTAKE_PROJECT_KEY
    from circuitcare.frontend.workbench_state import SCREEN_KEY, WORKFLOW_STATES_KEY

    project = ProjectContext("workflow-apptest", "workflow-board")
    component = SimpleNamespace(component_id="R1", board_component_id="BC-R1",
                                component_type="resistor", presence_status="UNCONFIRMED")
    registry = {"project_id": project.project_id, "board_id": project.board_id,
                "version": "registry-test", "components": [{
                    "board_component_id": "BC-R1", "component_type": "resistor",
                    "presence_status": "UNCONFIRMED", "source_components": [
                        {"photo_id": "photo-1", "component_id": "R1", "review_status": "UNCONFIRMED"},
                    ],
                }]}
    monkeypatch.setattr(workspace, "_prepare_board_registry_snapshot",
                        lambda *_args, **_kwargs: (registry, {"components": (component,)}))
    monkeypatch.setattr(workspace, "diagnosis_components", lambda _registry: (component,))
    monkeypatch.setattr(workspace, "_render_pcb", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workspace, "_render_component_inspector", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workspace, "_diagnosis_status_label", lambda *_args, **_kwargs: "진단 전")
    monkeypatch.setattr(workspace, "current_analysis_snapshot", lambda *_args, **_kwargs: {
        "result": None, "record": {"status": "NOT_RUN"}, "input_fingerprint": "mock-fp",
    })
    monkeypatch.setattr(workspace, "render_symptom_diagnosis_panel",
                        lambda st, *_args, **_kwargs: st.markdown("질문 단계 AppTest 확인"))
    app = app_testing.AppTest.from_file(str(Path(__file__).parents[1] / "src/circuitcare/ui.py"),
                                        default_timeout=30)
    app.session_state[INTAKE_PROJECT_KEY] = project
    app.session_state[INTAKE_IMAGES_KEY] = []
    app.session_state[SCREEN_KEY] = "workbench"
    app.run()
    assert not app.exception
    assert any(button.label == "검수 완료 · 고장 질문으로 진행" for button in app.button)

    next(button for button in app.button if button.label == "검수 완료 · 고장 질문으로 진행").click().run()

    assert not app.exception
    assert any("질문 단계 AppTest 확인" in item.value for item in app.markdown)
    workflow_store = app.session_state[WORKFLOW_STATES_KEY]
    assert workflow_store[f"{project.project_id}|{project.board_id}"]["stage"] == WORKFLOW_QUESTIONS


def test_apptest_questions_next_uses_current_result_and_real_registry_targets(monkeypatch):
    app_testing = pytest.importorskip("streamlit.testing.v1")
    from circuitcare.frontend.state import INTAKE_IMAGES_KEY, INTAKE_PROJECT_KEY
    from circuitcare.frontend.workbench_state import (
        SCREEN_KEY, WORKFLOW_STATES_KEY, transition_workflow,
    )

    project = ProjectContext("workflow-inspection-apptest", "workflow-inspection-board")
    registry = build_board_component_registry(
        {}, project_id=project.project_id, board_id=project.board_id,
        photo_snapshots=_snapshots(),
    )
    components = workspace.diagnosis_components(registry)
    snapshot = {"components": components}
    calls = {"analysis": 0}
    monkeypatch.setattr(workspace, "_prepare_board_registry_snapshot",
                        lambda *_args, **_kwargs: (registry, snapshot))
    monkeypatch.setattr(workspace, "diagnosis_components", lambda _registry: components)
    monkeypatch.setattr(workspace, "_render_pcb", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workspace, "_render_component_inspector", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workspace, "_diagnosis_status_label", lambda *_args, **_kwargs: "현재 분석")

    def current_snapshot(*_args, **_kwargs):
        calls["analysis"] += 1
        return {
            "result": {"candidate_fusion": {"candidates": [{
                "board_component_id": registry["components"][0]["board_component_id"],
                "component_type": registry["components"][0]["component_type"],
                "fault_hypotheses": [{"fault_id": "RESISTOR_OPEN"}],
            }]}},
            "record": {"status": AnalysisStatus.CURRENT.value},
            "input_fingerprint": "current-workflow-fingerprint",
        }

    monkeypatch.setattr(workspace, "current_analysis_snapshot", current_snapshot)
    monkeypatch.setattr(workspace, "render_symptom_diagnosis_panel",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workspace, "render_measurement_panel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workspace, "load_knowledge", lambda: {"component_test_profiles": {"profiles": {}}})
    measurement = SimpleNamespace(measurement_id="measurement-1",
                                  board_component_id=registry["components"][0]["board_component_id"],
                                  fault_hypothesis=None)
    monkeypatch.setattr(workspace, "get_measurements", lambda *_args, **_kwargs: [measurement])

    app = app_testing.AppTest.from_file(str(Path(__file__).parents[1] / "src/circuitcare/ui.py"),
                                        default_timeout=30)
    app.session_state[INTAKE_PROJECT_KEY] = project
    app.session_state[INTAKE_IMAGES_KEY] = []
    app.session_state[SCREEN_KEY] = "workbench"
    workflow_store: dict = {}
    transition_workflow(workflow_store, project, WORKFLOW_QUESTIONS, event="review_complete")
    app.session_state[WORKFLOW_STATES_KEY] = workflow_store[WORKFLOW_STATES_KEY]
    app.run()
    assert not app.exception
    assert any(button.label == "검사 후보 선정" for button in app.button)

    next(button for button in app.button if button.label == "검사 후보 선정").click().run()

    assert not app.exception
    workflow_state = app.session_state[WORKFLOW_STATES_KEY][f"{project.project_id}|{project.board_id}"]
    assert workflow_state["stage"] == WORKFLOW_INSPECTION
    valid_ids = {row["board_component_id"] for row in registry["components"]
                 if row.get("presence_status") != "EXCLUDED"}
    assert workflow_state["inspection_targets"]
    assert {row["board_component_id"] for row in workflow_state["inspection_targets"]} <= valid_ids
    assert any(item.label == "검사할 소자" for item in app.selectbox)
    assert calls["analysis"] == 6  # three current-state reads per render; no detector is invoked by the transition

    next(button for button in app.button if button.label == "측정 결과 분석").click().run()

    assert not app.exception
    workflow_state = app.session_state[WORKFLOW_STATES_KEY][f"{project.project_id}|{project.board_id}"]
    assert workflow_state["stage"] == "REASSESSMENT"
    assert workflow_state["reassessment_history"][-1]["measurement_ids"] == ["measurement-1"]
