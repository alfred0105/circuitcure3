from circuitcare.analysis.fingerprints import build_analysis_fingerprints
from circuitcare.analysis.models import AnalysisScope, AnalysisStatus
from circuitcare.analysis.state_manager import (
    begin_analysis, commit_analysis_result, fail_analysis, get_current_result,
    get_current_state, invalidate_analysis,
)


def _fingerprints(*, symptoms="no_power", measurements=(), rules="rules-v1"):
    return build_analysis_fingerprints(
        input_payload={"symptoms": symptoms, "component": "U1", "review": "CONFIRMED"},
        measurements=measurements,
        extra_rules_payload={"engine_profile": rules},
    )


def test_changed_inputs_stale_result_and_preserve_history():
    state = {}
    scope = AnalysisScope("p1", "b1", "photo-1")
    first = _fingerprints()
    run_id = begin_analysis(state, scope=scope, fingerprints=first)
    assert get_current_state(state, scope)["status"] == AnalysisStatus.RUNNING
    assert commit_analysis_result(state, scope=scope, run_id=run_id, fingerprints=first,
                                  current_fingerprints=first, result={"status": "OK"})
    assert get_current_result(state, scope=scope, fingerprints=first) == {"status": "OK"}

    changed = _fingerprints(symptoms="no_led")
    assert get_current_result(state, scope=scope, fingerprints=changed) is None
    record = get_current_state(state, scope)
    assert record["status"] == AnalysisStatus.STALE
    assert record["history"][-1]["results"]["symptom"] == {"status": "OK"}


def test_analysis_changed_while_running_cannot_commit_old_result():
    state = {}
    scope = AnalysisScope("p1", "b1", "photo-1")
    started = _fingerprints()
    run_id = begin_analysis(state, scope=scope, fingerprints=started)
    invalidate_analysis(state, scope=scope, reason="component changed")
    assert not commit_analysis_result(state, scope=scope, run_id=run_id,
                                      fingerprints=started, result={"status": "OLD"})
    assert get_current_result(state, scope=scope, fingerprints=started) is None
    assert get_current_state(state, scope)["status"] == AnalysisStatus.STALE


def test_commit_rejects_changed_live_fingerprint_even_if_run_token_is_valid():
    state = {}
    scope = AnalysisScope("p1", "b1", "photo-1")
    started = _fingerprints()
    changed = _fingerprints(symptoms="no_led")
    run_id = begin_analysis(state, scope=scope, fingerprints=started)
    assert not commit_analysis_result(state, scope=scope, run_id=run_id,
                                      fingerprints=started, current_fingerprints=changed,
                                      result={"status": "OLD"})
    assert get_current_result(state, scope=scope, fingerprints=started) is None


def test_identical_semantic_inputs_and_display_only_scopes_do_not_invalidate():
    state = {}
    scope = AnalysisScope("p1", "b1", "photo-1")
    fingerprints = _fingerprints(measurements=[{"value": 3.3, "unit": "V"}])
    run_id = begin_analysis(state, scope=scope, fingerprints=fingerprints)
    assert commit_analysis_result(state, scope=scope, run_id=run_id,
                                  fingerprints=fingerprints, result={"status": "OK"})
    # Zoom, tab and selected box are intentionally absent from the analysis payload.
    same = _fingerprints(measurements=[{"unit": "V", "value": 3.3}])
    assert get_current_result(state, scope=scope, fingerprints=same) == {"status": "OK"}


def test_project_board_and_photo_scopes_are_isolated():
    state = {}
    first = AnalysisScope("p1", "b1", "photo-1")
    second_photo = AnalysisScope("p1", "b1", "photo-2")
    other_board = AnalysisScope("p1", "b2", "photo-1")
    other_project = AnalysisScope("p2", "b1", "photo-1")
    fingerprints = _fingerprints()
    run_id = begin_analysis(state, scope=first, fingerprints=fingerprints)
    assert commit_analysis_result(state, scope=first, run_id=run_id,
                                  fingerprints=fingerprints, result={"status": "OK"})
    invalidate_analysis(state, scope=second_photo, reason="other photo changed")
    assert get_current_result(state, scope=first, fingerprints=fingerprints) == {"status": "OK"}
    for other in (second_photo, other_board, other_project):
        assert get_current_result(state, scope=other, fingerprints=fingerprints) is None


def test_rule_fingerprint_and_failure_status_are_tracked():
    state = {}
    scope = AnalysisScope("p1", "b1", "photo-1")
    first = _fingerprints()
    run_id = begin_analysis(state, scope=scope, fingerprints=first)
    assert fail_analysis(state, scope=scope, run_id=run_id, error="engine failure")
    assert get_current_state(state, scope)["status"] == AnalysisStatus.FAILED
    assert get_current_result(state, scope=scope, fingerprints=_fingerprints(rules="rules-v2")) is None
