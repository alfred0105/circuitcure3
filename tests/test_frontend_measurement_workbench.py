from dataclasses import replace
import math

import pytest

from circuitcare.frontend.measurement_models import create_measurement_record
from circuitcare.frontend.measurement_state import (
    diagnosis_state, get_measurements, save_measurement, update_measurement,
    delete_measurement, current_measurement_set_sha256,
)
from circuitcare.dashboard.views.measurement_panel import (
    _measurement_verification_gaps, _sigma_for_input_unit,
)
from circuitcare.analysis.fingerprints import build_analysis_fingerprints
from circuitcare.analysis.models import AnalysisScope, AnalysisStatus
from circuitcare.analysis.state_manager import (
    begin_analysis, commit_analysis_result, get_current_result, get_current_state,
)


def _measurement(**overrides):
    values = {
        "project_id": "p1", "board_id": "b1", "component_id": "c1",
        "component_identity_status": "SUGGESTED_UNCONFIRMED",
        "component_map_version": "map-v1", "measurement_type": "voltage",
        "value": "0", "unit": "V", "uncertainty": "0.1",
        "source": "user_measurement", "measurement_id": "m1", "timestamp": "t1",
    }
    values.update(overrides)
    return create_measurement_record(**values)


@pytest.mark.parametrize(("value", "unit", "expected", "base_unit"), [
    ("0", "V", 0.0, "V"), ("-0.25", "V", -0.25, "V"),
    ("1250", "mV", 1.25, "V"), ("3.5", "kΩ", 3500.0, "Ω"),
    ("250", "mA", 0.25, "A"),
])
def test_measurement_value_accepts_zero_negative_and_converts_units(value, unit, expected, base_unit):
    measurement_type = {"V": "voltage", "mV": "voltage", "kΩ": "resistance", "mA": "current"}[unit]
    row = _measurement(value=value, unit=unit, measurement_type=measurement_type)
    assert row.value == expected
    assert row.unit == base_unit
    assert row.entered_value == float(value)
    assert row.entered_unit == unit
    assert row.node_id is None and row.probe_id is None
    assert row.mapping_status == "NODE_MAPPING_UNAVAILABLE"


@pytest.mark.parametrize("value", ["", "NaN", "Infinity", "-Infinity", "abc"])
def test_invalid_numeric_values_are_rejected(value):
    with pytest.raises(ValueError):
        _measurement(value=value)


@pytest.mark.parametrize("uncertainty", ["0", "-0.1", "NaN", "Infinity", "abc"])
def test_invalid_uncertainty_is_rejected(uncertainty):
    with pytest.raises(ValueError):
        _measurement(uncertainty=uncertainty)


def test_uncertainty_may_be_absent_for_record_only_measurement():
    row = _measurement(uncertainty="")
    assert row.uncertainty is None
    assert row.verification_status == "USER_REPORTED"
    assert "측정 불확실도 미입력" in _measurement_verification_gaps(row)
    assert "전기적 측정 지점 연결 미확인" in _measurement_verification_gaps(row)


def test_measurement_record_can_be_saved_without_verification_data():
    row = _measurement(uncertainty="", measurement_target="")
    assert row.measurement_id == "m1"
    assert len(_measurement_verification_gaps(row)) >= 3


def test_measurement_history_is_project_board_and_component_scoped_and_mutations_invalidate():
    state = {}
    row = _measurement()
    save_measurement(state, row)
    first_set = current_measurement_set_sha256(state, project_id="p1", board_id="b1")
    assert len(get_measurements(state, project_id="p1", board_id="b1", component_id="c1")) == 1
    assert get_measurements(state, project_id="p2", board_id="b1") == []
    assert get_measurements(state, project_id="p1", board_id="b2") == []
    assert diagnosis_state(state, project_id="p1", board_id="b1")["status"] == "STALE_RESULT"

    update_measurement(state, replace(row, value=2.5, entered_value=2.5, changed_at="t2"))
    updated_set = current_measurement_set_sha256(state, project_id="p1", board_id="b1")
    assert updated_set != first_set
    assert get_measurements(state, project_id="p1", board_id="b1")[0].value == 2.5
    assert delete_measurement(state, project_id="p1", board_id="b1", measurement_id="m1")
    assert get_measurements(state, project_id="p1", board_id="b1") == []


def test_measurement_cannot_be_relinked_to_another_component_on_edit():
    state = {}
    row = _measurement()
    save_measurement(state, row)
    with pytest.raises(ValueError, match="부품 연결"):
        update_measurement(state, replace(row, component_id="c2"))


def test_measurement_cannot_be_relinked_to_a_new_map_or_identity_on_edit():
    state = {}
    row = _measurement()
    save_measurement(state, row)
    with pytest.raises(ValueError, match="Component Map"):
        update_measurement(state, replace(row, component_map_version="map-v2"))
    with pytest.raises(ValueError, match="Component Map"):
        update_measurement(state, replace(row, component_identity_status="USER_CONFIRMED"))


def test_same_measurement_facts_keep_current_analysis_but_changed_facts_stale_it():
    state = {}
    row = _measurement()
    save_measurement(state, row)
    scope = AnalysisScope("p1", "b1", "photo-1")
    fingerprint = build_analysis_fingerprints(
        input_payload={"symptoms": ["no_power"]}, measurements=[row],
    )
    run_id = begin_analysis(state, scope=scope, fingerprints=fingerprint)
    assert commit_analysis_result(state, scope=scope, run_id=run_id,
                                  fingerprints=fingerprint, result={"status": "OK"})

    update_measurement(state, replace(row, changed_at="later", measurement_set_id="different"))
    assert get_current_result(state, scope=scope, fingerprints=fingerprint) == {"status": "OK"}
    assert get_current_state(state, scope)["status"] == AnalysisStatus.CURRENT

    update_measurement(state, replace(row, value=2.5, entered_value=2.5, changed_at="changed"))
    assert get_current_result(state, scope=scope, fingerprints=fingerprint) is None
    assert get_current_state(state, scope)["status"] == AnalysisStatus.STALE
    assert get_current_state(state, scope)["history"][-1]["results"]["symptom"] == {"status": "OK"}


def test_ui_uncertainty_stays_in_base_unit_when_measurement_unit_changes():
    assert float(_sigma_for_input_unit("0.1", "voltage", "mV")) == pytest.approx(100.0)
    assert float(_sigma_for_input_unit("5", "resistance", "kΩ")) == pytest.approx(0.005)
