"""Condition-scoped measurement checks tied to validated component test profiles."""

from __future__ import annotations

from math import isfinite
from typing import Any


AssessmentKey = tuple[str, str, str, str, str]
VERIFIED_REFERENCE_STATES = {"DATASHEET_VERIFIED", "BOARD_SPEC_VERIFIED"}


def _fold(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def evaluate_component_measurements(
    measurements: list[dict[str, Any]] | None, *, profiles: dict[str, Any] | None = None,
    current_context: dict[str, Any] | None = None,
) -> dict[AssessmentKey, dict[str, Any]]:
    """Evaluate each distinct test/target/condition; never trust a free-text range by itself."""
    profile_map = (profiles or {}).get("profiles", {})
    context = current_context or {}
    evaluations: dict[AssessmentKey, dict[str, Any]] = {}
    for row in measurements or []:
        if not isinstance(row, dict):
            continue
        component_id = str(row.get("component_id") or "")
        fault_id = str(row.get("fault_hypothesis") or "")
        test_id = str(row.get("test_id") or "")
        target = str(row.get("measurement_target") or "").strip()
        condition = str(row.get("operating_conditions") or "").strip()
        normalized_condition = _fold(condition)
        key: AssessmentKey = (component_id, fault_id, test_id, _fold(target), normalized_condition)
        measurement_id = str(row.get("measurement_id") or "")
        reason = None
        verdict = "non_diagnostic"
        status = "INCONCLUSIVE"
        if not component_id or not fault_id or not test_id or not measurement_id:
            reason = "부품·가설·검사 항목·측정 ID 중 누락된 항목이 있습니다."
        elif row.get("source") != "user_measurement" or row.get("verification_status") != "USER_REPORTED":
            reason = "실제 사용자 측정값이 아닙니다."
        elif row.get("component_identity_status") != "USER_CONFIRMED":
            reason = "현재 보드에서 부품 존재와 식별이 사용자 확인되지 않았습니다."
        elif row.get("location_verified") is not True:
            reason = "측정 위치와 해당 부품 단자를 확인하지 않았습니다."
        elif row.get("uncertainty") is None or not str(row.get("uncertainty_source") or "").strip():
            reason = "측정 불확실도와 그 출처가 기록되지 않았습니다."
        elif context and (
            row.get("project_id") != context.get("project_id")
            or row.get("board_id") != context.get("board_id")
            or row.get("component_map_version") != context.get("component_map_version")
        ):
            reason = "프로젝트·보드·Component Map 버전이 현재 분석과 다릅니다."
        elif ("component_identity_statuses" in context
              and context.get("component_identity_statuses", {}).get(component_id)
              != row.get("component_identity_status")):
            reason = "현재 Component Map의 부품 식별 상태와 측정 기록이 다릅니다."
        elif row.get("reference_verification_status") not in VERIFIED_REFERENCE_STATES:
            reason = "정상 기준 출처가 검증되지 않았습니다. 입력한 범위는 참고값으로만 둡니다."
        elif not str(row.get("reference_source_id") or "").strip() or not str(row.get("reference_source") or "").strip():
            reason = "검증된 데이터시트/보드 사양의 출처 ID와 설명이 필요합니다."
        elif not target:
            reason = "측정 대상/단자가 지정되지 않았습니다."
        elif not normalized_condition or normalized_condition != _fold(row.get("reference_conditions")):
            reason = "실제 측정 조건과 정상 기준 조건이 일치하지 않습니다."
        else:
            component_type = str(row.get("component_type") or "").casefold()
            test_profile = profile_map.get(component_type, {}).get("tests", {}).get(test_id, {})
            if not test_profile:
                reason = "이 부품 종류의 검사 프로파일에 등록되지 않은 검사입니다."
            elif test_profile.get("measurement_type") != row.get("measurement_type"):
                reason = "검사 프로파일의 측정 종류와 기록의 측정 종류가 다릅니다."
            elif not any(_fold(term) in _fold(target) for term in test_profile.get("target_keywords", [])):
                reason = "측정 대상이 검사 프로파일에 지정된 단자/위치와 맞지 않습니다."
            elif not any(_fold(term) in normalized_condition for term in test_profile.get("condition_keywords", [])):
                reason = "운전 조건이 검사 프로파일에서 요구하는 조건을 포함하지 않습니다."
            elif fault_id not in test_profile.get("fault_relations", {}):
                reason = "이 검사 항목은 선택한 고장 가설을 판별하지 않습니다."
            else:
                low, high = row.get("reference_min"), row.get("reference_max")
                try:
                    value = float(row["value"])
                    low_value, high_value = float(low), float(high)
                    sigma = float(row["uncertainty"])
                    if not all(isfinite(number) for number in (value, low_value, high_value, sigma)):
                        raise ValueError
                    if low_value > high_value or sigma <= 0:
                        raise ValueError
                except (KeyError, TypeError, ValueError, OverflowError):
                    reason = "측정값·범위·불확실도가 유효하지 않습니다."
                else:
                    measured_low, measured_high = value - sigma, value + sigma
                    if measured_low >= low_value and measured_high <= high_value:
                        status = "TEST_NORMAL"
                        reason = "해당 검사와 조건에서 측정 범위가 검증 기준 안에 있습니다."
                    elif measured_high < low_value or measured_low > high_value:
                        status = "TEST_ABNORMAL"
                        reason = "해당 검사와 조건에서 측정 범위가 검증 기준을 벗어났습니다."
                    else:
                        status = "INCONCLUSIVE"
                        reason = "측정 불확실도 범위가 검증 기준의 경계와 겹칩니다."
                    verdict = test_profile["fault_relations"][fault_id].get(status, "non_diagnostic")

        item = {
            "component_id": component_id, "fault_id": fault_id, "test_id": test_id,
            "measurement_target": target, "operating_conditions": condition,
            "measurement_id": measurement_id, "status": status, "verdict": verdict,
            "reason": reason, "reference_source_id": row.get("reference_source_id"),
            "reference_verification_status": row.get("reference_verification_status"),
            "evidence_ids": [measurement_id] if measurement_id else [],
        }
        previous = evaluations.get(key)
        if previous is None:
            evaluations[key] = item
        else:
            evidence_ids = list(dict.fromkeys([*previous.get("evidence_ids", []), *item["evidence_ids"]]))
            old_valid = previous.get("verdict") in {"supports", "refutes"}
            new_valid = verdict in {"supports", "refutes"}
            if old_valid and new_valid and previous.get("verdict") != verdict:
                previous.update(status="CONFLICTED", verdict="conflicted",
                                reason="동일한 검사·단자·조건의 결과가 서로 충돌합니다.")
            elif new_valid and not old_valid:
                previous.update(item)
            previous["evidence_ids"] = evidence_ids
    return evaluations
