"""Explain what is known, inferred, contradicted, and still needed per hypothesis."""

from __future__ import annotations

from typing import Any


def analyze_evidence_gaps(
    *, hypotheses: list[dict[str, Any]], knowledge: dict[str, Any],
    confirmed_types: set[str] | list[str], functional_blocks: list[dict[str, Any]],
    component_candidates: list[dict[str, Any]], measurement_evidence: list[dict[str, Any]],
    evidence_ledger: list[dict[str, Any]], connectivity_status: str = "none",
) -> list[dict[str, Any]]:
    """Create transparent next-evidence requirements; never infer missing values."""
    modes = knowledge.get("fault_modes", {}).get("fault_modes", {})
    tests = knowledge.get("test_catalog", {}).get("tests", {})
    confirmed = {str(value) for value in confirmed_types}
    output = []
    for hypothesis in hypotheses:
        fault_id = str(hypothesis.get("fault_id") or "")
        mode = modes.get(fault_id, {})
        related_ids = [str(value) for value in hypothesis.get("related_component_ids", []) if value]
        related_candidates = [row for row in component_candidates
                              if str(row.get("fault_hypothesis") or row.get("fault_id") or "") == fault_id]
        relevant_ledger = [item for item in evidence_ledger
                           if not item.get("component_id") or item.get("component_id") in related_ids]
        missing = []
        required_types = {str(value) for value in mode.get("required_component_types", [])}
        missing_types = sorted(required_types - confirmed)
        if missing_types:
            missing.append({"kind": "component_identity", "status": "UNKNOWN",
                            "description": "확인된 소자 종류가 부족합니다: " + ", ".join(missing_types),
                            "verification_methods": ["PCB 소자 검수"]})
        required_roles = {str(value) for value in mode.get("functional_roles", [])}
        available_roles = {str(role) for row in functional_blocks for role in row.get("candidate_blocks", [])}
        if required_roles and not (required_roles & available_roles):
            missing.append({"kind": "functional_role", "status": "INCONCLUSIVE",
                            "description": "필요 기능 역할을 확인하지 못했습니다: " + ", ".join(sorted(required_roles)),
                            "verification_methods": ["각인·실크스크린·부품 정보 확인"]})
        required_tests = {str(value) for value in mode.get("required_measurement_tests", [])}
        completed_tests = {str(row.get("test_id")) for row in measurement_evidence
                           if row.get("test_id") and row.get("used_for_ranking")}
        for test_id in sorted(required_tests - completed_tests):
            test = tests.get(test_id, {})
            missing.append({"kind": "measurement", "status": "UNKNOWN",
                            "description": f"필수 검증 측정 미확인: {test.get('name', test_id)}",
                            "test_id": test_id, "verification_methods": [test.get("question") or test.get("name") or test_id],
                            "risk_level": test.get("risk_level", "unknown"),
                            "requires_power_off": bool(test.get("requires_power_off", False))})
        if mode.get("level") in {"component", "functional_block"} and not related_ids:
            missing.append({"kind": "component_scope", "status": "UNKNOWN",
                            "description": "가설에 연결할 확인된 소자 후보가 없습니다.",
                            "verification_methods": ["소자 종류와 기능 역할 검수"]})
        if required_roles and connectivity_status not in {"verified", "partial_verified"}:
            missing.append({"kind": "electrical_relation", "status": "UNKNOWN",
                            "description": "소자 근접성·기능 역할만으로 전기적 연결을 확정할 수 없습니다.",
                            "verification_methods": ["회로도·검증된 연결 정보 또는 안전한 측정 기록 확인"]})
        known = [item for item in relevant_ledger if item.get("status") in {"VERIFIED", "USER_REPORTED"}]
        inferred = [item for item in relevant_ledger if item.get("status") in {"INFERRED", "AI_SUGGESTED"}]
        opposed = list(hypothesis.get("opposing_evidence", []))
        recommended = []
        for test_id in mode.get("recommended_tests", []):
            test = tests.get(str(test_id), {})
            if test and str(test_id) not in completed_tests:
                recommended.append({"test_id": str(test_id), "name": test.get("name", str(test_id)),
                                    "risk_level": test.get("risk_level", "unknown"),
                                    "requires_power_off": bool(test.get("requires_power_off", False)),
                                    "question": test.get("question")})
        output.append({"fault_id": fault_id, "level": hypothesis.get("level"),
                       "confirmed_facts": known, "inferred_facts": inferred,
                       "opposing_evidence": opposed, "missing_evidence": missing,
                       "verification_methods": list(dict.fromkeys(method for item in missing
                                                                    for method in item.get("verification_methods", []))),
                       "recommended_checks": recommended,
                       "candidate_component_ids": related_ids,
                       "candidate_group_ids": sorted({str(row.get("candidate_group_id"))
                                                       for row in related_candidates if row.get("candidate_group_id")}),
                       "status": "INCONCLUSIVE" if missing else "UNVERIFIED_HYPOTHESIS"})
    return output
