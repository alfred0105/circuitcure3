"""Evidence-bounded local LLM assistance for symptom diagnosis."""

from __future__ import annotations

import json
from typing import Any

from circuitcare.local_llm import LocalLLMError, OllamaClient


def suggest_pcb_fault_candidates(
    client: OllamaClient, model: str, *, pcb_context: dict[str, Any],
    diagnosis: dict[str, Any], knowledge: dict[str, Any],
) -> dict[str, Any]:
    """Return schema-bounded Qwen proposals, separate from deterministic ranks."""
    components = {str(row.get("component_id")): row for row in pcb_context.get("focus_components", [])
                  if isinstance(row, dict) and row.get("component_id")}
    confirmed_ids = sorted(cid for cid, row in components.items() if row.get("review_status") == "CONFIRMED")
    fault_modes = knowledge.get("fault_modes", {}).get("fault_modes", {})
    modes = sorted(fault_modes)
    if not modes or not confirmed_ids:
        return {"proposals": [], "model": model, "status": "NO_ELIGIBLE_COMPONENTS"}
    block_values = sorted({"power", "input", "control", "control_input", "control_timing",
                           "driver", "output", "protection", "thermal", "buzzer", "relay"})
    evidence_values = {str(key) for key in pcb_context.get("observations", {})}
    evidence_values.update(str(row.get("id")) for row in diagnosis.get("matched_rules", []) if row.get("id"))
    evidence_values.update(f"measurement:{row.get('measurement_id')}" for row in diagnosis.get("measurement_evidence", [])
                           if row.get("measurement_id"))
    evidence_options = sorted(evidence_values) or ["__none__"]
    schema = {
        "type": "object", "additionalProperties": False, "required": ["proposals"],
        "properties": {"proposals": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["component_id", "suspected_fault_id", "functional_block", "reason",
                         "supporting_evidence_ids", "opposing_evidence_ids", "missing_information",
                         "recommended_next_action", "uncertainty_status"],
            "properties": {
                "component_id": {"type": "string", "enum": confirmed_ids},
                "suspected_fault_id": {"type": "string", "enum": modes},
                "functional_block": {"type": "string", "enum": block_values + ["UNKNOWN"]},
                "reason": {"type": "string", "maxLength": 500},
                "supporting_evidence_ids": {"type": "array", "maxItems": 6,
                                             "items": {"type": "string", "enum": evidence_options}},
                "opposing_evidence_ids": {"type": "array", "maxItems": 6,
                                           "items": {"type": "string", "enum": evidence_options}},
                "missing_information": {"type": "array", "maxItems": 5,
                                        "items": {"type": "string", "maxLength": 160}},
                "recommended_next_action": {"type": "string", "maxLength": 240},
                "uncertainty_status": {"type": "string", "enum": ["UNVERIFIED", "INCONCLUSIVE"]},
            },
        }}}
    }
    payload = {"pcb_context": pcb_context, "rule_hypotheses": pcb_context.get("hypotheses", []),
               "allowed_component_ids": confirmed_ids, "allowed_fault_ids": modes,
               "evidence_ids": evidence_options}
    response = client.chat_json(model, [
        {"role": "system", "content": (
            "PCB 진단 보조자다. 입력 JSON의 실제 확인된 소자, 허용 고장 유형, 증거만 사용한다. "
            "사용자 의심은 미검증 단서다. 존재하지 않는 측정·배선·핀·연결·소자·증거를 만들지 않는다. "
            "근거가 부족하면 후보를 생략하거나 INCONCLUSIVE로 제안한다. Qwen 제안만으로 고장을 확정하거나 "
            "FAULT_SUPPORTED 판정을 내리지 않는다. 기능부는 추정 후보일 뿐 연결 사실이 아니다. "
            "JSON 스키마에 맞는 제안 최대 3개를 반환한다."
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
    ], output_schema=schema)
    rows = response.get("content", {}).get("proposals", [])
    validated = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        cid, fault_id = str(row.get("component_id") or ""), str(row.get("suspected_fault_id") or "")
        mode = fault_modes.get(fault_id)
        component = components.get(cid)
        if not component or component.get("review_status") != "CONFIRMED" or not isinstance(mode, dict):
            continue
        accepted_types = {str(value).lower() for value in mode.get("candidate_component_types", [])}
        actual_type = str(component.get("type") or "unknown").lower()
        aliases = {"integrated_circuit": "ic", "ic": "integrated_circuit"}
        if not accepted_types or (actual_type not in accepted_types and aliases.get(actual_type) not in accepted_types):
            continue
        supports = [value for value in row.get("supporting_evidence_ids", []) if value in evidence_values]
        opposes = [value for value in row.get("opposing_evidence_ids", []) if value in evidence_values]
        validated.append({
            "component_id": cid, "suspected_fault_id": fault_id,
            "functional_block": row.get("functional_block") if row.get("functional_block") in block_values else "UNKNOWN",
            "reason": str(row.get("reason") or "")[:500],
            "supporting_evidence_ids": supports, "opposing_evidence_ids": opposes,
            "missing_information": [str(value)[:160] for value in row.get("missing_information", [])[:5]],
            "recommended_next_action": str(row.get("recommended_next_action") or "")[:240],
            "uncertainty_status": row.get("uncertainty_status", "UNVERIFIED"),
            "evidence_status": "UNVERIFIED", "rank_status": "NOT_MERGED",
        })
    return {"proposals": validated, "model": response.get("model", model),
            "status": "LOCAL_LLM_PROPOSED_UNVERIFIED"}


def select_follow_up_question(client: OllamaClient, model: str, *, pcb_context: dict[str, Any],
                              diagnosis: dict[str, Any], answered_test_ids: list[str]) -> dict[str, Any] | None:
    """Select one unasked, pre-authored diagnostic question; never invent a test."""
    available = [row for row in diagnosis.get("next_tests", [])
                 if isinstance(row, dict) and row.get("type") == "question"
                 and row.get("test_id") and row.get("test_id") not in set(answered_test_ids)]
    if not available:
        return None
    allowed = [str(row["test_id"]) for row in available]
    schema = {"type": "object", "additionalProperties": False, "required": ["test_id", "reason"],
              "properties": {"test_id": {"type": "string", "enum": allowed},
                             "reason": {"type": "string", "maxLength": 220}}}
    response = client.chat_json(model, [
        {"role": "system", "content": (
            "한 개의 다음 질문만 선택한다. 입력의 기존 안전 질문 중 후보를 구분하는 데 가장 유용한 질문을 고른다. "
            "질문·측정값·조건을 새로 만들지 않는다. 통전/고전압 검사 대신 기존 질문만 고른다. JSON만 반환한다."
        )},
        {"role": "user", "content": json.dumps({"pcb_context": pcb_context,
                                                   "candidates": available[:8]}, ensure_ascii=False, default=str)},
    ], output_schema=schema)
    data = response.get("content", {})
    choice = next((row for row in available if row.get("test_id") == data.get("test_id")), None)
    return ({**choice, "selection_reason": str(data.get("reason", ""))[:220],
             "question_source": "QWEN_SELECTED_EXISTING_SAFE_QUESTION"} if choice else None)


def explain_candidates(client: OllamaClient, model: str, *, symptom_text: str,
                       diagnosis: dict[str, Any]) -> dict[str, Any]:
    """Ask a local model to explain existing candidates without adding diagnoses."""
    payload = {
        "symptom_text": symptom_text[:2000],
        "hypotheses": [{"fault_id": row["fault_id"], "title": row["title"],
                        "supporting_evidence": row["supporting_evidence"],
                        "opposing_evidence": row["opposing_evidence"]}
                       for row in diagnosis.get("hypotheses", [])[:5]],
        "questions_available": [row.get("question", "") for row in diagnosis.get("next_tests", [])
                                if row.get("type") == "question"][:5],
    }
    schema = {
        "type": "object", "additionalProperties": False,
        "required": ["summary"],
        "properties": {
            "summary": {"type": "string", "maxLength": 900},
        },
    }
    response = client.chat_json(model, [
        {"role": "system", "content": (
            "너는 PCB 고장 진단 설명 보조자다. 제공된 JSON의 사실과 후보만 사용한다. "
            "새 고장 원인이나 후보 ID를 추가하지 말고, 확률·확정 판정을 말하지 않는다. "
            "근거가 부족하면 부족하다고 명시한다. 한국어 JSON 객체로만 답한다. "
            '고장 가설별 근거는 새로 만들지 말고 한 문장으로 전체 결과를 설명한다. '
            '형식은 반드시 {"summary":"한국어 한두 문장"} 하나뿐이다.'
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ], output_schema=schema)
    data = response.get("content", {})
    if not isinstance(data.get("summary"), str):
        raise LocalLLMError("로컬 모델 설명에 summary 문자열이 없습니다.")
    return {"summary": data["summary"][:900], "model": response.get("model", model),
            "status": "LOCAL_LLM_ASSISTED"}


def explain_component_candidates(client: OllamaClient, model: str, *,
                                 candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Explain only the supplied Top-3 component candidates in concise Korean."""
    candidates = candidates[:3]
    allowed_ids = [str(row.get("component_id", "")) for row in candidates]
    payload = [{"component_id": row.get("component_id"), "component_type": row.get("component_type"),
                "fault_hypothesis": row.get("fault_hypothesis"), "fault_title": row.get("fault_title"),
                "evidence": row.get("evidence", [])[:6],
                "supporting_evidence": row.get("supporting_evidence", [])[:6],
                "opposing_evidence": row.get("opposing_evidence", [])[:6],
                "measurement_status": row.get("measurement_status")} for row in candidates]
    schema = {"type": "object", "additionalProperties": False, "required": ["explanations"],
              "properties": {"explanations": {"type": "array", "maxItems": 3, "items": {
                  "type": "object", "additionalProperties": False,
                  "required": ["component_id", "explanation"],
                  "properties": {"component_id": {"type": "string", "enum": allowed_ids},
                                 "explanation": {"type": "string", "maxLength": 220}},
              }}}}
    response = client.chat_json(model, [
        {"role": "system", "content": (
            "PCB 고장 후보를 설명한다. 입력에 있는 후보 ID와 근거만 사용한다. "
            "후보마다 쉬운 한국어 한 문장으로, 어떤 관측이 왜 이 부품을 검사 후보로 올렸는지 말한다. "
            "근거가 약하면 그 점을 그대로 말한다. 확률, 확정 고장, 새 측정값, 새 후보를 만들지 않는다. "
            "정상 측정은 명시된 검사 가설에만 적용한다. JSON만 반환한다."
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ], output_schema=schema)
    rows = response.get("content", {}).get("explanations", [])
    allowed = set(allowed_ids)
    explanations = {str(row["component_id"]): str(row["explanation"])[:220]
                    for row in rows if isinstance(row, dict)
                    and str(row.get("component_id")) in allowed
                    and isinstance(row.get("explanation"), str)}
    return {"explanations": explanations, "model": response.get("model", model),
            "status": "LOCAL_LLM_ASSISTED"}


def suggest_unregistered_causes(client: OllamaClient, model: str, *, symptom_text: str,
                                diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a separate, low-priority LLM-only bucket; never merge into rule ranks."""
    payload = {"symptom_text": symptom_text[:2000],
               "known_hypotheses": [{"id": row["fault_id"], "title": row["title"]}
                                    for row in diagnosis.get("hypotheses", [])[:8]],
               "known_evidence": diagnosis.get("observations", {}),
               "confirmed_component_inventory": diagnosis.get("component_inventory", {})}
    evidence_values = sorted(diagnosis.get("observations", {}).keys()) or ["__none__"]
    schema = {"type": "object", "additionalProperties": False,
              "required": ["suggestions"], "properties": {
                  "suggestions": {"type": "array", "maxItems": 2, "items": {
                      "type": "object", "additionalProperties": False,
                      "required": ["title", "reason", "evidence_used"],
                      "properties": {"title": {"type": "string", "maxLength": 160},
                                     "reason": {"type": "string", "maxLength": 500},
                                     "evidence_used": {"type": "array", "maxItems": 5,
                                                       "items": {"type": "string", "enum": evidence_values}}},
                  }}},
              }
    response = client.chat_json(model, [
        {"role": "system", "content": (
            "PCB 고장 원인 보완 검토자다. 규칙 엔진 후보에 없는 가능성을 최대 2개 제안한다. "
            "제안은 미검증 가설이며 순위를 매기거나 확정하지 않는다. 없는 배선·핀·측정값을 만들지 않는다. "
            "부품 고장은 규칙 엔진이 확인된 소자 목록으로만 평가한다. 확인 목록에 없는 소자 고장을 제안하지 말고, "
            "구체 부품 ID나 위치를 지목하지 않는다. 목록이 미완료면 부품의 존재/부재를 단정하지 않는다. "
            "JSON: {\"suggestions\":[{\"title\":\"...\","
            "\"reason\":\"입력 증상과의 관계\",\"evidence_used\":[\"입력에 존재하는 관측 키\"]}]}"
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ], output_schema=schema)
    data = response.get("content", {})
    observed = set(diagnosis.get("observations", {}))
    suggestions = []
    suggestion_rows = data.get("suggestions", [])
    if not isinstance(suggestion_rows, list):
        suggestion_rows = []
    for row in suggestion_rows[:2]:
        if not isinstance(row, dict) or not isinstance(row.get("title"), str) or not isinstance(row.get("reason"), str):
            continue
        evidence_used = [code for code in row.get("evidence_used", []) if code in observed]
        suggestions.append({"title": row["title"][:160], "reason": row["reason"][:500],
                            "evidence_used": evidence_used, "status": "UNVERIFIED_LLM_SUGGESTION",
                            "ranked_with_rule_candidates": False,
                            "id": f"LLM_UNVERIFIED_{len(suggestions) + 1}"})
    return suggestions
