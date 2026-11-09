"""Version-scoped diagnosis overlay state shared by the panel and PCB viewer."""

from __future__ import annotations

from typing import Any


def build_overlay_state(
    *, project_id: str, board_id: str | None, component_map_version: str | None,
    result: dict[str, Any], previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate the current Top-3 overlay from durable verified component history."""
    candidates = result.get("component_candidates", [])[:3]
    history = list((previous or {}).get("verified_component_history", []))
    historical_rows = [*(previous or {}).get("active_candidates", {}).values(), *candidates]
    known_history = {(row.get("component_id"), row.get("fault_hypothesis"), row.get("measurement_verdict"),
                      tuple(row.get("evidence_ids", []))) for row in history if isinstance(row, dict)}
    for row in historical_rows:
        if not isinstance(row, dict) or row.get("measurement_verdict") not in {"supports", "refutes"}:
            continue
        key = (row.get("component_id"), row.get("fault_hypothesis"), row.get("measurement_verdict"),
               tuple(row.get("evidence_ids", [])))
        if key not in known_history:
            history.append(dict(row, input_fingerprint=result.get("input_fingerprint")))
            known_history.add(key)
    return {
        "project_id": project_id,
        "board_id": board_id,
        "component_map_version": component_map_version,
        "diagnosis_id": result.get("diagnosis_id"),
        "input_fingerprint": result.get("input_fingerprint"),
        "active_candidates": {str(row["component_id"]): row for row in candidates},
        "verified_component_history": history[-1000:],
        "joint_candidates": result.get("joint_candidates", []),
    }


def current_overlay_candidates(
    value: Any, *, project_id: str, board_id: str | None,
    component_map_version: str | None, input_fingerprint: str | None,
) -> dict[str, dict[str, Any]]:
    """Return candidates only for a complete, exact current project/board/map/input scope."""
    if not isinstance(value, dict) or not project_id or not board_id or not component_map_version or not input_fingerprint:
        return {}
    if value.get("project_id") != project_id or value.get("board_id") != board_id:
        return {}
    if value.get("component_map_version") != component_map_version:
        return {}
    if value.get("input_fingerprint") != input_fingerprint:
        return {}
    candidates = value.get("active_candidates")
    return candidates if isinstance(candidates, dict) else {}
