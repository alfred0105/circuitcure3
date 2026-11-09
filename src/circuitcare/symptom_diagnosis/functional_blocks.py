"""Low-confidence functional block hypotheses from labels, OCR hints, and layout."""

from __future__ import annotations

from math import hypot
import re
from typing import Any

from circuitcare.symptom_diagnosis.ic_identity import resolve_ic_identity


def analyze_functional_blocks(
    components: list[dict[str, Any]], taxonomy: dict[str, list[str]], *,
    silkscreen_labels: list[dict[str, Any]] | None = None,
    ic_catalog: dict[str, dict] | None = None,
) -> list[dict[str, Any]]:
    """Return inferred role candidates with evidence and explicit uncertainty."""
    boxes = [row.get("bbox_xyxy", row.get("bbox")) for row in components]
    valid = [box for box in boxes if isinstance(box, (list, tuple)) and len(box) == 4]
    if valid:
        width = max(float(box[2]) for box in valid) - min(float(box[0]) for box in valid)
        height = max(float(box[3]) for box in valid) - min(float(box[1]) for box in valid)
        diagonal = max(hypot(width, height), 1.0)
    else:
        diagonal = 1.0

    identities: dict[str, dict] = {}
    result = []
    labels = silkscreen_labels or []
    for component in components:
        component_id = component.get("component_id")
        kind = str(component.get("component_type", "unknown"))
        own_roles = set(taxonomy.get(kind, []))
        identity = None
        if kind == "ic" and component.get("ocr_marking"):
            identity = resolve_ic_identity(str(component["ocr_marking"]), ic_catalog)
            own_roles.update(identity["role_candidates"])
        identities[str(component_id)] = {"roles": own_roles, "identity": identity}

    for component in components:
        component_id = component.get("component_id")
        kind = str(component.get("component_type", "unknown"))
        box = component.get("bbox_xyxy", component.get("bbox"))
        nearby = []
        if isinstance(box, (list, tuple)) and len(box) == 4:
            cx, cy = (float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2
            for other in components:
                if other is component:
                    continue
                other_box = other.get("bbox_xyxy", other.get("bbox"))
                if not isinstance(other_box, (list, tuple)) or len(other_box) != 4:
                    continue
                ox, oy = (float(other_box[0]) + float(other_box[2])) / 2, (float(other_box[1]) + float(other_box[3])) / 2
                distance = hypot(cx - ox, cy - oy) / diagonal
                if distance <= 0.18:
                    nearby.append((other, distance))

        votes: dict[str, float] = {}
        evidence_by_role: dict[str, list[str]] = {}
        own = identities.get(str(component_id), {})
        for role in own.get("roles", set()):
            votes[role] = votes.get(role, 0.0) + 1.0
            evidence_by_role.setdefault(role, []).append("부품 종류 또는 IC 각인 힌트")
        related_ids = []
        for other, distance in nearby:
            other_id = str(other.get("component_id"))
            other_info = identities.get(other_id, {})
            shared_roles = set(own.get("roles", set())) & set(other_info.get("roles", set()))
            complement_roles = set(other_info.get("roles", set()))
            for role in complement_roles:
                # Neighbor support is deliberately weaker than a part's own role.
                votes[role] = votes.get(role, 0.0) + max(0.1, 0.35 * (1.0 - distance / 0.18))
                evidence_by_role.setdefault(role, []).append(f"인접 부품 {other_id}")
            if shared_roles:
                related_ids.append(other_id)

        cx = cy = None
        if isinstance(box, (list, tuple)) and len(box) == 4:
            cx, cy = (float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2
        nearby_text = []
        if cx is not None:
            for label in labels:
                text_box = label.get("bbox_xyxy", label.get("bbox"))
                if not isinstance(text_box, (list, tuple)) or len(text_box) != 4:
                    continue
                tx, ty = (float(text_box[0]) + float(text_box[2])) / 2, (float(text_box[1]) + float(text_box[3])) / 2
                if hypot(cx - tx, cy - ty) / diagonal <= 0.08:
                    nearby_text.append(str(label.get("text", "")))

        # Reference designators and nearby signal labels are weak hints, not
        # confirmed connectivity. Apply them to the component whose box is nearby.
        reference_roles = {
            "Q": ("driver",), "U": ("control", "power"), "FB": ("power",),
            "L": ("power",), "D": ("power",), "C": ("power",),
            "R": ("power",), "BZ": ("buzzer",), "K": ("relay",),
            "X": ("control_timing",), "Y": ("control_timing",),
            "J": ("power",), "SW": ("control_input",),
        }
        hint_text = " ".join(nearby_text).upper()
        for prefix, roles_for_prefix in reference_roles.items():
            if re.search(rf"(?<![A-Z0-9]){prefix}\s*\d+", hint_text):
                for role in roles_for_prefix:
                    votes[role] = votes.get(role, 0.0) + 0.5
                    evidence_by_role.setdefault(role, []).append(f"근처 실크스크린 참조명 {prefix}")
        if re.search(r"\b(VIN|VOUT|VCC|GND|VBAT|3V3|5V)\b", hint_text):
            votes["power"] = votes.get("power", 0.0) + 0.6
            evidence_by_role.setdefault("power", []).append("근처 실크스크린 전원 신호명")

        roles = sorted(votes, key=lambda role: (-votes[role], role))[:4]
        identity = own.get("identity")
        result.append({
            "component_id": component_id, "component_type": kind,
            "candidate_blocks": roles, "nearby_component_ids": [str(row.get("component_id")) for row, _ in nearby],
            "related_component_ids": related_ids, "nearby_silkscreen": nearby_text,
            "role_evidence": {role: list(dict.fromkeys(evidence_by_role[role])) for role in roles},
            "ic_identity": identity, "status": "INFERRED", "verified": False,
        })
    return result
