"""Shared component labels for PCB detection, review, and diagnosis."""

from __future__ import annotations

COMPONENT_CLASSES = (
    "battery", "switch", "buzzer", "capacitor", "clock", "connector", "diode",
    "display", "fuse", "heatsink", "ic", "inductor", "led", "potentiometer",
    "relay", "resistor", "transformer", "transistor",
)
UNKNOWN_CLASS = "unknown"
NOT_COMPONENT_CLASS = "not_component"
COMPONENT_LABELS = (*COMPONENT_CLASSES, UNKNOWN_CLASS, NOT_COMPONENT_CLASS)

CLASS_ALIASES = {
    "button": "switch", "integrated_circuit": "ic", "unknown_component": "unknown",
    "IC": "ic", "LED": "led",
}


def normalize_component_class(value: object, *, allow_not_component: bool = True) -> str:
    """Return a canonical shared label, retaining ambiguity as ``unknown``."""
    raw = str(value or "").strip()
    canonical = CLASS_ALIASES.get(raw, CLASS_ALIASES.get(raw.casefold(), raw.casefold()))
    allowed = COMPONENT_LABELS if allow_not_component else (*COMPONENT_CLASSES, UNKNOWN_CLASS)
    return canonical if canonical in allowed else UNKNOWN_CLASS
