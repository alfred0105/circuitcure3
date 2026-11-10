import json
from types import SimpleNamespace

import pytest

from circuitcare.frontend.board_component_registry import (
    REGISTRY_KEY, build_board_component_registry, confirm_component_match,
    current_source_observation_fingerprint, diagnosis_components, measurement_component_aliases, remove_component_match,
    resolve_measurement_board_component, source_component_key,
)


def _component(component_id, *, status="UNCONFIRMED", component_type="resistor", bbox=(1, 2, 8, 9)):
    return SimpleNamespace(
        component_id=component_id, component_type=component_type, bbox_xyxy=bbox,
        coordinate_frame="SOURCE_PHOTO_PIXEL", identification_status=status,
        presence_status="CONFIRMED" if status == "USER_CONFIRMED" else "UNCONFIRMED",
        component_type_status=status, ocr_marking="103" if status == "USER_CONFIRMED" else None,
        ocr_candidates=("R1",), possible_part_number=None, confirmed_functional_roles=(),
        evidence=("test evidence",), overlay_source="yolo",
    )


def _snapshots():
    return {
        "front": {"components": (_component("local-1", status="USER_CONFIRMED"),),
                  "image_fingerprint": "front-hash", "review_version": "front-review"},
        "back": {"components": (_component("local-1", bbox=(1, 2, 8, 9)),),
                 "image_fingerprint": "back-hash", "review_version": "back-review"},
        "close": {"components": (_component("close-1", bbox=(1, 2, 8, 9)),),
                  "image_fingerprint": "close-hash", "review_version": "close-review"},
    }


def test_registry_keeps_close_boxes_independent_without_explicit_confirmation():
    state = {}
    registry = build_board_component_registry(
        state, project_id="p1", board_id="b1", photo_snapshots=_snapshots(),
    )
    assert len(registry["components"]) == 3
    assert len(set(registry["source_to_board_id"].values())) == 3
    # No pixel coordinates are projected into board-level diagnosis inputs.
    assert all(item.bbox_xyxy is None for item in diagnosis_components(registry))


def test_user_confirmed_link_merges_sources_preserves_canonical_id_and_can_split():
    state = {}
    first = build_board_component_registry(state, project_id="p1", board_id="b1",
                                           photo_snapshots=_snapshots())
    front = source_component_key("front", "local-1")
    back = source_component_key("back", "local-1")
    front_id = first["source_to_board_id"][front]
    confirm_component_match(state, project_id="p1", board_id="b1",
                            left_source_key=front, right_source_key=back)
    merged = build_board_component_registry(state, project_id="p1", board_id="b1",
                                            photo_snapshots=_snapshots())
    assert len(merged["components"]) == 2
    assert merged["source_to_board_id"][front] == merged["source_to_board_id"][back] == front_id
    merged_row = next(row for row in merged["components"] if row["board_component_id"] == front_id)
    assert merged_row["identity_status"] == "USER_CONFIRMED"
    assert set(merged_row["source_photo_ids"]) == {"front", "back"}

    assert remove_component_match(state, project_id="p1", board_id="b1",
                                  left_source_key=front, right_source_key=back)
    split = build_board_component_registry(state, project_id="p1", board_id="b1",
                                           photo_snapshots=_snapshots())
    assert len(split["components"]) == 3
    assert split["source_to_board_id"][front] == front_id
    assert split["source_to_board_id"][front] != split["source_to_board_id"][back]


def test_replacing_a_photo_disables_old_identity_link_until_reconfirmed():
    state = {}
    before = _snapshots()
    build_board_component_registry(state, project_id="p1", board_id="b1", photo_snapshots=before)
    front = source_component_key("front", "local-1")
    back = source_component_key("back", "local-1")
    confirm_component_match(state, project_id="p1", board_id="b1",
                            left_source_key=front, right_source_key=back)
    replaced = _snapshots()
    replaced["front"] = {**replaced["front"], "image_fingerprint": "replacement-image"}
    registry = build_board_component_registry(state, project_id="p1", board_id="b1",
                                              photo_snapshots=replaced)
    assert registry["source_to_board_id"][front] != registry["source_to_board_id"][back]
    assert any(item["reason"] == "PHOTO_REPLACED_RECONFIRMATION_REQUIRED"
               for item in registry["inactive_links"])


def test_redetection_review_version_disables_link_until_user_reconfirms():
    state = {}
    snapshots = _snapshots()
    build_board_component_registry(state, project_id="p1", board_id="b1",
                                   photo_snapshots=snapshots)
    front = source_component_key("front", "local-1")
    back = source_component_key("back", "local-1")
    confirm_component_match(state, project_id="p1", board_id="b1",
                            left_source_key=front, right_source_key=back)
    changed = _snapshots()
    changed["front"] = {**changed["front"], "review_version": "redetected"}
    registry = build_board_component_registry(state, project_id="p1", board_id="b1",
                                              photo_snapshots=changed)
    assert registry["source_to_board_id"][front] != registry["source_to_board_id"][back]
    assert any(item["reason"] == "PHOTO_REDETECTED_RECONFIRMATION_REQUIRED"
               for item in registry["inactive_links"])


def test_merge_rejects_direct_and_transitive_same_photo_collisions_without_mutating_state():
    state = {}
    snapshots = {
        "a": {"components": (_component("a1"), _component("a2", bbox=(20, 20, 30, 30))),
              "image_fingerprint": "a-hash"},
        "b": {"components": (_component("b1"),), "image_fingerprint": "b-hash"},
    }
    registry = build_board_component_registry(state, project_id="p1", board_id="b1",
                                              photo_snapshots=snapshots)
    a1, a2, b1 = (source_component_key(photo, local)
                  for photo, local in (("a", "a1"), ("a", "a2"), ("b", "b1")))
    confirm_component_match(state, project_id="p1", board_id="b1",
                            left_source_key=a1, right_source_key=b1)
    merged = build_board_component_registry(state, project_id="p1", board_id="b1",
                                            photo_snapshots=snapshots)
    owner = next(iter(state[REGISTRY_KEY]))
    before_links = state[REGISTRY_KEY][owner]["confirmed_links"]
    with pytest.raises(ValueError, match="같은 사진"):
        confirm_component_match(state, project_id="p1", board_id="b1",
                                left_source_key=a2, right_source_key=b1)
    after_links = state[REGISTRY_KEY][owner]["confirmed_links"]
    assert after_links == before_links
    assert merged["source_to_board_id"][a1] == merged["source_to_board_id"][b1]
    assert merged["source_to_board_id"][a2] != merged["source_to_board_id"][b1]


def test_corrupt_transitive_same_photo_link_is_retained_but_never_merged():
    state = {}
    snapshots = {
        "a": {"components": (_component("a1"), _component("a2", bbox=(20, 20, 30, 30))),
              "image_fingerprint": "a-hash"},
        "b": {"components": (_component("b1"),), "image_fingerprint": "b-hash"},
    }
    registry = build_board_component_registry(state, project_id="p1", board_id="b1",
                                              photo_snapshots=snapshots)
    a1 = source_component_key("a", "a1")
    a2 = source_component_key("a", "a2")
    b1 = source_component_key("b", "b1")
    links = [tuple(sorted((a1, b1))), tuple(sorted((a2, b1)))]
    owner = next(iter(state[REGISTRY_KEY]))
    entry = state[REGISTRY_KEY][owner]
    source_map = {source["source_key"]: source for row in registry["components"]
                  for source in row["source_components"]}
    entry["confirmed_links"] = [list(link) for link in links]
    entry["link_versions"] = {
        json.dumps(list(link), separators=(",", ":")):
            [{"image_fingerprint": ("a-hash" if json.loads(key)[0] == "a" else "b-hash"),
              "review_version": None, "component_map_version": None,
              "source_observation_fingerprint": source_map[key]["source_observation_fingerprint"]}
             for key in link]
        for link in links
    }
    rebuilt = build_board_component_registry(state, project_id="p1", board_id="b1",
                                             photo_snapshots=snapshots)
    assert rebuilt["source_to_board_id"][a1] == rebuilt["source_to_board_id"][b1]
    assert rebuilt["source_to_board_id"][a2] != rebuilt["source_to_board_id"][b1]
    assert any(item["reason"] == "SAME_PHOTO_COMPONENT_COLLISION_RECONFIRMATION_REQUIRED"
               for item in rebuilt["inactive_links"])


def test_measurement_aliases_are_only_emitted_when_local_id_is_unambiguous():
    registry = build_board_component_registry(
        {}, project_id="p1", board_id="b1", photo_snapshots=_snapshots(),
    )
    aliases = measurement_component_aliases(registry)
    assert aliases["local-1"] is None
    assert aliases["close-1"].startswith("bc-")
    observation_fp = current_source_observation_fingerprint(
        registry, photo_id="front", component_id="local-1")
    board_id, status = resolve_measurement_board_component(
        registry, photo_id="front", component_id="local-1", observation_fingerprint=observation_fp)
    assert board_id is not None and status == "VERIFIED_SOURCE_IDENTITY"
    assert resolve_measurement_board_component(
        registry, photo_id="front", component_id="local-1",
        photo_fingerprint="replacement-hash", observation_fingerprint=observation_fp,
    ) == (None, "PHOTO_REPLACED_RECONFIRMATION_REQUIRED")
    assert resolve_measurement_board_component(
        registry, photo_id=None, component_id="local-1") == (None, "UNRESOLVED")
    unique_id, unique_status = resolve_measurement_board_component(
        registry, photo_id=None, component_id="close-1")
    assert unique_id is None
    assert unique_status == "RECONFIRMATION_REQUIRED"


def test_project_and_board_registry_data_are_isolated_and_photo_manifest_versions_changes():
    state = {}
    first = build_board_component_registry(state, project_id="p1", board_id="b1",
                                           photo_snapshots=_snapshots())
    second = build_board_component_registry(state, project_id="p2", board_id="b1",
                                            photo_snapshots=_snapshots())
    assert first["components"][0]["board_component_id"] != second["components"][0]["board_component_id"]
    assert len(state[REGISTRY_KEY]) == 2
    altered = _snapshots()
    altered["top_extra"] = {"components": (), "image_fingerprint": "added", "review_version": "v1"}
    updated = build_board_component_registry(state, project_id="p1", board_id="b1",
                                             photo_snapshots=altered)
    assert updated["version"] != first["version"]


def test_excluded_photo_box_is_retained_in_registry_but_not_sent_to_diagnosis():
    snapshots = {"front": {"registry_components": ({
        "component_id": "excluded-1", "component_type": "ic", "bbox_xyxy": [0, 0, 10, 10],
        "identification_status": "EXCLUDED", "presence_status": "EXCLUDED",
        "component_type_status": "UNCONFIRMED",
    },)}}
    registry = build_board_component_registry({}, project_id="p1", board_id="b1",
                                              photo_snapshots=snapshots)
    assert registry["components"][0]["source_components"][0]["review_status"] == "EXCLUDED"
    assert diagnosis_components(registry) == ()
