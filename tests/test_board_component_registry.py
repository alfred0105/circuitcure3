from types import SimpleNamespace

from circuitcare.frontend.board_component_registry import (
    REGISTRY_KEY, build_board_component_registry, confirm_component_match,
    diagnosis_components, measurement_component_aliases, remove_component_match,
    source_component_key,
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


def test_measurement_aliases_are_only_emitted_when_local_id_is_unambiguous():
    registry = build_board_component_registry(
        {}, project_id="p1", board_id="b1", photo_snapshots=_snapshots(),
    )
    aliases = measurement_component_aliases(registry)
    assert aliases["local-1"] is None
    assert aliases["close-1"].startswith("bc-")


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
