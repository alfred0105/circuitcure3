import io
from types import SimpleNamespace

from PIL import Image

from circuitcare.dashboard.components.pcb_workspace import (
    OVERLAPPING_SELECTION_KEY, coordinate_status_for_viewer, render_component_viewer,
    scale_overlay_to_display, validate_component_selection_event,
)
from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.frontend.component_overlay_models import display_identity_status
from circuitcare.dashboard.views.unified_workspace import (
    _bind_review_models_to_map, _prepare_photo_component_snapshot,
)
from circuitcare.frontend.models import ImageRecord, ProjectContext, SelectionState
from circuitcare.vision.box_reconciliation import board_visible_review_rows
from circuitcare.vision.component_recognition_panel import (
    _store_predictions, ensure_review_row_for_component, invalidate_component_review_state,
    record_box_exclusion, review_keys, restore_review_row, undo_last_box_exclusion, update_review_row,
)


def _model():
    return ComponentOverlayViewModel(
        project_id="p1", board_id="b1", board_side="TOP", component_id="c1",
        component_type="resistor", display_label="R4", canonical_frame_id="frame1",
        coordinate_frame="EXIF_CORRECTED_IMAGE_PIXEL", bbox_xyxy=(10, 20, 30, 50),
        polygon_xy=((10, 20), (30, 20), (30, 50), (10, 50)),
        source_backend_photo_ids=("bp1",), observation_ids=("o1",),
        detection_confidence=.8, identification_status="SUGGESTED_UNCONFIRMED",
        identity_match_status="NEW_COMPONENT", ocr_marking=None,
        possible_part_number=None, evidence=("MODEL_DETECTION_CANDIDATE",),
        overlay_status="DETECTION_CANDIDATE",
    )


def test_original_coordinates_scale_with_known_exif_corrected_resize():
    scaled = scale_overlay_to_display((_model(),), source_size=(100, 100), display_size=(50, 50))[0]
    assert scaled.bbox_xyxy == (5, 10, 15, 25)
    assert scaled.polygon_xy == ((5, 10), (15, 10), (15, 25), (5, 25))


def test_click_and_clear_selection_events_are_scoped_and_monotonic():
    event = {"project_id": "p1", "board_id": "b1", "board_side": "TOP",
             "canonical_frame_id": "frame1", "component_id": "c1",
             "selection_action": "select", "event_seq": 1, "event_id": "evt-1"}
    kwargs = dict(project_id="p1", board_id="b1", board_side="TOP", frame_id="frame1",
                  allowed_ids={"c1"}, last_sequence=0)
    assert validate_component_selection_event(event, **kwargs) == (True, None)
    overlapping = {**event, "overlapping_component_ids": ["c1", "c2"]}
    assert validate_component_selection_event(overlapping, **{**kwargs, "allowed_ids": {"c1", "c2"}}) == (True, None)
    # Other overlap candidates may be stale between the browser click and the
    # current render; the selected, owner-validated ID must remain selectable.
    assert validate_component_selection_event(overlapping, **kwargs) == (True, None)
    assert validate_component_selection_event(
        {**event, "component_id": "missing", "overlapping_component_ids": ["missing", "c2"]},
        **{**kwargs, "allowed_ids": {"c1"}},
    )[0] is False
    assert validate_component_selection_event(
        {**event, "event_epoch": "old"}, **{**kwargs, "event_epoch": "current"}
    ) == (False, "이전 Viewer 인스턴스의 선택 이벤트")
    assert validate_component_selection_event({**event, "project_id": "other"}, **kwargs)[0] is False
    assert validate_component_selection_event({**event, "board_id": "other"}, **kwargs)[0] is False
    assert validate_component_selection_event({**event, "board_side": "BOTTOM"}, **kwargs)[0] is False
    assert validate_component_selection_event({**event, "canonical_frame_id": "old-frame"}, **kwargs)[0] is False
    assert validate_component_selection_event({**event, "component_id": "missing"}, **kwargs)[0] is False
    assert validate_component_selection_event(event, **{**kwargs, "last_sequence": 1})[0] is False
    assert validate_component_selection_event({k: v for k, v in event.items() if k != "event_id"}, **kwargs)[0] is False
    clear = {**event, "component_id": None, "selection_action": "clear", "event_seq": 2}
    assert validate_component_selection_event(clear, **{**kwargs, "last_sequence": 1}) == (True, None)


def test_viewer_click_event_updates_real_streamlit_selection_and_overlap_stack(monkeypatch):
    import circuitcare.dashboard.components.pcb_workspace as workspace

    streamlit = SimpleNamespace(session_state={})
    project = ProjectContext(project_id="p1", board_id="b1")
    from circuitcare.frontend.workbench_state import SELECTION_KEY
    streamlit.session_state[SELECTION_KEY] = SelectionState(project_id="p1", board_id="b1")
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(output, format="PNG")
    components = tuple(ComponentOverlayViewModel(**{**_model().__dict__, "component_id": item})
                       for item in ("c1", "c2"))

    def fake_render(_st, **kwargs):
        key = kwargs["key"]
        streamlit.session_state[key] = {"selection_event": {
            "project_id": "p1", "board_id": "b1", "board_side": "TOP",
            "canonical_frame_id": "photo1", "component_id": "c1", "selection_action": "select",
            "overlapping_component_ids": ["c1", "c2"], "event_seq": 1, "event_id": "click-1",
            "event_epoch": kwargs["event_epoch"],
        }}
        kwargs["on_selection_event_change"]()
        return object()

    monkeypatch.setattr(workspace, "render_pcb_viewer", fake_render)
    available, viewer_key, _status = render_component_viewer(
        streamlit, image=None, source_image_bytes=output.getvalue(), canonical_png_bytes=None,
        image_size_px=(32, 32), components=components, project=project, board_side="TOP",
        frame_id="photo1", selected_component_id=None, map_cache_key="map1", image_mode="원본 사진",
    )
    assert available is True
    selection = streamlit.session_state[SELECTION_KEY]
    assert selection.selected_component == "c1"
    assert streamlit.session_state[f"{OVERLAPPING_SELECTION_KEY}_{viewer_key}"] == ["c1", "c2"]


def test_viewer_click_keeps_valid_selection_and_filters_stale_overlap_ids(monkeypatch):
    import circuitcare.dashboard.components.pcb_workspace as workspace

    streamlit = SimpleNamespace(session_state={})
    project = ProjectContext(project_id="p1", board_id="b1")
    from circuitcare.frontend.workbench_state import SELECTION_KEY
    streamlit.session_state[SELECTION_KEY] = SelectionState(project_id="p1", board_id="b1")
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(output, format="PNG")
    components = (_model(),)

    def fake_render(_st, **kwargs):
        key = kwargs["key"]
        streamlit.session_state[key] = {"selection": {
            "project_id": "p1", "board_id": "b1", "board_side": "TOP",
            "canonical_frame_id": "photo1", "component_id": "c1", "selection_action": "select",
            "overlapping_component_ids": ["c1", "deleted-stale-id"], "event_seq": 1,
            "event_id": "click-stale-overlap", "event_epoch": kwargs["event_epoch"],
        }}
        kwargs["on_selection_change"]()
        return object()

    monkeypatch.setattr(workspace, "render_pcb_viewer", fake_render)
    _available, viewer_key, _status = render_component_viewer(
        streamlit, image=None, source_image_bytes=output.getvalue(), canonical_png_bytes=None,
        image_size_px=(32, 32), components=components, project=project, board_side="TOP",
        frame_id="photo1", selected_component_id=None, map_cache_key="map1", image_mode="원본 사진",
    )
    assert streamlit.session_state[SELECTION_KEY].selected_component == "c1"
    assert streamlit.session_state[f"{OVERLAPPING_SELECTION_KEY}_{viewer_key}"] == ["c1"]


def test_component_result_is_applied_even_when_callback_session_view_is_late(monkeypatch):
    import circuitcare.dashboard.components.pcb_workspace as workspace

    reruns = []
    streamlit = SimpleNamespace(session_state={}, rerun=lambda: reruns.append(True))
    project = ProjectContext(project_id="p1", board_id="b1")
    from circuitcare.frontend.workbench_state import SELECTION_KEY
    streamlit.session_state[SELECTION_KEY] = SelectionState(project_id="p1", board_id="b1")
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(output, format="PNG")
    event = {"project_id": "p1", "board_id": "b1", "board_side": "TOP",
             "canonical_frame_id": "photo1", "component_id": "c1", "selection_action": "select",
             "overlapping_component_ids": ["c1"], "event_seq": 1, "event_id": "result-click",
             "event_epoch": "unused"}

    def fake_render(_st, **kwargs):
        return SimpleNamespace(selection={**event, "event_epoch": kwargs["event_epoch"]}, viewer_status=None)

    monkeypatch.setattr(workspace, "render_pcb_viewer", fake_render)
    workspace.render_component_viewer(
        streamlit, image=None, source_image_bytes=output.getvalue(), canonical_png_bytes=None,
        image_size_px=(32, 32), components=(_model(),), project=project, board_side="TOP",
        frame_id="photo1", selected_component_id=None, map_cache_key="map1", image_mode="원본 사진",
    )
    assert streamlit.session_state[SELECTION_KEY].selected_component == "c1"
    assert reruns == [True]


def test_pcb_viewer_uses_shared_hit_test_from_real_click_and_does_not_capture_plain_click():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "src/circuitcare/dashboard/components/assets/pcb_viewer.js").read_text()
    assert 'listen(viewport,"click"' in source
    assert "const hits=hitComponents(event)" in source
    assert "[pcb-selection] click" in source
    assert "viewport.setPointerCapture(event.pointerId)" in source
    # Pointer capture is acquired only after the movement crosses the drag threshold.
    pointerdown = source.split('listen(viewport,"pointerdown"', 1)[1].split('listen(viewport,"pointermove"', 1)[0]
    plain_pan_start = pointerdown.split("state.drag={", 1)[1]
    assert "setPointerCapture" not in plain_pan_start


def test_review_row_is_single_source_for_selected_box_edit_exclusion_and_restore():
    box = _model()
    rows = []
    row = ensure_review_row_for_component(rows, box)
    update_review_row(row, component_type="capacitor", status="확인", bbox_xyxy=[12, 22, 34, 56])
    assert row["component_id"] == "c1"
    assert row["소자 종류"] == "capacitor" and row["상태"] == "확인"
    assert [row[key] for key in ("x1", "y1", "x2", "y2")] == [12, 22, 34, 56]
    state = {}
    record_box_exclusion(state, scope="scope", row=row)
    assert row["상태"] == "제외" and row["user_edited"] is True
    assert board_visible_review_rows(rows) == []
    assert undo_last_box_exclusion(state, scope="scope", rows=rows) == "c1"
    assert row["상태"] == "확인" and board_visible_review_rows(rows) == [row]

    rerun_state = {"rows": [row]}
    record_box_exclusion(rerun_state, scope="scope", row=row)
    restore_review_row(row)
    assert row["상태"] == "확인"
    record_box_exclusion(rerun_state, scope="scope", row=row)
    _store_predictions(rerun_state, "rows", {"detections": [{
        "class_name": "resistor", "confidence": .91, "bbox_xyxy": [11, 21, 31, 51],
    }]})
    assert rerun_state["rows"][0]["component_id"] == "c1"
    assert rerun_state["rows"][0]["상태"] == "제외"
    assert board_visible_review_rows(rerun_state["rows"]) == []


def test_component_change_invalidates_diagnosis_but_preserves_measurement_history():
    state = {"circuitcare_v5_measurements": {"owner": ["measurement"]},
             "cc_symptom_diagnosis_p1": {"result": {"status": "old"}, "input_fingerprint": "old"}}
    invalidate_component_review_state(state, scope="scope", project_id="p1", board_id="b1")
    assert state["cc_symptom_diagnosis_p1"]["result"] is None
    assert state["circuitcare_v5_measurements"]["owner"] == ["measurement"]


def test_map_and_detector_boxes_bind_to_one_review_identity():
    map_box = ComponentOverlayViewModel(**{**_model().__dict__, "component_id": "map-id",
                                           "fault_candidate_status": "SUSPECTED_FAULT",
                                           "fault_history": ({"visual_status": "FAULT_REFUTED"},)})
    review_box = ComponentOverlayViewModel(**{**_model().__dict__, "component_id": "detector-id",
                                              "component_type": "capacitor", "bbox_xyxy": (11, 21, 31, 51),
                                              "overlay_source": "yolo"})
    rows = [{"component_id": "detector-id", "소자 종류": "capacitor", "상태": "확인",
             "x1": 11, "y1": 21, "x2": 31, "y2": 51}]
    merged, rows = _bind_review_models_to_map((map_box,), (review_box,), rows)
    assert len(merged) == 1
    assert merged[0].component_id == "detector-id"
    assert merged[0].bbox_xyxy == (11, 21, 31, 51)
    assert merged[0].fault_candidate_status is None
    assert merged[0].fault_history == ({"visual_status": "FAULT_REFUTED"},)
    assert rows[0]["associated_component_ids"] == ["map-id"]
    rows[0]["상태"] = "제외"
    hidden, _rows = _bind_review_models_to_map((map_box,), (review_box,), rows)
    assert hidden == ()


def test_ambiguous_map_geometry_is_not_arbitrarily_bound_or_drawn_twice():
    map_box = ComponentOverlayViewModel(**{**_model().__dict__, "component_id": "map-id"})
    left = ComponentOverlayViewModel(**{**_model().__dict__, "component_id": "review-a",
                                        "overlay_source": "yolo"})
    right = ComponentOverlayViewModel(**{**_model().__dict__, "component_id": "review-b",
                                         "overlay_source": "yolo"})
    rows = [
        {"component_id": item, "소자 종류": "resistor", "상태": "검수 필요",
         "x1": 10, "y1": 20, "x2": 30, "y2": 50}
        for item in ("review-a", "review-b")
    ]
    merged, rows = _bind_review_models_to_map((map_box,), (left, right), rows)
    assert {item.component_id for item in merged} == {"review-a", "review-b"}
    assert all("associated_component_ids" not in row for row in rows)
    assert all(row["map_match_status"] == "AMBIGUOUS_GEOMETRY" for row in rows)


def test_photo_component_snapshots_are_prepared_without_rendering_the_viewer(monkeypatch):
    image_bytes = []
    for color in ("red", "blue"):
        output = io.BytesIO()
        Image.new("RGB", (40, 30), color).save(output, format="PNG")
        image_bytes.append(output.getvalue())
    project = ProjectContext(
        "p1", "b1", image_records=(
            ImageRecord("photo-1", side="front"), ImageRecord("photo-2", side="back"),
        ),
    )
    state = {"circuitcare_v2_active_photo_id": "photo-1",
             "circuitcare_v2_photo_choice": "photo-1"}
    st = SimpleNamespace(session_state=state)

    def fake_detection(st_api, *, image_bytes, project_id, photo_id,
                       review_image_bytes=None, review_image_size=None):
        scope = unified_workspace.component_review_scope(review_image_bytes, project_id, photo_id)
        _result_key, rows_key, _error_key = review_keys(scope)
        component_id = "R1" if photo_id == "photo-1" else "C1"
        st_api.session_state[rows_key] = [{
            "component_id": component_id, "소자 종류": "resistor" if component_id == "R1" else "capacitor",
            "상태": "검수 필요", "x1": 1, "y1": 1, "x2": 10, "y2": 10,
        }]
        return {"detections": []}, None

    from circuitcare.dashboard.views import unified_workspace
    monkeypatch.setattr(unified_workspace, "ensure_component_detection", fake_detection)
    photos = [{"image_id": "photo-1", "data": image_bytes[0]},
              {"image_id": "photo-2", "data": image_bytes[1]}]
    first = _prepare_photo_component_snapshot(st, project, photos)
    assert [item.component_id for item in first["components"]] == ["R1"]

    state["circuitcare_v2_photo_choice"] = "photo-2"
    state["circuitcare_v2_active_photo_id"] = "photo-2"
    second = _prepare_photo_component_snapshot(st, project, photos)
    assert [item.component_id for item in second["components"]] == ["C1"]
    snapshots = state["circuitcare_v6_photo_component_snapshots"]
    assert first["analysis_scope_key"] != second["analysis_scope_key"]
    assert snapshots[first["analysis_scope_key"]]["photo_id"] == "photo-1"
    assert snapshots[second["analysis_scope_key"]]["photo_id"] == "photo-2"


def test_redetection_keeps_exclusion_and_does_not_create_a_resurrection_duplicate():
    old_rows = [{"component_id": "excluded-1", "소자 종류": "resistor", "상태": "제외",
                 "x1": 10, "y1": 20, "x2": 30, "y2": 50, "user_edited": True,
                 "source": "yolo", "exclusion_reason": "qwen_not_component"}]
    detections = [
        {"class_name": "resistor", "confidence": .91, "bbox_xyxy": [10, 20, 30, 50]},
        {"class_name": "resistor", "confidence": .89, "bbox_xyxy": [10, 20, 30, 50]},
    ]
    from circuitcare.vision.box_reconciliation import preserve_review_rows
    rows = preserve_review_rows(old_rows, detections)
    assert len(rows) == 1
    assert rows[0]["component_id"] == "excluded-1"
    assert rows[0]["상태"] == "제외"
    assert board_visible_review_rows(rows) == []


def test_all_exclusion_sources_share_undo_snapshot_and_preserve_reason():
    row = {"component_id": "c1", "소자 종류": "resistor", "상태": "확인",
           "x1": 1, "y1": 2, "x2": 10, "y2": 12}
    state = {}
    record_box_exclusion(state, scope="scope", row=row, reason="qwen_duplicate_review",
                         review={"decision": "same_component"}, duplicate_of="c0")
    assert row["상태"] == "제외" and row["duplicate_of"] == "c0"
    assert undo_last_box_exclusion(state, scope="scope", rows=[row]) == "c1"
    assert row["상태"] == "확인" and "duplicate_of" not in row


def test_hover_identity_status_is_user_facing_korean_and_unknown_is_preserved():
    assert display_identity_status("SUGGESTED_UNCONFIRMED") == "미확인 후보"
    assert display_identity_status("IDENTITY_MATCHED") == "동일 부품 후보"
    assert display_identity_status("IDENTITY_AMBIGUOUS") == "식별 모호"
    assert display_identity_status("FUTURE_STATUS") == "FUTURE_STATUS"


def test_coordinate_status_requires_client_mount_and_valid_frame():
    assert coordinate_status_for_viewer(api_available=True, client_ready=False) == "UNVERIFIED_RESPONSIVE_CSS"
    assert coordinate_status_for_viewer(api_available=True, client_ready=True) == "VERIFIED_SHARED_SVG_VIEWBOX"
    assert coordinate_status_for_viewer(api_available=False, client_ready=False) == "VIEWER_UNAVAILABLE"
    assert coordinate_status_for_viewer(api_available=True, client_ready=True, frame_valid=False) == "INVALID_COORDINATE_FRAME"
