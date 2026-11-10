from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from circuitcare.frontend.state import INTAKE_PROJECT_KEY


def _photo(color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (48, 32), color).save(stream, format="PNG")
    return stream.getvalue()


def test_photo_upload_enters_current_pcb_workbench():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "src/circuitcare/ui.py"), default_timeout=30).run()
    app.file_uploader[0].set_value(("board.png", _photo((20, 90, 140)), "image/png")).run()
    next(widget for widget in app.button if widget.label == "진단 시작").click().run()

    assert not app.exception
    assert app.session_state[INTAKE_PROJECT_KEY].image_records
    assert app.session_state["circuitcare_v2_screen"] == "workbench"
    assert any("소자 탐지" in item.label for item in app.expander)
    assert not any(widget.label == "PCB 사진 인식" for widget in app.button)


@pytest.mark.stress
def test_twenty_photo_project_transitions_keep_app_live():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "src/circuitcare/ui.py"), default_timeout=30).run()
    for index in range(20):
        if INTAKE_PROJECT_KEY in app.session_state:
            next(widget for widget in app.button if widget.label == "새 프로젝트").click().run()
        photo = _photo((index * 9 % 255, 60, 120))
        app.file_uploader[0].set_value((f"board-{index}.png", photo, "image/png")).run()
        next(widget for widget in app.button if widget.label == "진단 시작").click().run()
        assert not app.exception
        assert app.session_state[INTAKE_PROJECT_KEY].image_records
        assert app.session_state["circuitcare_v2_screen"] == "workbench"
