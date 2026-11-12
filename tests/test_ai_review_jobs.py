from __future__ import annotations

from threading import current_thread

from circuitcare.vision import ai_review_jobs


def test_ai_review_worker_uses_isolated_state_adapter_and_returns_result(monkeypatch):
    from circuitcare.vision import component_recognition_panel

    observed = {}

    def fake_review(api, *, image_bytes, scope, rows, records_key, progress):
        observed["thread"] = current_thread().name
        observed["api_type"] = type(api).__name__
        observed["image_bytes"] = image_bytes
        observed["scope"] = scope
        api.session_state[records_key] = [{**rows[0], "ocr_candidates": ["R1"]}]
        progress.write("review complete")
        return "complete"

    monkeypatch.setattr(component_recognition_panel, "ensure_ai_component_review", fake_review)
    result = ai_review_jobs._run({
        "state": {"cc_rows": [{"component_id": "c1"}]}, "image_bytes": b"image",
        "scope": "scope", "rows": [{"component_id": "c1"}],
        "records_key": "cc_rows", "fingerprint": "fp",
    })
    assert observed == {"thread": current_thread().name, "api_type": "_WorkerAPI",
                        "image_bytes": b"image", "scope": "scope"}
    assert result["status"] == "COMPLETE"
    assert result["state"]["cc_rows"][0]["ocr_candidates"] == ["R1"]
    assert result["messages"] == ["review complete"]
    assert result["fingerprint"] == "fp"


def test_ai_review_job_results_are_owner_scoped_and_duplicate_jobs_reused(monkeypatch):
    from concurrent.futures import Future

    class _ImmediateExecutor:
        def __init__(self):
            self.calls = 0

        def submit(self, fn, snapshot):
            self.calls += 1
            future = Future()
            future.set_result({"status": "COMPLETE", "scope": snapshot["scope"],
                               "fingerprint": snapshot["fingerprint"]})
            return future

    executor = _ImmediateExecutor()
    monkeypatch.setattr(ai_review_jobs, "_EXECUTOR", executor)
    monkeypatch.setattr(ai_review_jobs, "_JOBS", {})
    args = {"scope": "photo-a", "fingerprint": "f1", "image_bytes": b"x",
            "rows": [], "records_key": "rows", "state_snapshot": {}}
    job_id, status = ai_review_jobs.submit_ai_review(owner="session-a", **args)
    same_job, reused_status = ai_review_jobs.submit_ai_review(owner="session-a", **args)
    assert status == "PENDING"
    assert reused_status == "COMPLETE"
    assert job_id == same_job
    assert executor.calls == 1
    assert ai_review_jobs.get_ai_review_job(owner="session-b", job_id=job_id) is None
    assert ai_review_jobs.get_ai_review_job(owner="session-a", job_id=job_id)["fingerprint"] == "f1"


def test_review_progress_tracks_stage_and_batch_counts_without_streamlit(monkeypatch):
    progress = ai_review_jobs._Progress()
    progress.write("PCB OCR · 완료 · 8개 문자 후보")
    assert progress.snapshot()["stage"] == "OCR_BOARD"
    progress.write("Qwen 소자 검수 · 3/10 배치 진행 중")
    state = progress.snapshot()
    assert state["stage"] == "QWEN_CLASS_REVIEW"
    assert state["current_batch"] == 3
    assert state["completed_batches"] == 2
    assert state["total_batches"] == 10


def test_running_job_exposes_thread_safe_progress_snapshot(monkeypatch):
    from concurrent.futures import Future

    class _DelayedExecutor:
        def submit(self, fn, snapshot):
            self.future = Future()
            self.snapshot = snapshot
            return self.future

    executor = _DelayedExecutor()
    monkeypatch.setattr(ai_review_jobs, "_EXECUTOR", executor)
    monkeypatch.setattr(ai_review_jobs, "_JOBS", {})
    job_id, _status = ai_review_jobs.submit_ai_review(
        owner="owner", scope="scope", fingerprint="f", image_bytes=b"x",
        rows=[], records_key="rows", state_snapshot={},
    )
    job = ai_review_jobs._JOBS[("owner", job_id)]
    job.progress.set_stage("OCR_MARKINGS", "소자 각인 OCR 처리 중")
    running = ai_review_jobs.get_ai_review_job(owner="owner", job_id=job_id)
    assert running["status"] == "RUNNING"
    assert running["progress"]["stage"] == "OCR_MARKINGS"
    assert running["progress"]["message"] == "소자 각인 OCR 처리 중"
