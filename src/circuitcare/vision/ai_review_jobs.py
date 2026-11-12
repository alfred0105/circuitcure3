"""Bounded, session-isolated background execution for PCB AI review.

Workers receive immutable input snapshots and never call Streamlit APIs. Results
are read and committed by the owning Streamlit session after fingerprint checks.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import threading
import time
from typing import Any


@dataclass
class _Job:
    owner: str
    job_id: str
    scope: str
    fingerprint: str
    future: Future
    started_at: float = field(default_factory=time.perf_counter)


class _Progress:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def write(self, message: str) -> None:
        self.messages.append(str(message)[:180])
        del self.messages[:-8]

    def update(self, **_kwargs: Any) -> None:
        return None


class _WorkerAPI:
    """Small Streamlit-free adapter for the existing review workflow."""
    def __init__(self, state: dict[str, Any]) -> None:
        self.session_state = state

    def spinner(self, *_args: Any, **_kwargs: Any) -> Any:
        from contextlib import nullcontext
        return nullcontext()

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def status(self, *_args: Any, **_kwargs: Any) -> _Progress:
        return _Progress()


_LOCK = threading.RLock()
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="circuitcare-ai-review")
_JOBS: dict[tuple[str, str], _Job] = {}
_MAX_JOBS = 32
_MAX_PENDING = 2


def _run(snapshot: dict[str, Any]) -> dict[str, Any]:
    from circuitcare.vision.component_recognition_panel import ensure_ai_component_review

    started = time.perf_counter()
    state = deepcopy(snapshot["state"])
    progress = _Progress()
    api = _WorkerAPI(state)
    try:
        result = ensure_ai_component_review(
            api, image_bytes=snapshot["image_bytes"], scope=snapshot["scope"],
            rows=deepcopy(snapshot["rows"]), records_key=snapshot["records_key"],
            progress=progress,
        )
        return {"status": result.upper(), "state": state, "messages": progress.messages,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "fingerprint": snapshot["fingerprint"], "scope": snapshot["scope"]}
    except Exception as exc:  # Keep worker failures isolated from Streamlit's script thread.
        return {"status": "FAILED", "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                "messages": progress.messages,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "fingerprint": snapshot["fingerprint"], "scope": snapshot["scope"]}


def submit_ai_review(*, owner: str, scope: str, fingerprint: str, image_bytes: bytes,
                     rows: list[dict[str, Any]], records_key: str,
                     state_snapshot: dict[str, Any]) -> tuple[str | None, str]:
    """Submit or reuse one review job; never queue an unbounded model backlog."""
    job_id = hashlib.sha256(f"{owner}|{scope}|{fingerprint}".encode()).hexdigest()[:24]
    key = (owner, job_id)
    with _LOCK:
        existing = _JOBS.get(key)
        if existing:
            if existing.future.done():
                return job_id, "COMPLETE"
            return job_id, "RUNNING"
        active = [job for job in _JOBS.values() if not job.future.done()]
        if len(active) >= _MAX_PENDING + 1:
            return None, "QUEUE_FULL"
        snapshot = {"state": deepcopy(state_snapshot), "image_bytes": bytes(image_bytes),
                    "scope": scope, "rows": deepcopy(rows), "records_key": records_key,
                    "fingerprint": fingerprint}
        future = _EXECUTOR.submit(_run, snapshot)
        _JOBS[key] = _Job(owner, job_id, scope, fingerprint, future)
        while len(_JOBS) > _MAX_JOBS:
            oldest = next(iter(_JOBS))
            if not _JOBS[oldest].future.done():
                break
            _JOBS.pop(oldest, None)
        return job_id, "PENDING"


def get_ai_review_job(*, owner: str, job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get((owner, job_id))
        if not job:
            return None
        if not job.future.done():
            return {"status": "RUNNING", "scope": job.scope, "fingerprint": job.fingerprint,
                    "elapsed_ms": round((time.perf_counter() - job.started_at) * 1000, 2)}
        try:
            return job.future.result()
        except Exception as exc:
            return {"status": "FAILED", "scope": job.scope, "fingerprint": job.fingerprint,
                    "error": f"{type(exc).__name__}: {str(exc)[:400]}"}

