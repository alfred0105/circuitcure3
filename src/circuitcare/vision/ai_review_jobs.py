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
import re


@dataclass
class _Job:
    owner: str
    job_id: str
    scope: str
    fingerprint: str
    future: Future
    progress: "_Progress"
    started_at: float = field(default_factory=time.perf_counter)


class _Progress:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.messages: list[str] = []
        self.stage = "PENDING"
        self.message = "AI 검수 대기 중"
        self.completed_batches = 0
        self.total_batches = 0
        self.current_batch = 0
        self.errors: list[str] = []
        self.updated_at = time.time()

    def set_stage(self, stage: str, message: str, *, completed_batches: int | None = None,
                  total_batches: int | None = None, current_batch: int | None = None,
                  error: str | None = None) -> None:
        with self._lock:
            self.stage = str(stage)
            self.message = str(message)[:240]
            if completed_batches is not None:
                self.completed_batches = max(0, int(completed_batches))
            if total_batches is not None:
                self.total_batches = max(0, int(total_batches))
            if current_batch is not None:
                self.current_batch = max(0, int(current_batch))
            if error:
                self.errors.append(str(error)[:400])
                del self.errors[:-8]
            self.updated_at = time.time()

    def write(self, message: str) -> None:
        value = str(message)[:180]
        with self._lock:
            self.messages.append(value)
            del self.messages[:-8]
        lower = value.lower()
        if "실패" in lower or "오류" in lower:
            failure_stage = "QWEN_DUPLICATE_REVIEW" if "중복" in lower else "FAILED"
            self.set_stage(failure_stage, value, error=value)
        elif "pcb ocr" in lower:
            self.set_stage("OCR_BOARD", value)
        elif "각인 ocr" in lower:
            self.set_stage("OCR_MARKINGS", value)
        elif "qwen 소자 검수" in lower and "배치" in lower:
            match = re.search(r"(\d+)\s*/\s*(\d+)", value)
            if match:
                current, total = map(int, match.groups())
                self.set_stage("QWEN_CLASS_REVIEW", value,
                               completed_batches=current - 1,
                               total_batches=total, current_batch=current)
        elif "중복 후보" in lower or "중복 검토" in lower:
            self.set_stage("QWEN_DUPLICATE_REVIEW", value)
        elif "완료" in lower:
            self.set_stage("RESULT_VALIDATION", value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"stage": self.stage, "message": self.message,
                    "completed_batches": self.completed_batches,
                    "total_batches": self.total_batches, "current_batch": self.current_batch,
                    "last_event": self.messages[-1] if self.messages else self.message,
                    "messages": list(self.messages), "errors": list(self.errors),
                    "updated_at": self.updated_at}

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


def _run(snapshot: dict[str, Any], progress: _Progress | None = None) -> dict[str, Any]:
    from circuitcare.vision.component_recognition_panel import ensure_ai_component_review

    started = time.perf_counter()
    state = deepcopy(snapshot["state"])
    progress = progress or _Progress()
    progress.set_stage("OCR_BOARD", "RapidOCR·Qwen 검수 작업 시작")
    api = _WorkerAPI(state)
    try:
        result = ensure_ai_component_review(
            api, image_bytes=snapshot["image_bytes"], scope=snapshot["scope"],
            rows=deepcopy(snapshot["rows"]), records_key=snapshot["records_key"],
            progress=progress,
        )
        final_stage = "FAILED" if result.upper() in {"FAILED", "UNAVAILABLE"} else "COMPLETE"
        progress.set_stage(final_stage, "AI 검수 작업 실패" if final_stage == "FAILED" else "AI 검수 작업 완료")
        return {"status": result.upper(), "state": state, "messages": progress.snapshot()["messages"],
                "progress": progress.snapshot(),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "fingerprint": snapshot["fingerprint"], "scope": snapshot["scope"]}
    except Exception as exc:  # Keep worker failures isolated from Streamlit's script thread.
        error = f"{type(exc).__name__}: {str(exc)[:400]}"
        progress.set_stage("FAILED", error, error=error)
        return {"status": "FAILED", "error": error,
                "messages": progress.snapshot()["messages"], "progress": progress.snapshot(),
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
        progress = _Progress()
        progress.set_stage("PENDING", "AI 검수 대기 중")
        future = _EXECUTOR.submit(lambda queued_snapshot: _run(queued_snapshot, progress), snapshot)
        _JOBS[key] = _Job(owner, job_id, scope, fingerprint, future, progress)
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
            progress = job.progress.snapshot()
            return {"status": "PENDING" if progress.get("stage") == "PENDING" else "RUNNING",
                    "scope": job.scope, "fingerprint": job.fingerprint,
                    "elapsed_ms": round((time.perf_counter() - job.started_at) * 1000, 2),
                    "progress": progress}
        try:
            return job.future.result()
        except Exception as exc:
            return {"status": "FAILED", "scope": job.scope, "fingerprint": job.fingerprint,
                    "error": f"{type(exc).__name__}: {str(exc)[:400]}"}

