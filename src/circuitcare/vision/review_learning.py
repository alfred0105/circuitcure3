"""Private store and isolated training workflow for user-reviewed PCB boxes."""
from __future__ import annotations

import hashlib
import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from circuitcare.component_taxonomy import COMPONENT_CLASSES, normalize_component_class

ROOT = Path(__file__).resolve().parents[3]
USER_DATA_HOME = Path(os.environ.get("CIRCUITCARE_REVIEW_LEARNING_DATA") or
                      os.environ.get("LOCALAPPDATA") or
                      os.environ.get("XDG_DATA_HOME") or
                      (Path.home() / ".local" / "share"))
PRIVATE_ROOT = USER_DATA_HOME / "circuitcare-ai" / "review_learning"
DB_PATH = PRIVATE_ROOT / "review_learning.sqlite3"
MIN_UNIQUE_IMAGES = 50
_SAMPLE_WRITER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pcb-reviewed-sample")
_PENDING_SAVES: list[Future] = []
_SAVE_ERRORS: list[str] = []


def _db() -> sqlite3.Connection:
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS reviewed_images (
            image_sha256 TEXT PRIMARY KEY, project_id TEXT NOT NULL, photo_id TEXT NOT NULL,
            image_path TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reviewed_boxes (
            image_sha256 TEXT NOT NULL, component_id TEXT NOT NULL, class_name TEXT NOT NULL,
            x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
            PRIMARY KEY (image_sha256, component_id),
            FOREIGN KEY (image_sha256) REFERENCES reviewed_images(image_sha256)
        );
        CREATE TABLE IF NOT EXISTS component_knowledge (
            knowledge_id INTEGER PRIMARY KEY AUTOINCREMENT, component_type TEXT NOT NULL,
            marking_key TEXT NOT NULL, part_number TEXT NOT NULL, subtype TEXT NOT NULL,
            functional_role TEXT NOT NULL, notes TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE(component_type, marking_key, part_number)
        );
        CREATE TABLE IF NOT EXISTS training_jobs (
            job_id TEXT PRIMARY KEY, dataset_sha256 TEXT NOT NULL, status TEXT NOT NULL,
            pid INTEGER, output_dir TEXT NOT NULL, log_path TEXT NOT NULL,
            started_at TEXT NOT NULL, finished_at TEXT, error TEXT
        );
        CREATE INDEX IF NOT EXISTS training_jobs_dataset_status
            ON training_jobs(dataset_sha256, status);
        CREATE UNIQUE INDEX IF NOT EXISTS training_jobs_one_active_dataset
            ON training_jobs(dataset_sha256) WHERE status IN ('QUEUED','RUNNING');
        CREATE UNIQUE INDEX IF NOT EXISTS training_jobs_one_successful_dataset
            ON training_jobs(dataset_sha256) WHERE status='SUCCEEDED';
    """)
    return connection


def _pixel_fingerprint(image_bytes: bytes) -> tuple[str, int, int]:
    from PIL import Image
    import io

    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        digest = hashlib.sha256(f"{width}x{height}:RGB:".encode() + rgb.tobytes()).hexdigest()
    return digest, width, height


def save_reviewed_boxes(image_bytes: bytes, *, project_id: str, photo_id: str,
                        rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Upsert explicitly user-labeled boxes; duplicate pixels count as one image."""
    fingerprint, width, height = _pixel_fingerprint(image_bytes)
    image_dir = PRIVATE_ROOT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{fingerprint}.png"
    if not image_path.exists():
        from PIL import Image
        import io

        with Image.open(io.BytesIO(image_bytes)) as source:
            source.convert("RGB").save(image_path, format="PNG", optimize=True)
    now = datetime.now(timezone.utc).isoformat()
    annotations = []
    for row in rows:
        kind = normalize_component_class(row.get("confirmed_base_class") or row.get("effective_class")
                                         or row.get("소자 종류") or "unknown", allow_not_component=False)
        explicitly_labeled = row.get("training_label_user_confirmed") is True
        if (row.get("상태") == "제외" or kind not in COMPONENT_CLASSES or not explicitly_labeled):
            continue
        try:
            x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            continue
        x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
        y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        annotations.append((fingerprint, str(row.get("component_id") or uuid.uuid4().hex),
                            kind, x1, y1, x2, y2))
    with _db() as db:
        db.execute("INSERT INTO reviewed_images VALUES(?,?,?,?,?,?,?,?) "
                   "ON CONFLICT(image_sha256) DO UPDATE SET project_id=excluded.project_id, "
                   "photo_id=excluded.photo_id, updated_at=excluded.updated_at",
                   (fingerprint, str(project_id), str(photo_id), str(image_path), width, height, now, now))
        # A reviewed image is a current snapshot: remove labels that were removed or excluded.
        db.execute("DELETE FROM reviewed_boxes WHERE image_sha256=?", (fingerprint,))
        db.executemany("INSERT INTO reviewed_boxes VALUES(?,?,?,?,?,?,?)", annotations)
        unique_images = int(db.execute("SELECT COUNT(*) FROM reviewed_images WHERE EXISTS "
                                       "(SELECT 1 FROM reviewed_boxes WHERE reviewed_boxes.image_sha256="
                                       "reviewed_images.image_sha256)").fetchone()[0])
        box_count = int(db.execute("SELECT COUNT(*) FROM reviewed_boxes").fetchone()[0])
    return {"image_sha256": fingerprint, "unique_images": unique_images,
            "box_count": box_count, "added_boxes": len(annotations)}


def queue_reviewed_boxes(image_bytes: bytes, *, project_id: str, photo_id: str,
                         rows: list[dict[str, Any]]) -> None:
    """Persist a copy of a reviewed photo asynchronously and without UI state."""
    for future in list(_PENDING_SAVES):
        if future.done():
            _PENDING_SAVES.remove(future)
            try:
                future.result()
            except Exception as exc:
                _SAVE_ERRORS.append(f"{type(exc).__name__}: {exc}")
    _PENDING_SAVES.append(_SAMPLE_WRITER.submit(
        save_reviewed_boxes, bytes(image_bytes), project_id=str(project_id),
        photo_id=str(photo_id), rows=json.loads(json.dumps(rows, default=str)),
    ))


def flush_review_samples() -> list[str]:
    """Wait for queued saves only when the user opens the separate training tab."""
    errors = list(_SAVE_ERRORS)
    _SAVE_ERRORS.clear()
    while _PENDING_SAVES:
        future = _PENDING_SAVES.pop(0)
        try:
            future.result()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return errors


def save_component_knowledge(*, component_type: str, marking: str, part_number: str,
                             subtype: str, functional_role: str, notes: str) -> None:
    kind = normalize_component_class(component_type, allow_not_component=False)
    if kind not in COMPONENT_CLASSES:
        raise ValueError("기존 18종 중 하나로 소자 종류를 지정해야 합니다.")
    marking = " ".join(str(marking or "").split())[:100]
    part_number = " ".join(str(part_number or "").split())[:100]
    marking_key = "".join(character for character in (marking or part_number).upper() if character.isalnum())
    if not marking_key:
        raise ValueError("다음 분류에서 찾을 각인 또는 부품번호를 입력해 주세요.")
    with _db() as db:
        db.execute("INSERT INTO component_knowledge(component_type,marking_key,part_number,subtype,"
                   "functional_role,notes,updated_at) VALUES(?,?,?,?,?,?,?) "
                   "ON CONFLICT(component_type,marking_key,part_number) DO UPDATE SET subtype=excluded.subtype, "
                   "functional_role=excluded.functional_role,notes=excluded.notes,updated_at=excluded.updated_at",
                   (kind, marking_key, part_number, str(subtype or "")[:100],
                    str(functional_role or "")[:120], str(notes or "")[:500],
                    datetime.now(timezone.utc).isoformat()))


def matching_component_knowledge(markings: list[str], component_type: str = "") -> list[dict[str, str]]:
    values = []
    for value in markings:
        if isinstance(value, dict):
            values.extend(value.get(key) for key in ("text", "value", "marking", "candidate") if value.get(key))
        elif value:
            values.append(value)
    keys = {"".join(character for character in str(value).upper() if character.isalnum())
            for value in values if value}
    keys.discard("")
    with _db() as db:
        rows = db.execute("SELECT component_type, marking_key, part_number, subtype, functional_role, notes "
                          "FROM component_knowledge ORDER BY updated_at DESC").fetchall()
    exact = [dict(row) for row in rows if row["marking_key"] in keys
             or "".join(character for character in str(row["part_number"]).upper()
                        if character.isalnum()) in keys]
    if exact:
        return exact[:8]
    normalized_type = normalize_component_class(component_type, allow_not_component=False)
    return [dict(row) for row in rows if row["component_type"] == normalized_type][:4]


def _dataset_snapshot() -> tuple[str, int, list[sqlite3.Row], list[sqlite3.Row]]:
    with _db() as db:
        images = db.execute("SELECT * FROM reviewed_images WHERE EXISTS (SELECT 1 FROM reviewed_boxes "
                            "WHERE reviewed_boxes.image_sha256=reviewed_images.image_sha256) "
                            "ORDER BY image_sha256").fetchall()
        boxes = db.execute("SELECT * FROM reviewed_boxes ORDER BY image_sha256,component_id").fetchall()
    digest = hashlib.sha256()
    for image in images:
        digest.update(image["image_sha256"].encode())
    for box in boxes:
        digest.update(json.dumps(dict(box), sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest(), len(images), images, boxes


def training_summary() -> dict[str, Any]:
    digest, image_count, _, boxes = _dataset_snapshot()
    with _db() as db:
        last = db.execute("SELECT * FROM training_jobs ORDER BY started_at DESC LIMIT 1").fetchone()
        same = db.execute("SELECT * FROM training_jobs WHERE dataset_sha256=? AND status='SUCCEEDED' "
                          "ORDER BY started_at DESC LIMIT 1", (digest,)).fetchone()
        active = db.execute("SELECT * FROM training_jobs WHERE dataset_sha256=? AND status IN "
                            "('QUEUED','RUNNING') ORDER BY started_at DESC LIMIT 1", (digest,)).fetchone()
    return {"dataset_sha256": digest, "unique_images": image_count, "box_count": len(boxes),
            "last_job": dict(last) if last else None,
            "already_trained": dict(same) if same else None,
            "active_job": dict(active) if active else None}


def _export_yolo_dataset(images: list[sqlite3.Row], boxes: list[sqlite3.Row], dataset_sha256: str) -> Path:
    import random

    output = PRIVATE_ROOT / "datasets" / dataset_sha256[:16]
    if (output / "data.yaml").is_file():
        return output / "data.yaml"
    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    image_by_hash = {row["image_sha256"]: row for row in images}
    box_by_image: dict[str, list[sqlite3.Row]] = {}
    for box in boxes:
        box_by_image.setdefault(box["image_sha256"], []).append(box)
    image_hashes = [row["image_sha256"] for row in images]
    random.Random(21).shuffle(image_hashes)
    val_count = max(1, round(len(image_hashes) * 0.2))
    validation = set(image_hashes[:val_count])
    for image_hash in image_hashes:
        row = image_by_hash[image_hash]
        split = "val" if image_hash in validation else "train"
        source = Path(row["image_path"])
        target_name = f"{image_hash}.png"
        (output / "images" / split / target_name).write_bytes(source.read_bytes())
        width, height = int(row["width"]), int(row["height"])
        label_lines = []
        for box in box_by_image.get(image_hash, []):
            class_id = COMPONENT_CLASSES.index(box["class_name"])
            center_x = (box["x1"] + box["x2"]) / (2 * width)
            center_y = (box["y1"] + box["y2"]) / (2 * height)
            box_width, box_height = (box["x2"] - box["x1"]) / width, (box["y2"] - box["y1"]) / height
            label_lines.append(f"{class_id} {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}")
        (output / "labels" / split / f"{image_hash}.txt").write_text("\n".join(label_lines) + "\n")
    yaml_text = f"path: {output.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
    yaml_text += "".join(f"  {index}: {name}\n" for index, name in enumerate(COMPONENT_CLASSES))
    (output / "data.yaml").write_text(yaml_text)
    return output / "data.yaml"


def start_retraining(*, checkpoint: Path, epochs: int = 50) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"로컬 초기 가중치를 찾을 수 없습니다: {checkpoint}")
    dataset_sha, image_count, images, boxes = _dataset_snapshot()
    if image_count < MIN_UNIQUE_IMAGES:
        raise ValueError(f"재학습에는 중복 제거 후 {MIN_UNIQUE_IMAGES}장 이상 필요합니다. 현재 {image_count}장입니다.")
    with _db() as db:
        if db.execute("SELECT 1 FROM training_jobs WHERE dataset_sha256=? AND status='SUCCEEDED'", (dataset_sha,)).fetchone():
            return {"status": "ALREADY_TRAINED", "dataset_sha256": dataset_sha}
        active = db.execute("SELECT job_id FROM training_jobs WHERE dataset_sha256=? AND status IN "
                            "('QUEUED','RUNNING')", (dataset_sha,)).fetchone()
        if active:
            return {"status": "RUNNING", "job_id": active[0], "dataset_sha256": dataset_sha}
    data_yaml = _export_yolo_dataset(images, boxes, dataset_sha)
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    output_dir = PRIVATE_ROOT / "candidate_weights" / job_id
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "training.log"
    command = [sys.executable, str(ROOT / "scripts" / "component_training.py"), "train",
               "--data", str(data_yaml), "--init-weights", str(checkpoint),
               "--profile", "baseline", "--epochs", str(max(1, min(500, int(epochs)))),
               "--workers", "0", "--batch", "2", "--output-root", str(output_dir / "runs")]
    try:
        with _db() as db:
            db.execute("INSERT INTO training_jobs(job_id,dataset_sha256,status,pid,output_dir,log_path,started_at) "
                       "VALUES(?,?,?,?,?,?,?)", (job_id, dataset_sha, "QUEUED", None,
                                                  str(output_dir), str(log_path),
                                                  datetime.now(timezone.utc).isoformat()))
    except sqlite3.IntegrityError:
        with _db() as db:
            active = db.execute("SELECT job_id FROM training_jobs WHERE dataset_sha256=? AND status IN "
                                "('QUEUED','RUNNING')", (dataset_sha,)).fetchone()
            success = db.execute("SELECT job_id FROM training_jobs WHERE dataset_sha256=? AND status='SUCCEEDED'",
                                 (dataset_sha,)).fetchone()
        if active:
            return {"status": "RUNNING", "job_id": active[0], "dataset_sha256": dataset_sha}
        if success:
            return {"status": "ALREADY_TRAINED", "dataset_sha256": dataset_sha}
        raise
    worker_env = os.environ.copy()
    worker_env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + worker_env.get("PYTHONPATH", "")
    try:
        process = subprocess.Popen([sys.executable, "-m", "circuitcare.vision.review_learning",
                                    "_worker", job_id, *command[1:]], cwd=ROOT,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=True, env=worker_env)
    except OSError as exc:
        with _db() as db:
            db.execute("UPDATE training_jobs SET status='FAILED',finished_at=?,error=? WHERE job_id=?",
                       (datetime.now(timezone.utc).isoformat(), str(exc), job_id))
        raise
    with _db() as db:
        db.execute("UPDATE training_jobs SET status='RUNNING',pid=? WHERE job_id=?",
                   (process.pid, job_id))
    return {"status": "RUNNING", "job_id": job_id, "dataset_sha256": dataset_sha,
            "pid": process.pid, "output_dir": str(output_dir), "log_path": str(log_path)}


def refresh_training_jobs() -> None:
    with _db() as db:
        jobs = db.execute("SELECT * FROM training_jobs WHERE status='RUNNING'").fetchall()
        for job in jobs:
            try:
                os.kill(int(job["pid"]), 0)
                continue
            except (OSError, TypeError, ValueError):
                pass
            log = Path(job["log_path"])
            text = log.read_text(errors="replace") if log.exists() else ""
            if job["status"] in {"RUNNING", "QUEUED"}:
                db.execute("UPDATE training_jobs SET status='FAILED',finished_at=?,error=? WHERE job_id=?",
                           (datetime.now(timezone.utc).isoformat(), text[-1200:] or "Training worker ended unexpectedly.", job["job_id"]))


def _run_training_worker(job_id: str, command: list[str]) -> int:
    with _db() as db:
        job = db.execute("SELECT * FROM training_jobs WHERE job_id=?", (job_id,)).fetchone()
    if not job:
        return 2
    output_dir, log_path = Path(job["output_dir"]), Path(job["log_path"])
    try:
        from ultralytics import YOLO
        model = YOLO(command[command.index("--init-weights") + 1])
        names = model.names
        ordered_names = [names[index] for index in sorted(names)] if isinstance(names, dict) else list(names)
        normalized_names = [normalize_component_class(name, allow_not_component=False) for name in ordered_names]
        missing_classes = set(COMPONENT_CLASSES) - set(normalized_names)
        if missing_classes:
            raise ValueError("초기 가중치에 기존 YOLO 분류 체계와 공통인 클래스가 부족합니다: "
                             + ", ".join(sorted(missing_classes)))
        if len(normalized_names) == len(COMPONENT_CLASSES) and tuple(normalized_names) != COMPONENT_CLASSES:
            raise ValueError("초기 가중치와 현재 18종 분류의 클래스 순서가 다릅니다. "
                             "같은 크기의 출력층을 다른 클래스 의미로 재사용하지 않도록 학습을 중단했습니다.")
        with log_path.open("wb") as log:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env=os.environ.copy(), check=False)
        record_line = next((line for line in log_path.read_text(errors="replace").splitlines()
                            if line.startswith("Experiment record: ")), None)
        if result.returncode != 0 or not record_line:
            raise RuntimeError(f"학습 프로세스가 실패했습니다 (exit={result.returncode}).")
        experiment = Path(record_line.split(": ", 1)[1])
        best = experiment.parent / "weights" / "best.pt"
        if not best.is_file():
            raise RuntimeError("학습은 종료했지만 후보 best.pt를 만들지 못했습니다.")
        candidate = output_dir / "candidate_best.pt"
        candidate.write_bytes(best.read_bytes())
        metadata = {"job_id": job_id, "dataset_sha256": job["dataset_sha256"],
                    "source_checkpoint": command[command.index("--init-weights") + 1],
                    "source_experiment": str(experiment), "candidate_weights": str(candidate),
                    "class_names": list(COMPONENT_CLASSES), "source_class_names": ordered_names,
                    "class_head_reinitialized": len(normalized_names) != len(COMPONENT_CLASSES),
                    "created_at": datetime.now(timezone.utc).isoformat()}
        (output_dir / "candidate.json").write_text(json.dumps(metadata, indent=2))
        with _db() as db:
            db.execute("UPDATE training_jobs SET status='SUCCEEDED',finished_at=?,error=NULL WHERE job_id=?",
                       (datetime.now(timezone.utc).isoformat(), job_id))
        return 0
    except Exception as exc:
        with log_path.open("a") as log:
            log.write(f"\n{type(exc).__name__}: {exc}\n")
        with _db() as db:
            db.execute("UPDATE training_jobs SET status='FAILED',finished_at=?,error=? WHERE job_id=?",
                       (datetime.now(timezone.utc).isoformat(), f"{type(exc).__name__}: {exc}", job_id))
        return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("_worker",))
    parser.add_argument("job_id")
    parser.add_argument("train_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    raise SystemExit(_run_training_worker(args.job_id, args.train_command))


if __name__ == "__main__":
    main()
