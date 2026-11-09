"""Lightweight checks for the independent training helpers."""
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("component_training", ROOT / "scripts" / "component_training.py")
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


def test_device_selection_requires_successful_mps_probe():
    assert training.select_device({"probe_ok": True}) == "mps"
    assert training.select_device({"probe_ok": False, "error": "unsupported op"}) == "cpu"
    assert training.select_device({}) == "cpu"


def test_sha256_streaming_hash(tmp_path):
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"component-training")
    assert training._hash(sample) == "ae6e8b46d64f6c2f7c7c2131c06a323e01f8fb4b11d023e0da069ce485df5dcf"
