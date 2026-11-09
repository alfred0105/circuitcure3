import hashlib
import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("acquire_component_dataset", ROOT / "scripts" / "acquire_component_dataset.py")
acquire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquire)


class FakeResponse(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data))}

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def test_download_verifies_sha_and_resumes_partial_file(tmp_path, monkeypatch):
    payload = b"source-data-with-resume"
    target = tmp_path / "data.bin"
    target.with_name("data.bin.part").write_bytes(payload[:7])

    def fake_open(request, timeout):
        assert request.get_header("Range") == "bytes=7-"
        response = FakeResponse(payload[7:], status=206)
        response.headers["Content-Range"] = f"bytes 7-{len(payload)-1}/{len(payload)}"
        return response

    monkeypatch.setattr(acquire.urllib.request, "urlopen", fake_open)
    result = acquire.download_file("https://example.invalid/dataset", target,
        expected_size=len(payload), expected_sha256=hashlib.sha256(payload).hexdigest())
    assert target.read_bytes() == payload
    assert result["status"] == "DOWNLOADED_AND_VERIFIED"


def test_download_keeps_partial_on_interrupted_size(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"short"))
    with pytest.raises(OSError, match="download incomplete"):
        acquire.download_file("https://example.invalid/dataset", tmp_path / "x.bin", expected_size=10)
    assert (tmp_path / "x.bin.part").read_bytes() == b"short"


def test_download_rejects_checksum_mismatch_and_preserves_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"wrong"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        acquire.download_file("https://example.invalid/dataset", tmp_path / "x.bin",
                              expected_size=5, expected_sha256="0" * 64)
    assert (tmp_path / "x.bin.part").read_bytes() == b"wrong"
    assert not (tmp_path / "x.bin").exists()


def test_download_surfaces_url_failure_without_publishing(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError("connection refused")
    monkeypatch.setattr(acquire.urllib.request, "urlopen", fail)
    with pytest.raises(OSError, match="connection refused"):
        acquire.download_file("https://example.invalid/dataset", tmp_path / "x.bin", expected_size=5)
    assert not (tmp_path / "x.bin").exists()


def test_resume_rejects_wrong_content_range(tmp_path, monkeypatch):
    target = tmp_path / "x.bin"
    target.with_name("x.bin.part").write_bytes(b"ab")
    response = FakeResponse(b"cdef", status=206)
    response.headers["Content-Range"] = "bytes 0-3/6"
    monkeypatch.setattr(acquire.urllib.request, "urlopen", lambda *a, **k: response)
    with pytest.raises(OSError, match="invalid Content-Range"):
        acquire.download_file("https://example.invalid/dataset", target, expected_size=6)
    assert target.with_name("x.bin.part").read_bytes() == b"ab"


def test_tar_preflight_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, "w") as archive:
        payload = b"bad"
        info = tarfile.TarInfo("../outside.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe archive path"):
        acquire.inspect_tar(archive_path)


def test_tar_preflight_reports_expanded_size_and_safe_extract(tmp_path):
    archive_path = tmp_path / "safe.tar"
    with tarfile.open(archive_path, "w") as archive:
        payload = b"ok"
        info = tarfile.TarInfo("folder/file.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    assert acquire.inspect_tar(archive_path)["uncompressed_bytes"] == 2
    out = tmp_path / "out"
    acquire.safe_extract_tar(archive_path, out)
    assert (out / "folder/file.txt").read_bytes() == b"ok"


def test_zip_preflight_and_safe_extract(tmp_path):
    archive_path = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("folder/file.txt", "ok")
    assert acquire.inspect_zip(archive_path)["uncompressed_bytes"] == 2
    out = tmp_path / "out"
    acquire.safe_extract_zip(archive_path, out)
    assert (out / "folder/file.txt").read_text() == "ok"


def test_zip_preflight_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="unsafe archive path"):
        acquire.inspect_zip(archive_path)


def test_zip_preflight_rejects_symlink_member(tmp_path):
    archive_path = tmp_path / "link.zip"
    info = zipfile.ZipInfo("shortcut")
    info.create_system = 3
    info.external_attr = (0o120777 << 16)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(info, "../outside")
    with pytest.raises(ValueError, match="unsupported archive member type"):
        acquire.inspect_zip(archive_path)
