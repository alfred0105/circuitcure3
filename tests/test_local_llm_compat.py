from __future__ import annotations

import pytest

from circuitcare.local_llm import LocalLLMError, OllamaClient


def _response(content: str = '{"ok":true}') -> dict:
    return {"model": "test:latest", "message": {"content": content}, "done": True}


def test_chat_json_falls_back_only_for_unsupported_structured_output(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(client, "model_info", lambda _name: {"name": "test:latest", "digest": "abc"})
    calls = []

    def request(_method, _path, payload):
        calls.append(payload.copy())
        if len(calls) == 1:
            raise LocalLLMError('Ollama request failed with HTTP 501: {"error":"structured output is unavailable"}')
        return _response()

    monkeypatch.setattr(client, "_request", request)
    result = client.chat_json("test:latest", [{"role": "user", "content": "JSON"}])
    assert result["content"] == {"ok": True}
    assert result["json_mode"] is False
    assert result["fallback_used"] is True
    assert result["fallback_reason"] == "structured_output_unavailable"
    assert calls[0]["format"] == "json" and "format" not in calls[1]
    assert calls[1]["think"] is False


@pytest.mark.parametrize("error", [
    'Ollama request failed with HTTP 501: {"error":"route unavailable"}',
    'Could not reach local Ollama at http://127.0.0.1:11434: connection refused',
    'Ollama request failed with HTTP 404: model not found',
])
def test_chat_json_does_not_fallback_for_other_errors(monkeypatch, error):
    client = OllamaClient()
    monkeypatch.setattr(client, "model_info", lambda _name: {"name": "test:latest"})
    calls = []

    def request(*_args):
        calls.append(1)
        raise LocalLLMError(error)

    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(LocalLLMError):
        client.chat_json("test:latest", [{"role": "user", "content": "JSON"}])
    assert len(calls) == 1


def test_chat_json_rejects_invalid_fallback_json(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(client, "model_info", lambda _name: {"name": "test:latest"})
    calls = iter([LocalLLMError("structured output is unavailable"), _response("not JSON")])
    monkeypatch.setattr(client, "_request", lambda *_args: (_raise(next(calls))))
    with pytest.raises(LocalLLMError, match="invalid JSON"):
        client.chat_json("test:latest", [{"role": "user", "content": "JSON"}])


@pytest.mark.parametrize("content", [
    '```json\n{"ok":true}\n```',
    '```json\n{"ok":true}',
    'Result:\n```json\n{"ok":true}\n```',
])
def test_chat_json_recovers_complete_json_from_fenced_model_output(monkeypatch, content):
    client = OllamaClient()
    monkeypatch.setattr(client, "model_info", lambda _name: {"name": "test:latest"})
    monkeypatch.setattr(client, "_request", lambda *_args: _response(content))
    assert client.chat_json("test:latest", [{"role": "user", "content": "JSON"}])["content"] == {"ok": True}


def test_chat_json_uses_configured_output_token_budget_and_reports_truncation(monkeypatch):
    client = OllamaClient(num_predict=4096)
    monkeypatch.setattr(client, "model_info", lambda _name: {"name": "test:latest"})
    calls = []

    def request(_method, _path, payload):
        calls.append(payload)
        return {**_response('```json\n{"incomplete":'), "done_reason": "length"}

    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(LocalLLMError, match="truncated at the output token limit"):
        client.chat_json("test:latest", [{"role": "user", "content": "JSON"}])
    assert calls[0]["options"]["num_predict"] == 4096


def _raise(value):
    if isinstance(value, Exception):
        raise value
    return value
