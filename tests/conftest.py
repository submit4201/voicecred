import os
import base64
import time
from typing import Any, Dict, Callable
import pytest
from fastapi.testclient import TestClient

# tests/conftest.py


import voicecred.main as main_mod

# Ensure tests use deterministic mock STT adapter unless explicitly overridden
os.environ.setdefault("STT_ADAPTER", "mock")
os.environ.setdefault("VOICECRED_DEBUG", "0")

@pytest.fixture(scope="session")
def app():
    """FastAPI app instance."""
    return main_mod.app

@pytest.fixture
def client(app) -> TestClient:
    """TestClient for the FastAPI app."""
    return TestClient(app)

def _encode_pcm_bytes(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")

@pytest.fixture
def make_pcm_frame() -> Callable[[int, str], Dict[str, Any]]:
    """
    Return a helper to construct a PCM frame dict suitable for WebSocket sends.
    Usage: frame = make_pcm_frame(duration_ms=20, transcript_override="hello")
    """
    def _make(duration_ms: int = 20, transcript_override: str = None) -> Dict[str, Any]:
        # Create a small chunk of silence (16kHz, 16-bit mono -> 2 bytes per sample)
        samples = int(16000 * (duration_ms / 1000.0))
        raw = b"\x00\x00" * samples
        frame = {"pcm": _encode_pcm_bytes(raw)}
        if transcript_override is not None:
            frame["transcript_override"] = transcript_override
        return frame
    return _make

@pytest.fixture
def recv_with_timeout():
    """
    Helper to receive JSON from a TestClient WebSocket with a timeout.
    Usage: msg = recv_with_timeout(ws, timeout=2.0)
    """
    def _recv(ws, timeout: float = 2.0):
        try:
            return ws.receive_json(timeout=timeout)
        except Exception as exc:
            pytest.fail(f"Timed out waiting for websocket message (timeout={timeout}): {exc}")
    return _recv

@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    """
    Ensure certain environment keys are stable across tests and provide a place
    to monkeypatch additional env vars in individual tests.
    """
    monkeypatch.setenv("STT_ADAPTER", os.environ.get("STT_ADAPTER", "mock"))
    monkeypatch.setenv("VOICECRED_DEBUG", os.environ.get("VOICECRED_DEBUG", "0"))
    # Allow tests to override heavy integrations via env flags
    monkeypatch.setenv("RUN_HEAVY_PYANNOTE", os.environ.get("RUN_HEAVY_PYANNOTE", "false"))
    yield