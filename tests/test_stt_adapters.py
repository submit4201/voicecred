import pytest

from voicecred.stt import create_stt_adapter, MockSTTAdapter


def test_create_adapter_factory_defaults_to_mock():
    a = create_stt_adapter(None)
    assert isinstance(a, MockSTTAdapter)


def test_mock_stt_transcribe_override():
    adapter = MockSTTAdapter()
    frames = [{"timestamp_ms": 0, "transcript_override": "this is test"}]
    res = adapter.transcribe(frames)
    assert isinstance(res, dict)
    assert "words" in res and isinstance(res["words"], list)
    assert res["raw"] == "this is test"
