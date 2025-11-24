from voicecred.speaker import MockSpeakerRecognitionAdapter, create_speaker_adapter


def test_mock_speaker_adapter_basic():
    sp = MockSpeakerRecognitionAdapter(speakers=["A", "B"])
    res = sp.recognize({"words": [{"word": "a", "start_ms": 0, "end_ms": 100}, {"word": "b", "start_ms": 100, "end_ms": 200}]})
    assert isinstance(res, dict)
    assert "segments" in res


def test_create_speaker_factory():
    sp = create_speaker_adapter("mock", speakers=["X", "Y"])
    assert isinstance(sp, MockSpeakerRecognitionAdapter)
