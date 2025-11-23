import pytest
import os
from voicecred.stt import MockSpeakerRecognitionAdapter, PyannoteSpeakerAdapter
from voicecred.stt import MockSTTAdapter
from concurrent.futures import ThreadPoolExecutor


def test_mock_speaker_adapter_basic():
    adapter = MockSpeakerRecognitionAdapter(speakers=["A","B"])
    asr = {"words": [{"word": "hello", "start_ms": 0, "end_ms": 200}, {"word": "world", "start_ms": 200, "end_ms": 400}]}
    res = adapter.recognize(asr)
    assert isinstance(res, dict)
    assert "segments" in res
    assert len(res["segments"]) == 2
    assert res["segments"][0]["speaker"] in ("A","B")


def test_parallel_mock_stt_and_speaker():
    stt = MockSTTAdapter(default_confidence=0.9)
    spk = MockSpeakerRecognitionAdapter(speakers=["X","Y"])
    frames = [{"timestamp_ms": 0, "pcm": b"dummy1"}, {"timestamp_ms": 200, "pcm": b"dummy2"}]

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(stt.transcribe, frames)
        # speaker recognize doesn't require ASR result for mock adapter, it can accept frames
        f2 = ex.submit(spk.recognize, {"words": [{"word":"a","start_ms":0,"end_ms":100}]})
        asr = f1.result()
        spk_res = f2.result()

    assert isinstance(asr, dict) and "words" in asr
    assert isinstance(spk_res, dict) and "segments" in spk_res


RUN_HEAVY_PYANNOTE = os.environ.get("RUN_HEAVY_PYANNOTE", "0").lower() in ("1", "true", "yes")

@pytest.mark.skipif(not RUN_HEAVY_PYANNOTE, reason="Skip heavy pyannote integration in CI unless RUN_HEAVY_PYANNOTE is enabled")
def test_pyannote_adapter_skip():
    # placeholder to indicate we intentionally skip heavy tests by default
    pass
