import pytest

from voicecred.stt import MockSTTAdapter
from voicecred.linguistic import LinguisticEngine
from voicecred.stt import create_stt_adapter, WhisperSTTAdapter
import os
import importlib


def test_mock_stt_adapter_basic():
    adapter = MockSTTAdapter(override="this is a test")
    out = adapter.transcribe([])
    print('MockSTTAdapter output:', out)
    assert "words" in out
    assert out["confidence"] == pytest.approx(0.95)
    assert out["raw"].startswith("this is")


def test_linguistic_engine_basic():
    adapter = MockSTTAdapter(override="I had a sandwich and the coffee was great")
    transcript = adapter.transcribe([])
    print('Transcript for linguistic start analysis:', transcript)
    engine = LinguisticEngine()
    print('Transcript for linguistic analysis:', transcript)
    res = engine.analyze(transcript, timestamp_ms=100)

    print('LinguisticEngine analysis end result:', res)
    assert res.linguistic["pronoun_ratio"] > 0
    assert res.asr_quality == pytest.approx(transcript["confidence"])
    assert res.linguistic["ttr"] <= 1.0


def test_linguistic_engine_empty():
    print('Starting test_linguistic_engine_empty')
    adapter = MockSTTAdapter(override="")
    transcript = adapter.transcribe([])
    print('Transcript for linguistic start analysis (empty):', transcript)
    engine = LinguisticEngine()
    print('Transcript for linguistic analysis (empty):', transcript)
    
    res = engine.analyze(transcript)
    print('LinguisticEngine analysis result (empty):', res)
    assert res.linguistic["tokens"] == pytest.approx(1.0)
    assert res.linguistic["ttr"] <= 1.0


def test_create_stt_adapter_factory_default():
    a = create_stt_adapter(None)
    assert isinstance(a, MockSTTAdapter)


def test_create_stt_adapter_factory_whisper():
    a = create_stt_adapter('whisper')
    assert isinstance(a, WhisperSTTAdapter)


def test_main_uses_env_adapter():
    # set env var then reload main to ensure the adapter created by env var is used
    os.environ['STT_ADAPTER'] = 'whisper'
    import voicecred.main as vcmain
    importlib.reload(vcmain)
    # Instance may be loaded from a different import path (src.voicecred.* vs voicecred.*)
    # so assert by type name to avoid module path mismatch issues in test environment.
    assert type(vcmain.stt_adapter).__name__ == 'WhisperSTTAdapter'
    # cleanup
    del os.environ['STT_ADAPTER']
    importlib.reload(vcmain)


def test_whisper_adapter_override_behavior():
    # Whisper adapter should honor transcript_override even when model not available
    a = WhisperSTTAdapter()
    frames = [{"timestamp_ms": 0, "pcm": [], "transcript_override": "this will be returned"}]
    out = a.transcribe(frames)
    assert out["raw"] == "this will be returned"
    assert out["confidence"] == pytest.approx(a.default_confidence)


def test_assembled_feature_contains_linguistic():
    # ensure that the assembler produces derived linguistic features in the feature_frame
    from voicecred.assembler import assemble_feature_frame
    ac = {"acoustic": [0.0, 0.0, 0.0, 0.0, 0.0], "qc": {"snr_db": 0.0}}
    ling = {"pronoun_ratio": 0.1, "article_ratio": 0.05, "ttr": 0.6, "avg_tokens_per_sentence": 3.0, "tokens": 5.0, "avg_word_length": 4.0, "lexical_density": 0.8, "asr_quality": 0.9}
    f = assemble_feature_frame("s1", ac, ling, 123)
    assert isinstance(f.get("derived"), list)
    assert any(isinstance(d, dict) and "avg_word_length" in d for d in f.get("derived", []))
    assert f.get("qc", {}).get("asr_conf") == pytest.approx(0.9)
    # ensure acoustic_named mapping is present and has predictable keys
    an = f.get("acoustic_named")
    assert isinstance(an, dict)
    assert "rms" in an
    # derived should include speaking_rate when supplied (tokens/sec) and pause_ratio derived from qc.speech_ratio
    assert any(isinstance(d, dict) and "speaking_rate" in d for d in f.get("derived", [])) is False
    # now simulate linguistic analysis having speaking_rate and QC with speech_ratio
    ac2 = {"acoustic": [0.0, 0.0, 0.0, 0.0, 0.0], "qc": {"snr_db": 0.0, "speech_ratio": 0.75}}
    ling2 = {**ling, "speaking_rate": 3.25}
    f2 = assemble_feature_frame("s1", ac2, ling2, 124)
    assert any(isinstance(d, dict) and "speaking_rate" in d for d in f2.get("derived", []))
    assert any(isinstance(d, dict) and "pause_ratio" in d for d in f2.get("derived", []))
