import pytest
from pydantic import ValidationError

from voicecred.schemas.envelope import EnvelopeV1, QC, AcousticNamed, STTResult, STTWord


def test_envelope_minimal_valid():
    qc = QC(snr_db=10.0, speech_ratio=0.5, voiced_seconds=0.1)
    acoustic = AcousticNamed(rms=0.1, zcr=0.01)
    env = EnvelopeV1(session_id="s1", window_id="w1", timestamp_ms=123, acoustic_named=acoustic, qc=qc)
    assert env.version == "v1"
    assert env.session_id == "s1"


def test_envelope_missing_qc_fails():
    with pytest.raises(ValidationError):
        EnvelopeV1(session_id="s1", window_id="w1", timestamp_ms=1)


def test_qc_invalid_ranges():
    # speech_ratio out of bounds
    with pytest.raises(ValidationError):
        QC(snr_db=0.0, speech_ratio=1.5, voiced_seconds=0.1)

    # voiced_seconds negative
    with pytest.raises(ValidationError):
        QC(snr_db=0.0, speech_ratio=0.5, voiced_seconds=-1.0)


def test_acoustic_named_required_fields():
    qc = QC(snr_db=0.0, speech_ratio=0.2, voiced_seconds=0.05)
    # rms and zcr missing -> should raise
    with pytest.raises(ValidationError):
        EnvelopeV1(session_id="s1", window_id="w1", timestamp_ms=999, acoustic_named=AcousticNamed(), qc=qc)

def test_stt_words_validation():
    qc = QC(snr_db=1.0, speech_ratio=0.5, voiced_seconds=0.05)
    acoustic = AcousticNamed(rms=0.1, zcr=0.01)
    words = [STTWord(word="hello", start_ms=0, end_ms=100, confidence=0.9)]
    stt = STTResult(raw="hello", confidence=0.9, words=words, is_final=True)
    env = EnvelopeV1(session_id="s2", window_id="w2", timestamp_ms=0, acoustic_named=acoustic, qc=qc, stt=stt)
    assert env.stt is not None and env.stt.confidence == 0.9
