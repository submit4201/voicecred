import pytest
from pydantic import ValidationError

from voicecred.schemas.events import (
    FrameReceived,
    AcousticFeaturesReady,
    STTPartial,
    STTFinal,
    SpeakerSegmentsReady,
    FeatureFrameAssembled,
)
from voicecred.schemas.envelope import EnvelopeV1, QC, AcousticNamed, STTResult, STTWord, SpeakerSegment


def _make_env():
    qc = QC(snr_db=0.0, speech_ratio=0.6, voiced_seconds=0.1)
    acoustic = AcousticNamed(rms=0.1, zcr=0.01)
    words = [STTWord(word="ok", start_ms=0, end_ms=100, confidence=0.9)]
    stt = STTResult(raw="ok", confidence=0.9, words=words, is_final=True)
    return EnvelopeV1(session_id="s", window_id="w", timestamp_ms=42, acoustic_named=acoustic, qc=qc, stt=stt)


def test_frame_received_valid():
    ev = FrameReceived(pcm_length=1024)
    assert ev.type == "FrameReceived"


def test_acoustic_features_requires_envelope():
    with pytest.raises(ValidationError):
        AcousticFeaturesReady()


def test_stt_partial_final_payloads():
    env = _make_env()
    partial = STTPartial(envelope=env, partial=env.stt)
    assert partial.type == "STTPartial"

    final = STTFinal(envelope=env, final=env.stt)
    assert final.type == "STTFinal"


def test_speaker_segments_ready():
    env = _make_env()
    seg = SpeakerSegment(speaker="spkA", start_ms=0, end_ms=500)
    ev = SpeakerSegmentsReady(envelope=env, segments=[seg])
    assert ev.type == "SpeakerSegmentsReady"
