import pytest

# if numpy isn't installed, skip the whole module
import numpy
np = numpy
import parselmouth
from voicecred.acoustic import AcousticEngine


def pcm_from_float32(arr: np.ndarray) -> bytes:
    # clamp and convert to int16 bytes
    clipped = np.clip(arr, -1.0, 1.0)
    print('clipped', clipped[:10])
    print('float32 to int16 bytes conversion')
    return (np.int16(clipped * 32767)).tobytes()


def test_sine_voice_features():
    # pytest.importorskip("parselmouth")#
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    freq = 220.0
    sine = 0.3 * np.sin(2 * np.pi * freq * t)
    frame = {"pcm": pcm_from_float32(sine), "timestamp_ms": 0, "sample_rate": sr}
    engine = AcousticEngine(sample_rate=sr)
    res = engine.process_frame(frame["pcm"], frame["timestamp_ms"])

    # Expect pitch around 220 Hz
    print('Pitch detected:', res.acoustic[0])
    print('Expected frequency:', freq)
    print('finished test_sine_voice_features\n')
    assert res.acoustic[0] is not None
    assert abs(res.acoustic[0] - freq) < 25.0
    assert res.qc["speech_ratio"] > 0.7


def test_silence_has_low_speech_ratio():
    # pytest.importorskip("parselmouth")#
    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    frame = {"pcm": pcm_from_float32(silence), "timestamp_ms": 10, "sample_rate": sr}
    engine = AcousticEngine(sample_rate=sr)
    res = engine.process_frame(frame["pcm"], frame["timestamp_ms"])
    print('Speech ratio for silence:', res.qc["speech_ratio"])
    print('finished test_silence_has_low_speech_ratio\n')
    assert res.qc["speech_ratio"] < 0.1

