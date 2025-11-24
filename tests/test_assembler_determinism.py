from src.voicecred.assembler.buffer import WindowBuffer


def make_parts():
    feature = {
        "acoustic_named": {"f0_mean": 100.0, "f0_median": 100.0, "f0_std": 0.0, "rms": 1.0, "zcr": 0.1},
        "linguistic": {"ttr": 0.5, "tokens": 10, "asr_quality": 0.95},
        "scoring": {"score": None}
    }
    qc = {"snr_db": 20.0, "speech_ratio": 0.9, "voiced_seconds": 0.5}
    stt = {"raw": "hello world", "confidence": 0.95, "is_final": True}
    speaker = [{"speaker": "A", "start_ms": 0, "end_ms": 1000}]
    return feature, qc, stt, speaker


def test_assembler_is_order_independent():
    buf1 = WindowBuffer()
    feature, qc, stt, speaker = make_parts()

    # sequence A: qc then feature
    r1 = buf1.add_part("s1", "w1", "qc", qc)
    assert r1 is None
    r2 = buf1.add_part("s1", "w1", "feature", {**feature, "timestamp_ms": 123})
    assert r2 and r2.get("action") == "assembled"
    env1 = r2["envelope"].model_dump()

    # sequence B: fresh buffer, feature then qc
    buf2 = WindowBuffer()
    r3 = buf2.add_part("s1", "w1", "feature", {**feature, "timestamp_ms": 123})
    assert r3 is None
    r4 = buf2.add_part("s1", "w1", "qc", qc)
    assert r4 and r4.get("action") == "assembled"
    env2 = r4["envelope"].model_dump()

    assert env1 == env2


def test_late_patch_generates_diff():
    buf = WindowBuffer()
    feature, qc, stt, speaker = make_parts()
    # initial assembly
    assert buf.add_part("s1", "w2", "qc", qc) is None
    result = buf.add_part("s1", "w2", "feature", {**feature, "timestamp_ms": 321})
    assert result and result.get("action") == "assembled"

    # now add an stt partial/final that should create a patch
    patch = buf.add_part("s1", "w2", "stt", stt)
    assert patch and patch.get("action") == "patch"
    assert "stt" in patch.get("changed_fields")
