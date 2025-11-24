from src.voicecred.session.baseline import compute_baseline_from_frames


def make_frames():
    # create 3 frames with acoustic, linguistic, derived and qc keys
    f1 = {
        "acoustic": {"f0_mean": 100.0, "rms": 1.0, "zcr": 0.1},
        "linguistic": {"ttr": 0.5, "tokens": 10, "avg_tokens_per_sentence": 5.0},
        "derived": [{"avg_word_length": 4.2}],
        "qc": {"snr_db": 20.0, "speech_ratio": 0.9, "voiced_seconds": 0.5}
    }
    f2 = {**f1}
    f3 = {**f1}
    return [f1, f2, f3]


def test_compute_baseline_basic():
    frames = make_frames()
    baseline = compute_baseline_from_frames(frames, min_frames=3)
    # baseline should be non-empty and include acoustic.f0_mean
    assert isinstance(baseline, dict)
    assert "acoustic.f0_mean" in baseline
    assert baseline["acoustic.f0_mean"]["count"] >= 3
