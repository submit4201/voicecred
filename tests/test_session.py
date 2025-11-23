import pytest
from voicecred.session import InMemorySessionStore

def test_create_and_get_session():
    store = InMemorySessionStore()
    s = store.create_session()
    assert s.session_id is not None
    got = store.get(s.session_id)
    print('Created session:', s)    
    assert got.session_id == s.session_id


def test_add_frame_and_finalize():
    store = InMemorySessionStore()
    s = store.create_session()
    store.add_frame(s.session_id, {"ts": 1})
    store.add_calib_frame(s.session_id, {"ts": 1})
    snapshot = store.finalize(s.session_id)
    print('Finalized session snapshot:', snapshot)
    assert snapshot["session_id"] == s.session_id
    assert snapshot["frames"] == 1
    assert snapshot["calib_frames"] == 1


def test_compute_and_store_baseline():
    store = InMemorySessionStore()
    s = store.create_session()
    # create three simple calib frames with consistent numeric keys
    for v in (100, 102, 98):
        frame = {
            "acoustic": {"pitch": v},
            "linguistic": {"ttr": 0.5},
            "derived": [{"avg_word_length": 4.2}],
            "qc": {"snr_db": 20.0},
        }
        store.add_calib_frame(s.session_id, frame)

    baseline = store.compute_and_store_baseline(s.session_id, min_frames=3)
    assert isinstance(baseline, dict)
    # expect acoustic.pitch baseline entry
    assert "acoustic.pitch" in baseline
    assert baseline["acoustic.pitch"]["count"] == 3
    # avg_word_length should be present
    assert "derived.avg_word_length" in baseline


def test_compute_and_store_baseline_skips_nan_metrics():
    store = InMemorySessionStore()
    s = store.create_session()
    # three calib frames but acoustic.pitch has only one valid value and two NaNs
    for v in (float('nan'), 100.0, float('nan')):
        frame = {
            "acoustic": {"pitch": v},
            "linguistic": {"ttr": 0.5},
        }
        store.add_calib_frame(s.session_id, frame)

    baseline = store.compute_and_store_baseline(s.session_id, min_frames=3)
    # acoustic.pitch should be skipped (only 1 valid < min_frames requirement)
    assert "acoustic.pitch" not in baseline
    # linguistic.ttr has valid numeric values and should be present
    assert "linguistic.ttr" in baseline


def test_compute_and_store_baseline_qc_gating_filters_silent_frames():
    store = InMemorySessionStore()
    s = store.create_session()

    # Add three calibration frames: two silent/low-QC and one voiced/high-QC
    silent = {
        "acoustic": {"pitch": float('nan')},
        "linguistic": {"ttr": 0.5},
        "qc": {"speech_ratio": 0.0, "voiced_seconds": 0.0},
    }
    voiced = {
        "acoustic": {"pitch": 100.0},
        "linguistic": {"ttr": 0.5},
        "qc": {"speech_ratio": 1.0, "voiced_seconds": 0.2},
    }

    store.add_calib_frame(s.session_id, silent)
    store.add_calib_frame(s.session_id, voiced)
    store.add_calib_frame(s.session_id, silent)

    # Require at least 2 frames to compute baselines: after QC gating only 1 qualifies
    baseline = store.compute_and_store_baseline(s.session_id, min_frames=2)
    assert baseline == {}

    # metadata should indicate filtering and insufficient frames
    meta = s.metadata.get("baseline_filtering", {})
    assert meta["total_calib_frames"] == 3
    assert meta["kept"] == 1
    assert meta.get("insufficient_after_qc", False) is True


def test_compute_and_store_baseline_qc_allows_voiced_frames():
    store = InMemorySessionStore()
    s = store.create_session()

    # three voiced frames should pass QC
    voiced = {
        "acoustic": {"pitch": 100.0},
        "linguistic": {"ttr": 0.5},
        "qc": {"speech_ratio": 1.0, "voiced_seconds": 0.2},
    }
    for _ in range(3):
        store.add_calib_frame(s.session_id, voiced)

    baseline = store.compute_and_store_baseline(s.session_id, min_frames=3)
    assert isinstance(baseline, dict)
    assert "acoustic.pitch" in baseline


def test_compute_and_store_baseline_production_policy_requires_more():
    store = InMemorySessionStore()
    s = store.create_session()

    # Add a few voiced frames totalling less than the prod voiced seconds default
    voiced_short = {
        "acoustic": {"pitch": 100.0},
        "linguistic": {"ttr": 0.5},
        "qc": {"speech_ratio": 1.0, "voiced_seconds": 0.5},
    }
    for _ in range(3):
        store.add_calib_frame(s.session_id, voiced_short)

    # Use production policy with an intentionally high voiced_seconds threshold; should be rejected
    baseline = store.compute_and_store_baseline(s.session_id, min_frames=None, use_production_policy=True, min_voiced_seconds=60.0)
    assert baseline == {}


def test_compute_and_store_baseline_production_policy_allows_when_enough_voiced():
    store = InMemorySessionStore()
    s = store.create_session()

    # Add frames totalling enough voiced seconds
    voiced = {
        "acoustic": {"pitch": 100.0},
        "linguistic": {"ttr": 0.5},
        "qc": {"speech_ratio": 1.0, "voiced_seconds": 30.0},
    }
    for _ in range(2):
        store.add_calib_frame(s.session_id, voiced)

    # Use production policy with min_voiced_seconds=60 -> total voiced = 60 -> should pass
    baseline = store.compute_and_store_baseline(s.session_id, min_frames=None, use_production_policy=True, min_voiced_seconds=60.0)
    assert isinstance(baseline, dict)


def test_baseline_persistence_opt_in():
    store = InMemorySessionStore()
    s = store.create_session()
    s.persist_baseline = True

    for v in (100, 102, 98):
        frame = {
            "acoustic": {"pitch": v},
            "linguistic": {"ttr": 0.5},
            "derived": [{"avg_word_length": 4.2}],
            "qc": {"snr_db": 20.0},
        }
        store.add_calib_frame(s.session_id, frame)

    baseline = store.compute_and_store_baseline(s.session_id, min_frames=3)
    # baseline persisted when the session opted in
    assert s.session_id in store.persistent_baselines
    assert store.persistent_baselines[s.session_id] == baseline
