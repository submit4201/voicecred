import pytest
from voicecred.scorer import Scorer


def test_scorer_basic():
    sc = Scorer()
    # baseline with simple rms metric
    baseline = {
        "acoustic.rms": {"median": 0.5, "mad": 0.1, "count": 10}
    }
    frame = {"acoustic_named": {"rms": 0.6}}
    res = sc.compute(frame, baseline)
    print('Score result:', res)
    assert 0.0 <= res.score <= 100.0
    
    assert "acoustic.rms" in res.z_scores
    assert "acoustic.rms" in res.contributions


def test_update_ema_and_buffer():
    sc = Scorer()
    state = {"ema_alpha": 0.5, "ema_last": None, "recent_scores": [], "score_window": 3}
    v1 = sc.update_ema(state, 1.0)
    assert v1 == 1.0
    v2 = sc.update_ema(state, 0.5)
    # with alpha=0.5, next ema = 0.5*0.5 + 0.5*1.0 = 0.75
    print('Updated EMA:', v2)
    assert abs(v2 - 0.75) < 1e-6
    # buffer maintains last <= window samples
    sc.update_ema(state, 2.0)
    sc.update_ema(state, 3.0)
    print('Recent scores buffer:', state["recent_scores"])
    assert len(state["recent_scores"]) <= 3


def test_ci_from_recent_scores():
    sc = Scorer()
    baseline = {"acoustic.rms": {"median": 0.2, "mad": 0.05, "count": 10}}
    frame = {"acoustic_named": {"rms": 0.3}, "_recent_scores": [0.1, 0.2, 0.3, 0.5]}
    res = sc.compute(frame, baseline)
    print('Score result with CI:', res)
    assert isinstance(res.ci, tuple) and len(res.ci) == 2
    assert 0.0 <= res.ci[0] <= res.ci[1] <= 100.0
