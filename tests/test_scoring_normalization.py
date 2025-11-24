from src.voicecred.scoring.scorer import Scorer


def test_scorer_computes_score_and_ci():
    scorer = Scorer()
    # simple baseline for two metrics
    baseline = {
        "acoustic.f0_mean": {"median": 100.0, "mad": 1.0, "count": 3},
        "linguistic.ttr": {"median": 0.5, "mad": 0.01, "count": 3},
    }

    frame = {
        "acoustic": {"f0_mean": 102.0},
        "linguistic": {"ttr": 0.52},
        "derived": [],
    }

    res = scorer.compute(frame, baseline)
    assert hasattr(res, "score")
    assert 0.0 <= res.score <= 100.0
    assert isinstance(res.ci, tuple) and len(res.ci) == 2
