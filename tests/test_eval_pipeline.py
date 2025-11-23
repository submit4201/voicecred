import math

from scripts.eval_pipeline import run_demo, run_evaluation


def test_eval_harness_runs_and_returns_scored_frames():
    trace = run_demo(n_silent=1, n_voiced=1)
    assert isinstance(trace, list)
    assert len(trace) >= 1
    f = trace[0]
    # frames should contain keys for acoustic and qc and may include score/normalized
    assert "acoustic" in f
    assert "qc" in f or isinstance(f.get("acoustic"), (list, dict))
    # if there was enough frames for baseline, there should be a score
    if any(isinstance(x, dict) and x.get("acoustic") for x in trace) and len(trace) >= 2:
        assert "score" in trace[0]


def test_run_evaluation_metrics():
    out = run_evaluation(n_calib=4, n_good=8, n_bad=8)
    assert isinstance(out, dict)
    assert "trace" in out and isinstance(out["trace"], list)
    assert "probs" in out and len(out["probs"]) == 16
    assert "labels" in out and len(out["labels"]) == 16
    # AUC should be between 0 and 1 or NaN if trivial
    assert "auc" in out
    if not (math.isnan(out["auc"])):
        assert 0.0 <= out["auc"] <= 1.0
    # Brier score is between 0 and 1
    assert "brier" in out
    assert 0.0 <= out["brier"] <= 1.0
    # calibration should be a list of tuples (bin_mid, avg_pred, avg_label) length 10
    assert "calibration" in out and isinstance(out["calibration"], list)
    assert len(out["calibration"]) == 10
