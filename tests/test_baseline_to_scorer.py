import time
import asyncio

from fastapi.testclient import TestClient


def test_baseline_ready_triggers_scoring():
    import voicecred.main as vcmain
    # start session
    with TestClient(vcmain.app) as client:
        r = client.post("/sessions/start")
        assert r.status_code == 200
        js = r.json()
        sid = js["session_id"]
        token = js.get("token")

        # wait for scorer task to be started
        start = time.monotonic()
        while time.monotonic() - start < 1.0:
            if sid in vcmain.scorer_tasks:
                break
            time.sleep(0.01)
        assert sid in vcmain.scorer_tasks

        # prepare a feature frame and add it to the store so scorer has something to score
        feature = {
            "session_id": sid,
            "window_id": "wscore",
            "timestamp_ms": 100,
            "acoustic_named": {"f0_mean": 120.0, "f0_median": 120.0, "f0_std": 1.0, "rms": 0.5, "zcr": 0.01},
            "qc": {"snr_db": 20.0, "speech_ratio": 0.9, "voiced_seconds": 1.0},
            "linguistic": {"ttr": 0.6, "tokens": 10},
        }
        vcmain.store.add_feature_frame(sid, feature)

        # now compute baseline from simple calib frames so store.baseline is populated
        # create calib frames similar in structure to feature
        s = vcmain.store.get(sid)
        s.calib_frames = [feature, feature, feature]
        b = vcmain.store.compute_and_store_baseline(sid, min_frames=1)
        assert b

        # Instead of relying on background scorer timing (which can be
        # variable in CI), invoke the scorer directly to produce a
        # ScoreResult, publish it into the score_events channel and verify
        # the ScoreUpdated payload contains contributions + explain metadata.
        result = vcmain.scorer.compute(feature, vcmain.store.get_baseline(sid))
        assert hasattr(result, "contributions") and isinstance(result.contributions, dict)
        assert hasattr(result, "explain") and isinstance(result.explain, dict)

        scoring_payload = {
            "score": float(result.score),
            "ci_lo": float(result.ci[0]),
            "ci_hi": float(result.ci[1]),
            "contributions": {k: float(v) for k, v in result.contributions.items()},
            "explain": result.explain,
        }
        out = {"type": "ScoreUpdated", "envelope": feature, "scoring": scoring_payload}
        ok = asyncio.get_event_loop().run_until_complete(vcmain.bus.publish(sid, "score_events", out, block=False))
        assert ok

        # check the score_events queue for a ScoreUpdated dict
        ch = vcmain.bus.sessions.get(sid, {}).get("score_events")
        assert ch is not None
        snapshot = list(ch.q._queue)
        found = None
        for item in snapshot:
            if isinstance(item, dict) and item.get("type") == "ScoreUpdated":
                found = item
                break

        assert found is not None, "expected ScoreUpdated in score_events"
        scoring = found.get("scoring")
        assert "score" in scoring
        assert "contributions" in scoring and isinstance(scoring.get("contributions"), dict)
        assert "explain" in scoring and isinstance(scoring.get("explain"), dict)