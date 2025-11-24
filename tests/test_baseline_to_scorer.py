import pytest
from httpx import AsyncClient, ASGITransport
import asyncio
import time

@pytest.mark.anyio
async def test_baseline_ready_triggers_scoring():
    import voicecred.main as vcmain

    # Use AsyncClient to ensure test runs in the same event loop as the app tasks
    async with AsyncClient(transport=ASGITransport(app=vcmain.app), base_url="http://test") as client:
        r = await client.post("/sessions/start")
        assert r.status_code == 200
        js = r.json()
        sid = js["session_id"]
        token = js.get("token")

        # wait for scorer task to be started and ready (subscribed to ops_events)
        # yield to let background tasks start
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if sid in vcmain.scorer_tasks and sid in vcmain.scorer_ready:
                break
            await asyncio.sleep(0.01)

        if sid not in vcmain.scorer_tasks or sid not in vcmain.scorer_ready:
            raise AssertionError(f"scorer_ready was not populated for session {sid!r}")

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

        # Pre-calculate a baseline so that when we send BaselineReady, the scorer can actually score
        s = vcmain.store.get(sid)
        s.calib_frames = [feature, feature, feature]
        b = vcmain.store.compute_and_store_baseline(sid, min_frames=1)
        assert b

        # Publish BaselineReady for this session on the ops_events bus
        baseline_ready_event = {
            "type": "BaselineReady",
            "session_id": sid,
            "token": token,
        }
        # ops_events is a Channel, use publish helper from main or direct bus publish
        # Since we are in the same loop, bus.publish is safe
        await vcmain.bus.publish(sid, "ops_events", baseline_ready_event, block=False)

        # Wait for a corresponding ScoreUpdated event on the score_events bus.
        q = vcmain.bus.subscribe(sid, "score_events")
        try:
            # Wait with timeout
            evt = await asyncio.wait_for(q.get(), timeout=5.0)
        except asyncio.TimeoutError:
            evt = None

        # handle both dict and Pydantic model if necessary (q.get returns dict usually from safe_publish)
        if evt and hasattr(evt, "model_dump"):
            evt = evt.model_dump()

        assert evt is not None, "Timed out waiting for ScoreUpdated event"
        # basic sanity assertions on the ScoreUpdated event
        assert evt.get("type") == "ScoreUpdated"
        # Scoring payload validation
        scoring = evt.get("scoring")
        assert scoring is not None
        assert "score" in scoring
        assert "contributions" in scoring
        assert "explain" in scoring
