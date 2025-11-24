import time

from voicecred.bus.bus import EventBus


def test_publish_and_subscribe_basic():
    bus = EventBus(default_maxsize=10)
    sid = "sess-1"
    bus.register_channel(sid, "frames_in", maxsize=5)
    q = bus.subscribe(sid, "frames_in")

    # publish some items
    import asyncio

    async def _pub():
        await bus.publish(sid, "frames_in", {"i": 1})
        await bus.publish(sid, "frames_in", {"i": 2})

    asyncio.get_event_loop().run_until_complete(_pub())

    assert q.get_nowait() == {"i": 1}
    assert q.get_nowait() == {"i": 2}


def test_backpressure_and_metrics():
    bus = EventBus(default_maxsize=1)
    sid = "sess-2"
    ch = bus.register_channel(sid, "frames_in", maxsize=1)
    q = bus.subscribe(sid, "frames_in")

    import asyncio

    async def _pub_many():
        ok1 = await bus.publish(sid, "frames_in", {"a": 1}, block=False)
        ok2 = await bus.publish(sid, "frames_in", {"a": 2}, block=False)
        return ok1, ok2

    ok1, ok2 = asyncio.get_event_loop().run_until_complete(_pub_many())
    # with maxsize=1 non-blocking publish: first should succeed, second should be dropped
    assert ok1 is True
    assert ok2 is False

    metrics = bus.metrics(sid)
    assert metrics["frames_in"]["dropped"] >= 1


def test_main_integration_frame_publish():
    # Use the app-level bus exposed in main to verify frame publishes for a live session
    from fastapi.testclient import TestClient
    import voicecred.main as vcmain

    client = TestClient(vcmain.app)
    r = client.post("/sessions/start")
    assert r.status_code == 200
    js = r.json()
    sid = js["session_id"]
    token = js["token"]

    # subscribe to the bus frames_in for this session (get the queue object)
    ch = vcmain.bus.sessions.get(sid, {}).get("frames_in")
    assert ch is not None, "frames_in channel should be present for session"

    # connect websocket and send a frame; an event should be published to frames_in
    with client.websocket_connect(f"/ws/{sid}?token={token}") as ws:
        _ = ws.receive_json()  # ack
        ws.send_json({"type": "frame", "ts": 1, "pcm": [0, 1, 2]})

        # poll the queue for a short time for frames_in event
        start = time.monotonic()
        found = None
        while time.monotonic() - start < 2.0:
            try:
                found = ch.q.get_nowait()
                break
            except Exception:
                time.sleep(0.01)

        assert found is not None and found.get("type") == "FrameReceived"

        # also ensure STT and Speaker events were published to their channels
        ch_stt = vcmain.bus.sessions.get(sid, {}).get("stt_q")
        ch_spk = vcmain.bus.sessions.get(sid, {}).get("speaker_q")

        # wait briefly for asynchronous background STT / speaker tasks
        start = time.monotonic()
        stt_found = None
        spk_found = None
        while time.monotonic() - start < 3.0 and (stt_found is None or spk_found is None):
            try:
                if stt_found is None and ch_stt is not None and ch_stt.q.qsize() > 0:
                    stt_found = ch_stt.q.get_nowait()
            except Exception:
                pass
            try:
                if spk_found is None and ch_spk is not None and ch_spk.q.qsize() > 0:
                    spk_found = ch_spk.q.get_nowait()
            except Exception:
                pass
            if stt_found and spk_found:
                break
            time.sleep(0.05)

        assert stt_found is not None and stt_found.get("type") in ("STTPartial", "STTFinal")
        assert spk_found is not None and spk_found.get("type") == "SpeakerSegmentsReady"
