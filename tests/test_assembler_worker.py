import time
import pytest
import asyncio
from fastapi.testclient import TestClient
import queue

def test_assembler_worker_start_and_publish():
    """Start a session; publish a QC partial and a feature_frame to window_buffer and
    assert the assembler background worker publishes a ui_out assembled/patch message.
    """
    import voicecred.main as vcmain

    # Create and set a new event loop for this test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        with TestClient(vcmain.app) as client:
            r = client.post("/sessions/start")
            assert r.status_code == 200
            js = r.json()
            sid = js["session_id"]
            token = js.get("token")

            # wait briefly for background assembler task to start
            start = time.monotonic()
            while time.monotonic() - start < 1.0 and sid not in vcmain.assembler_tasks:
                time.sleep(0.01)
            assert sid in vcmain.assembler_tasks

            qc = {"snr_db": 20.0, "speech_ratio": 0.9, "voiced_seconds": 0.5}
            feature = {
                "acoustic_named": {"f0_mean": 100.0, "f0_median": 100.0, "f0_std": 0.0, "rms": 1.0, "zcr": 0.1},
                "linguistic": {"ttr": 0.5, "tokens": 5},
                "timestamp_ms": 123,
            }

            async def _pub():
                await vcmain.bus.publish(sid, "window_buffer", {"type": "qc", "window_id": "w1", "qc": qc}, block=False)
                await vcmain.bus.publish(sid, "window_buffer", {"type": "feature_frame", "window_id": "w1", "feature": feature, "timestamp_ms": 123}, block=False)

            # create websocket to consume ui_out messages from the server's UI forwarder
            with client.websocket_connect(f"/ws/{sid}?token={token}") as ws:
                ack = ws.receive_json()
                assert ack.get("type") == "ack"

                # publish items into the session window_buffer (server will process them)
                # Use the loop we created to run the async publish
                loop.run_until_complete(_pub())

                # give the assembler a small moment to bind/process
                time.sleep(0.05)

                # task should still be running
                t = vcmain.assembler_tasks.get(sid)
                assert t is not None
                if t.done():
                    exc = None
                    try:
                        exc = t.exception()
                    except Exception:
                        pass
                    pytest.fail(f"assembler task completed early, exception={exc}")

                # consume messages from the websocket forwarder until we see a ui_diff payload
                found = None
                start = time.monotonic()
                while time.monotonic() - start < 3.0:
                    try:
                        msg = ws.receive_json()
                        if msg.get("type") == "ui_diff":
                            found = msg.get("payload")
                            break
                    except Exception:
                        time.sleep(0.01)

                assert found is not None, "expected a ui_diff message from server websocket"
                assert isinstance(found, dict)
                assert found.get("action") in ("assembled", "patch") or found.get("type") == "FeatureFrameAssembled"
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_assembler_task_removed_on_flush():
    """Start a session and then flush it; assembler background worker should be stopped/removed."""
    import voicecred.main as vcmain

    # Create and set a new event loop for this test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        with TestClient(vcmain.app) as client:
            r = client.post("/sessions/start")
            assert r.status_code == 200
            js = r.json()
            sid = js["session_id"]

            # ensure assembler task present
            start = time.monotonic()
            while time.monotonic() - start < 1.0 and sid not in vcmain.assembler_tasks:
                time.sleep(0.01)
            assert sid in vcmain.assembler_tasks

            # flush the session (should stop assembler)
            r = client.post(f"/admin/sessions/{sid}/flush")
            assert r.status_code == 200

            # task should be removed
            start = time.monotonic()
            removed = False
            while time.monotonic() - start < 1.0 and sid in vcmain.assembler_tasks:
                time.sleep(0.01)

            # Check if removed
            if sid not in vcmain.assembler_tasks:
                removed = True

            assert removed
    finally:
        asyncio.set_event_loop(None)
        loop.close()
