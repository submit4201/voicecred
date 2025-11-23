import pytest
import time
from fastapi.testclient import TestClient
from voicecred.main import app, store


client = TestClient(app)

def recv_with_timeout(ws, timeout=0.5):
    """Call ws.receive_json() in a thread and enforce a timeout so tests don't block forever.

    Using the shared TestClient WebSocket session's blocking receive_json directly will
    block the test thread; this helper runs it in a small threadpool and enforces a timeout
    via Future.result(timeout=...).
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(ws.receive_json)
        try:
            return fut.result(timeout=timeout)
        except Exception:
            try:
                fut.cancel()
            except Exception:
                pass
            raise


def test_health():
    r = client.get("/health")
    print('Health check response:', r.json())
    print('finished test_health\n')
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_start_and_ws_lifecycle():
    # start
    r = client.post("/sessions/start")
    assert r.status_code == 200
    js = r.json()
    sid = js["session_id"]
    token = js["token"]
    print(f'Started session {sid} with token {token}')

    # simple websocket lifecycle test using test client
    with client.websocket_connect(f"/ws/{sid}?token={token}") as ws:
        ack = ws.receive_json()
        assert ack["session_id"] == sid
        # control reset
        ws.send_json({"type": "control", "cmd": "reset"})
        out = ws.receive_json()
        assert out.get("status") == "calibrating"
        # send frame — the server may return a frame_ack OR an acoustic event depending on feature extraction
        ws.send_json({"type":"frame","ts": 2, "pcm": []})
        ack = ws.receive_json()
        assert ack.get("type") in ("frame_ack", "acoustic", "acoustic_batch")
        # finalize
        ws.send_json({"type":"control","cmd":"finalize"})
        print('Sent finalize command')
        out = ws.receive_json()
        print('Finalize response:', out)
        assert out.get("status") == "finalized"


def test_ws_batch_processing():
    # create a new session and send several frames, expect an acoustic_batch event
    r = client.post("/sessions/start")
    assert r.status_code == 200
    js = r.json()
    sid = js["session_id"]
    token = js["token"]
    print(f'Started session {sid} with token {token}')

    with client.websocket_connect(f"/ws/{sid}?token={token}") as ws:
        _ = ws.receive_json()
        # send multiple frames
        for i in range(3):
            ws.send_json({"type": "frame", "ts": i, "pcm": []})

        # collect messages until we see an acoustic_batch
        got_batch = False
        start = time.monotonic()
        while time.monotonic() - start < 5.0:
            try:
                msg = recv_with_timeout(ws, timeout=0.5)
            except Exception:
                # no message available yet — loop and try again until overall test timeout
                continue
            if msg.get("type") == "acoustic_batch":
                got_batch = True
                assert msg.get("size", 0) >= 1
                break
            print('Received message:', msg)
        print('Finished receiving messages for acoustic_batch')

        assert got_batch, "expected acoustic_batch event from background processor"


def test_ws_stt_and_linguistic_flow():
    # create a new session and send frames with transcript_override to trigger ASR & linguistic processing
    r = client.post("/sessions/start")
    assert r.status_code == 200
    js = r.json()
    sid = js["session_id"]
    token = js["token"]
    print(f'Started session {sid} with token {token}')
    print('Starting test_ws_stt_and_linguistic_flow\n\n')
    with client.websocket_connect(f"/ws/{sid}?token={token}") as ws:
        _ = ws.receive_json()
        # ask server to start calibration so feature frames will be used for baseline
        ws.send_json({"type": "control", "cmd": "reset"})
        _ = ws.receive_json()
        print('Sent reset command and received response',_)
        # send multiple frames that include a transcript_override so MockSTTAdapter picks it up
        for i in range(4):
            print(f'Sending frame {i} with transcript_override')
            ws.send_json({"type": "frame", "ts": i, "pcm": [], "transcript_override": "this is a test of asr"})

        got_asr = False
        got_ling = False
        got_feat = False
        pipeline_error = None
        # listen for several messages; expect asr_batch, linguistic_batch, feature_batch
        start = time.monotonic()
        while time.monotonic() - start < 5.0:
            print(time.monotonic() - start)
            try:
                msg = recv_with_timeout(ws, timeout=0.5)
                print('Received message:', msg)
                print('Received time:', time.monotonic() - start)
            except Exception:
                # no message available yet — loop and try again until overall test timeout
                print('No message received yet, continuing...',start)
                continue
            t = msg.get("type")
            print(f'Processing message of type: {t}')
            if t == "pipeline_status" and msg.get("stage") == "error":
                pipeline_error = msg
                print('Pipeline error received:', pipeline_error)
                break
            if t == "asr_batch":
                got_asr = True
                print('Received asr_batch:', msg)
            if t == "linguistic_batch":
                got_ling = True
                print('Received linguistic_batch:', msg)
            if t == "feature_batch":
                got_feat = True
                print('Received feature_batch:', msg)
                # check a few expected fields exist in assembled feature
                items = msg.get("items", [])
                if items:
                    f = items[0]
                    assert "qc" in f and isinstance(f.get("qc"), dict)
                    # assembled frames should include asr_conf in QC when linguistic present
                    assert "derived" in f and isinstance(f.get("derived"), list)
                    # derived should include speaking_rate (when available from ASR timestamps)
                    assert any(isinstance(d, dict) and "speaking_rate" in d for d in f.get("derived", []))
                    # and pause_ratio derived from acoustic QC should be present
                    assert any(isinstance(d, dict) and "pause_ratio" in d for d in f.get("derived", []))
                    # when calibration has occurred, assembled feature frames should include normalized and scoring outputs
                    assert "normalized" in f and isinstance(f.get("normalized"), dict)
                    assert "score" in f
                    assert "explain" in f and isinstance(f.get("explain"), dict)
            if got_asr and got_ling and got_feat:
                print('Received all expected batches: asr, linguistic, feature')
                break
        print('Finished receiving messages for ASR, linguistic, and feature batches')   
        assert pipeline_error is None, f"pipeline error during run: {pipeline_error}"
        assert got_asr, "expected asr_batch"
        assert got_ling, "expected linguistic_batch"
        assert got_feat, "expected assembled feature_batch"


def test_ws_asr_low_confidence_gating():
    # temporarily swap the stt_adapter used by the app to one that returns low confidence
    from voicecred.main import stt_adapter
    from voicecred.stt import MockSTTAdapter
    print('Starting test_ws_asr_low_confidence_gating')

    # set a low-confidence adapter
    old_adapter = stt_adapter
    print('Swapping to low confidence STT adapter')
    try:
        import voicecred.main as vcmain
        print('Swapping to low confidence STT adapter in voicecred.main')
        vcmain.stt_adapter = MockSTTAdapter(override="low confidence test", default_confidence=0.1)
        print('Swapped to low confidence STT adapter')
        r = client.post("/sessions/start")
        assert r.status_code == 200
        js = r.json()
        print('Session start response JSON:', js)
        sid = js["session_id"]
        print(f'Session ID: {sid}')
        token = js["token"]
        print(f'Token: {token}')
        print(f'Started session {sid} with token {token}')

        with client.websocket_connect(f"/ws/{sid}?token={token}") as ws:
            print('Connected to websocket')
            _ = ws.receive_json()
            print('Received initial ack from websocket')
            for i in range(3):
                print(f'Sending frame {i} with low confidence transcript_override')
                ws.send_json({"type": "frame", "ts": i, "pcm": [], "transcript_override": "low confidence test"})
            print('Sent frames with low confidence transcript_override')
            got_asr = False
            print('Waiting for ASR and linguistic batches')
            got_ling = False
            print('Waiting for pipeline error or timeout')
            pipeline_error = None
            print('Starting wait loop for pipeline messages')
            start = time.monotonic()
            print(f'Started session {sid} with token {token}')
            # timeout after 5 seconds
            print(start)
            while time.monotonic() - start < 5.0:
                try:
                    print(time.monotonic() - start)
                    msg = recv_with_timeout(ws, timeout=0.5)
                except Exception:
                    # no message available yet — loop and try again until overall test timeout
                    print('Exception while receiving JSON, continuing wait loop',start)
                    continue
                if msg.get("type") == "pipeline_status" and msg.get("stage") == "error":
                    print('Pipeline error received:', msg)
                    print("timeout/err:", start)
                    pipeline_error = msg
                    break
                if msg.get("type") == "asr_batch":
                    got_asr = True
                    print('Received ASR batch:', msg)
                    break
                if msg.get("type") == "linguistic_batch":
                    got_ling = True
                    print('Received linguistic batch:', msg)
                    break
                if got_asr and got_ling:
                    print('Received both ASR and linguistic batches, breaking loop')
                    break
                    
            print('Finished receiving messages for low confidence ASR gating test')
            assert pipeline_error is None, f"pipeline error during run: {pipeline_error}"
            print(f"Pipeline error: {pipeline_error}")
            
            assert got_asr
            print('Asserting linguistic batch was not received due to low ASR confidence')
            assert not got_ling, "linguistic pipeline should be gated on low ASR confidence"
            print('Asserting linguistic batch was not received due to low ASR confidence passed')
    finally:
        # restore original adapter
        print('Restoring original STT adapter')
        import voicecred.main as vcmain
        vcmain.stt_adapter = old_adapter

        print('Restored original STT adapter')
        print('finished test_ws_asr_low_confidence_gating\n')