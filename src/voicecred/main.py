from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from typing import Dict
import uvicorn
from src.voicecred.session import InMemorySessionStore
from src.voicecred.acoustic import AcousticEngine
from src.voicecred.stt import MockSTTAdapter, STTAdapter, create_stt_adapter
import os
from src.voicecred.linguistic import LinguisticEngine
from src.voicecred.speaker import create_speaker_adapter
from src.voicecred.assembler import assemble_feature_frame, Assembler
from src.voicecred.bus import EventBus
from src.voicecred.schemas.events import FrameReceived
from src.voicecred.scorer import Scorer
from src.voicecred.utils import baseline as baseline_utils
import asyncio
from src.voicecred.auth import create_session_token, verify_session_token
import dotenv

dotenv.load_dotenv(".env")
hf_api_key = os.getenv("HF_API_KEY")
import logging

from src.voicecred.utils.logger_util import get_logger, logging
logger=get_logger(__name__,logging.DEBUG)
app = FastAPI(title="voicecred-ingress", version="0.0.1")
store = InMemorySessionStore()
# lightweight in-memory event bus (per session channels)
bus = EventBus()
acoustic_engine = AcousticEngine()
# default adapters (pluggable in prod)
# Allow selecting adapter via environment variable STT_ADAPTER (e.g. 'mock' or 'whisper')
try:
    stt_adapter: STTAdapter = create_stt_adapter(os.environ.get("STT_ADAPTER", "mock"))
except Exception:
    # fall back to the Mock adapter if factory or config fails
    stt_adapter: STTAdapter = MockSTTAdapter()
# speaker adapter selection
try:
    speaker_adapter = create_speaker_adapter(os.environ.get("SPEAKER_ADAPTER", "mock"))
except Exception:
    speaker_adapter = None
linguistic_engine = LinguisticEngine()
# scorer instance (singleton per app)
scorer = Scorer()

# assembler worker / lifecycle state (one per session to keep things simple)
assembler = Assembler(bus)
assembler_tasks: Dict[str, asyncio.Task] = {}
scorer_tasks: Dict[str, asyncio.Task] = {}
# set of session_ids for which the scorer has subscribed to the ops_events queue
# this helps tests and other callers know the scorer loop is ready to receive events
scorer_ready: set[str] = set()


def _start_assembler_for_session(session_id: str):
    # create a background task for the assembler loop if not already running
    if session_id in assembler_tasks:
        return
    try:
        t = asyncio.create_task(assembler.start_for_session(session_id))
        assembler_tasks[session_id] = t
    except Exception:
        # best-effort: if creating a task fails, don't crash the server
        logger.debug("failed to start assembler task for %s", session_id, exc_info=True)


def _stop_assembler_for_session(session_id: str):
    t = assembler_tasks.pop(session_id, None)
    if t is not None:
        try:
            t.cancel()
        except Exception:
            logger.debug("failed to cancel assembler for %s", session_id, exc_info=True)


def _start_scorer_for_session(session_id: str):
    if session_id in scorer_tasks:
        return
    try:
        t = asyncio.create_task(_scorer_loop_for_session(session_id))
        scorer_tasks[session_id] = t
    except Exception:
        logger.debug("failed to start scorer task for %s", session_id, exc_info=True)


def _stop_scorer_for_session(session_id: str):
    t = scorer_tasks.pop(session_id, None)
    if t is not None:
        try:
            t.cancel()
        except Exception:
            logger.debug("failed to cancel scorer for %s", session_id, exc_info=True)
MIN_ASR_CONF = 0.6
# pipeline stage timeouts (seconds)
STAGE_TIMEOUTS = {
    "acoustic": 3.0,
    "stt": 2.0,
    "linguistic": 2.0,
}
security = HTTPBearer()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/sessions/start")
async def start_session():
    s = store.create_session()
    # ensure bus channels exist for this session
    bus._ensure_session(s.session_id)
    # create a dedicated channel for score events so tests and other
    # consumers can reliably subscribe before scoring happens
    try:
        bus.register_channel(s.session_id, "score_events")
    except Exception:
        pass
    # start assembler worker for this session
    _start_assembler_for_session(s.session_id)
    # start scorer loop for this session
    _start_scorer_for_session(s.session_id)
    token = create_session_token(s.session_id, ttl_sec=60 * 60)
    return JSONResponse({"session_id": s.session_id, "token": token, "status": s.state})


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    s = store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": s.session_id, "state": s.state, "created_at": s.created_at.isoformat()}


@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    # Accept connection, require token query param
    token = websocket.query_params.get("token")
    if not token or verify_session_token(token) != session_id:
        await websocket.close(code=4403)
        return

    await websocket.accept()
    s = store.get(session_id)
    if not s:
        # create if not exist
        s = store.create_session(session_id=session_id)
        # ensure session channels in the bus
        bus._ensure_session(session_id)
        try:
            bus.register_channel(session_id, "score_events")
        except Exception:
            pass
        # start assembler worker for this session
        _start_assembler_for_session(session_id)
        # start scorer worker for this session
        _start_scorer_for_session(session_id)

    await websocket.send_json({"type": "ack", "session_id": session_id, "state": s.state})

    # background batch processor will read pending frames and process them in batches
    async def _batch_processor():
        try:
            while True:
                await asyncio.sleep(0.5)
                pending = store.pop_frames(session_id)
                is_calib_batch = False
                # when session is calibrating, frames may be stored under calib_frames instead
                if not pending:
                    s_lookup = store.get(session_id)
                    if s_lookup and s_lookup.state == "calibrating" and getattr(s_lookup, "calib_frames", None):
                        # pick up calibration frames for processing
                        pending = store.pop_calib_frames(session_id)
                        is_calib_batch = True
                        logger.debug("Processing calib batch for session %s size=%s", session_id, len(pending))
                if not pending:
                    # no pending non-calibration frames to process — log session state for debugging
                    sstate = store.get(session_id)
                    try:
                        logger.debug("no pending frames: session=%s state=%s last_frames=%s calib_frames=%s", session_id, sstate.state if sstate else None, len(sstate.last_frames) if sstate else 0, len(sstate.calib_frames) if sstate else 0)
                    except Exception:
                        logger.debug("no pending frames and couldn't fetch detailed session state for session=%s", session_id)
                    continue
                try:
                    start_ts = asyncio.get_event_loop().time()
                    await websocket.send_json({"type": "pipeline_status", "stage": "start", "count": len(pending)})
                    try:
                        batch_res = await asyncio.wait_for(acoustic_engine.process_batch_async(pending), timeout=STAGE_TIMEOUTS["acoustic"])
                        acoustic_done_ts = asyncio.get_event_loop().time()
                        await websocket.send_json({"type": "pipeline_status", "stage": "acoustic_done", "duration_ms": int((acoustic_done_ts - start_ts) * 1000)})
                    except asyncio.TimeoutError:
                        await websocket.send_json({"type": "pipeline_status", "stage": "error", "stage_name": "acoustic", "error": "timeout"})
                        # skip this batch iteration
                        continue
                    # send compact acoustic batch event
                    await websocket.send_json({
                        "type": "acoustic_batch",
                        "size": len(batch_res),
                        "items": [ {"timestamp_ms": r.timestamp_ms, "acoustic": r.acoustic, "qc": r.qc} for r in batch_res],
                    })

                    # Run STT on the pending batch (off loop) and annotate
                    try:
                        asr_start = asyncio.get_event_loop().time()
                        try:
                            logger.debug("Calling STT adapter for session %s batch_len=%s", session_id, len(pending))
                            asr_res = await asyncio.wait_for(asyncio.to_thread(stt_adapter.transcribe, pending), timeout=STAGE_TIMEOUTS["stt"])
                            logger.debug("STT adapter returned for session %s: raw_len=%s conf=%s", session_id, len(str(asr_res.get('raw') or '')), asr_res.get('confidence'))
                            asr_done = asyncio.get_event_loop().time()
                        except asyncio.TimeoutError:
                            await websocket.send_json({"type": "pipeline_status", "stage": "error", "stage_name": "stt", "error": "timeout"})
                            # don't continue to linguistic if STT timed out
                            continue
                        await websocket.send_json({"type": "pipeline_status", "stage": "asr_done", "duration_ms": int((asr_done - asr_start) * 1000), "asr_conf": asr_res.get("confidence")})
                        # publish STT partial + final events to the bus (best-effort)
                        try:
                            stt_partial_event = {"type": "STTPartial", "session_id": session_id, "timestamp_ms": int(batch_res[0].timestamp_ms if batch_res else 0), "partial": asr_res}
                            await bus.publish(session_id, "stt_q", stt_partial_event, block=False)
                        except Exception:
                            pass

                        # store and emit ASR batch
                        store.add_asr_result(session_id, asr_res)
                        try:
                            stt_final_event = {"type": "STTFinal", "session_id": session_id, "timestamp_ms": int(batch_res[0].timestamp_ms if batch_res else 0), "final": asr_res}
                            await bus.publish(session_id, "stt_q", stt_final_event, block=False)
                        except Exception:
                            pass
                        await websocket.send_json({"type": "asr_batch", "raw": asr_res.get("raw"), "confidence": asr_res.get("confidence"), "words": asr_res.get("words")})

                        # Optionally gate linguistic analysis by ASR confidence
                        try:
                            if float(asr_res.get("confidence", 0.0)) >= MIN_ASR_CONF:
                                logger.debug("ASR confidence %s meets threshold %s — running linguistic analysis", asr_res.get('confidence'), MIN_ASR_CONF)
                                ling_start = asyncio.get_event_loop().time()
                                try:
                                    logger.debug("Starting LinguisticEngine.analyze for session %s", session_id)
                                    ling_res = await asyncio.wait_for(asyncio.to_thread(linguistic_engine.analyze, asr_res, int(batch_res[0].timestamp_ms if batch_res else 0)), timeout=STAGE_TIMEOUTS["linguistic"])
                                    logger.debug("LinguisticEngine.analyze returned for session %s asr_quality=%s", session_id, ling_res.asr_quality)
                                    ling_done = asyncio.get_event_loop().time()
                                except asyncio.TimeoutError:
                                    await websocket.send_json({"type": "pipeline_status", "stage": "error", "stage_name": "linguistic", "error": "timeout"})
                                    # proceed without linguistic outputs
                                    continue
                                await websocket.send_json({"type": "pipeline_status", "stage": "linguistic_done", "duration_ms": int((ling_done - ling_start) * 1000), "asr_quality": ling_res.asr_quality})
                                # Only emit linguistic outputs if the ASR quality is adequate
                                if float(ling_res.asr_quality) >= MIN_ASR_CONF:
                                    # compute baseline if the session is calibrating and we have enough calib frames
                                    s = store.get(session_id)
                                    if s and not s.baseline and len(s.calib_frames) >= 3:
                                        b = store.compute_and_store_baseline(session_id, min_frames=3)
                                        await websocket.send_json({"type": "pipeline_status", "stage": "baseline_computed", "metrics": list(b.keys())})
                                        if b:
                                            try:
                                                from src.voicecred.schemas.events import BaselineReady
                                                import uuid as _uuid
                                                br = BaselineReady(type="BaselineReady", envelope=None, baseline_id=str(_uuid.uuid4()))
                                                await bus.publish(session_id, "ops_events", br.model_dump(), block=False)
                                            except Exception:
                                                pass
                                    # first assemble all feature frames for this batch
                                    feature_frames = []
                                    for r in batch_res:
                                        feat = assemble_feature_frame(session_id, {"acoustic": r.acoustic, "qc": r.qc}, ling_res.linguistic, r.timestamp_ms)
                                        feat.setdefault("normalized", {})
                                        feature_frames.append(feat)

                                    # If this was a calibration batch and no baseline exists yet,
                                    # compute baseline from the assembled feature_frames so we can
                                    # normalize and score these same frames in this pass.
                                    s_before = store.get(session_id)
                                    if is_calib_batch and s_before and not s_before.baseline and len(feature_frames) >= 3:
                                        # temporarily store assembled frames as calib_frames and compute baseline
                                        s_before.calib_frames = list(feature_frames)
                                        b = store.compute_and_store_baseline(session_id, min_frames=3)
                                        if b:
                                            await websocket.send_json({"type": "pipeline_status", "stage": "baseline_computed", "metrics": list(b.keys())})
                                            try:
                                                from src.voicecred.schemas.events import BaselineReady
                                                import uuid as _uuid
                                                br = BaselineReady(type="BaselineReady", envelope=feature_frames[0] if feature_frames else None, baseline_id=str(_uuid.uuid4()))
                                                await bus.publish(session_id, "ops_events", br.model_dump(), block=False)
                                            except Exception:
                                                pass

                                    # now we have baseline (maybe) — compute normalization and scoring
                                    baseline = store.get_baseline(session_id)
                                    for feat in feature_frames:
                                        if baseline:
                                            # compute normalized z-scores for baseline metrics
                                            for m, mstat in baseline.items():
                                                val = scorer._lookup_metric(feat, m)
                                                if val is not None:
                                                    feat["normalized"][m] = baseline_utils.z_score(val, mstat.get("median", 0.0), mstat.get("mad", 0.0))
                                            # compute score
                                            score_result = scorer.compute(feat, baseline)
                                            feat["score"] = float(score_result.score)
                                            feat["ci"] = list(map(float, score_result.ci))
                                            feat["explain"] = score_result.explain
                                            # update per-session EMA & recent buffer
                                            store_s = store.get(session_id)
                                            if store_s:
                                                scorer.update_ema(store_s.scoring_state, score_result.explain.get("raw", 0.0))

                                        # persist frames or keep as calib as originally intended
                                        s2 = store.get(session_id)
                                        if s2 and s2.state == "calibrating":
                                            store.add_calib_frame(session_id, feat)
                                        else:
                                            store.add_feature_frame(session_id, feat)

                                    if feature_frames:
                                        # publish each assembled feature frame into the window buffer channel
                                        for feat in feature_frames:
                                            try:
                                                wind = feat.get("window_id") or str(feat.get("timestamp_ms", ""))
                                                await bus.publish(session_id, "window_buffer", {"type": "feature_frame", "session_id": session_id, "window_id": wind, "feature": feat, "timestamp_ms": feat.get("timestamp_ms")}, block=False)
                                            except Exception:
                                                # best effort — if bus publish fails keep going
                                                pass
                                        await websocket.send_json({"type": "feature_batch", "size": len(feature_frames), "items": feature_frames})

                                    await websocket.send_json({"type": "linguistic_batch", "timestamp_ms": ling_res.timestamp_ms, "linguistic": ling_res.linguistic, "asr_quality": ling_res.asr_quality})
                                    # attempt to run speaker recognition in background and publish segments
                                    try:
                                        if speaker_adapter is not None:
                                            segs = await asyncio.to_thread(speaker_adapter.recognize, asr_res)
                                            # publish speaker segments ready event
                                            try:
                                                ev = {"type": "SpeakerSegmentsReady", "session_id": session_id, "timestamp_ms": int(batch_res[0].timestamp_ms if batch_res else 0), "segments": segs.get("segments")}
                                                await bus.publish(session_id, "speaker_q", ev, block=False)
                                            except Exception:
                                                pass
                                    except Exception:
                                        # best-effort: speaker recognition failures should not fail pipeline
                                        logger.debug("speaker recognition failed for session %s", session_id, exc_info=True)
                            else:
                                # ASR confidence too low to run linguistic pipeline
                                await websocket.send_json({"type": "pipeline_status", "stage": "linguistic_skipped", "reason": "asr_conf_too_low", "asr_conf": asr_res.get("confidence")})
                        except Exception as e:
                            # emit pipeline error for linguistic step
                            await websocket.send_json({"type": "pipeline_status", "stage": "error", "stage_name": "linguistic", "error": str(e)})
                    except Exception as e:
                        # emit pipeline error for STT step
                        await websocket.send_json({"type": "pipeline_status", "stage": "error", "stage_name": "stt", "error": str(e)})
                except Exception as e:
                    # emit pipeline error for batch (acoustic) and continue
                    await websocket.send_json({"type": "pipeline_status", "stage": "error", "stage_name": "acoustic", "error": str(e)})
                    pass
        except asyncio.CancelledError:
            return

    batch_task = asyncio.create_task(_batch_processor())
    # also create a UI-out forwarder that will stream ui_out queue items to this WS
    async def _ui_forwarder():
        q = bus.subscribe(session_id, "ui_out")
        try:
            while True:
                item = await q.get()
                logger.debug("ui_forwarder session %s got item -> %s", session_id, item)
                try:
                    await websocket.send_json({"type": "ui_diff", "payload": item})
                    logger.debug("ui_forwarder session %s forwarded item to ws", session_id)
                except Exception:
                    # best-effort: ignore send errors
                    logger.debug("ui_forwarder session %s failed to send to ws", session_id, exc_info=True)
                    pass
        except asyncio.CancelledError:
            return

    ui_task = asyncio.create_task(_ui_forwarder())
    logger.debug("batch_task created for session %s", session_id)

    try:
        while True:
            msg = await websocket.receive_json()
            # Expect messages: {"type":"control","cmd":"reset|finalize"} or {"type":"frame","pcm":..., "ts":...}
            if msg.get("type") == "control":
                cmd = msg.get("cmd")
                if cmd == "reset":
                    store.set_state(session_id, "calibrating")
                    await websocket.send_json({"status": "calibrating", "duration_sec": 60})
                elif cmd == "finalize":
                    snapshot = store.finalize(session_id)
                    # stop assembler for this session when finalizing
                    try:
                        _stop_assembler_for_session(session_id)
                        _stop_scorer_for_session(session_id)
                    except Exception:
                        pass
                    await websocket.send_json({"status": "finalized", "snapshot": snapshot})
                    break
                else:
                    await websocket.send_json({"error": "unknown control"})
            elif msg.get("type") == "frame":
                # rate limit and add frame
                if not store.allow_frame(session_id):
                    await websocket.send_json({"error": "rate_limited"})
                    continue
                pcm = msg.get("pcm", [])
                ts = msg.get("ts", 0)
                # store full pcm so background batch processors can use it
                # when session is in calibrating state, treat these frames as calibration frames
                s = store.get(session_id)
                if s and s.state == "calibrating":
                    store.add_calib_frame(session_id, {"timestamp_ms": ts, "len": len(pcm), "pcm": pcm, "acoustic": None, "linguistic": None, "qc": {}})
                else:
                    store.add_frame(session_id, {"timestamp_ms": ts, "len": len(pcm), "pcm": pcm})

                # publish a FrameReceived event to session frames_in (best-effort)
                try:
                    ev = {"type": "FrameReceived", "session_id": session_id, "window_id": None, "timestamp_ms": ts, "pcm_length": len(pcm)}
                    # non-blocking; if queue is full it will be dropped
                    await bus.publish(session_id, "frames_in", ev, block=False)
                except Exception:
                    pass

                # process acoustic features in threadpool and send event back
                try:
                    # run blocking CPU work off the event loop
                    result = await asyncio.to_thread(acoustic_engine.process_frame, pcm, ts)
                    await websocket.send_json({"type": "acoustic", "timestamp_ms": result.timestamp_ms, "acoustic": result.acoustic, "qc": result.qc})
                except Exception:
                    # best-effort: ignore failures during acoustic extraction
                    await websocket.send_json({"type": "frame_ack", "ts": ts})

            else:
                await websocket.send_json({"error": "unknown message"})
    except WebSocketDisconnect:
        logger.info("ws disconnected (socket closed) for session %s", session_id)
    finally:
        # cancel background tasks (non-blocking best-effort)
        try:
            batch_task.cancel()
        except Exception:
            pass
        try:
            ui_task.cancel()
        except Exception:
            pass
        logger.debug("cleaned up background tasks for session %s", session_id)
        return


async def _scorer_loop_for_session(session_id: str):
    # per-session ops_events subscriber — respond to BaselineReady events by
    # computing scores for existing feature_frames and publishing ScoreUpdated
    q = bus.subscribe(session_id, "ops_events")
    # mark this session as ready so callers/tests can be sure the scorer is
    # actually subscribed before publishing operations events (fixes race)
    try:
        scorer_ready.add(session_id)
    except Exception:
        pass
    try:
        from src.voicecred.schemas.events import ScoreUpdated
        while True:
            item = await q.get()
            logger.debug("scorer consumed ops_events item for %s -> %s", session_id, item)
            try:
                if not isinstance(item, dict):
                    continue
                typ = item.get("type")
                if typ != "BaselineReady":
                    continue
                s = store.get(session_id)
                if not s:
                    continue
                baseline = s.baseline
                if not baseline:
                    continue

                # compute and publish score updates for each stored feature_frame
                for feat in list(s.feature_frames):
                    try:
                        score_result = scorer.compute(feat, baseline)
                        scorer.update_ema(s.scoring_state, score_result.explain.get("raw", 0.0))
                        scoring_payload = {
                            "score": float(score_result.score),
                            "ci_lo": float(score_result.ci[0]),
                            "ci_hi": float(score_result.ci[1]),
                            "contributions": {k: float(v) for k, v in score_result.contributions.items()},
                            "explain": score_result.explain,
                        }
                        out = {"type": "ScoreUpdated", "envelope": feat, "scoring": scoring_payload}
                        ok = await bus.publish(session_id, "ui_out", out, block=False)
                        # also publish to a dedicated scoring channel so tests and
                        # other background consumers can inspect completed scoring
                        # events without racing with potential UI-forwarders
                        try:
                            await bus.publish(session_id, "score_events", out, block=False)
                        except Exception:
                            pass
                        logger.debug("scorer published ui_out (out) for %s ok=%s", session_id, ok)
                        # ensure typed event conforms to EnvelopeV1 (fill missing linguistic keys with None)
                        env = dict(feat)
                        ling = env.get("linguistic") or {}
                        env["linguistic"] = {
                            "pronoun_ratio": ling.get("pronoun_ratio"),
                            "article_ratio": ling.get("article_ratio"),
                            "ttr": ling.get("ttr"),
                            "avg_tokens_per_sentence": ling.get("avg_tokens_per_sentence"),
                            "tokens": ling.get("tokens"),
                            "speaking_rate": ling.get("speaking_rate"),
                        }
                        try:
                            evt = ScoreUpdated(type="ScoreUpdated", envelope=env, scoring=scoring_payload)
                            ok2 = await bus.publish(session_id, "ui_out", evt.model_dump(), block=False)
                            try:
                                await bus.publish(session_id, "score_events", evt.model_dump(), block=False)
                            except Exception:
                                pass
                            logger.debug("scorer published ui_out (typed) for %s ok=%s", session_id, ok2)
                        except Exception:
                            logger.debug("failed to build/publish typed ScoreUpdated for %s", session_id, exc_info=True)
                    except Exception:
                        logger.debug("failed scoring frame for %s", session_id, exc_info=True)
            except Exception:
                logger.debug("scorer loop event handling failed for %s", session_id, exc_info=True)
    except asyncio.CancelledError:
        try:
            scorer_ready.discard(session_id)
        except Exception:
            pass
        return
    except Exception:
        # ensure we remove readiness on unexpected exit
        try:
            scorer_ready.discard(session_id)
        except Exception:
            pass
        raise


@app.get("/admin/sessions")
async def admin_list_sessions(req: Request):
    # NOTE: In prod protect with auth
    return {"sessions": store.list()}


@app.post("/admin/sessions/{session_id}/flush")
async def admin_flush_session(session_id: str):
    # stop any running assembler for the session and delete it
    try:
        _stop_assembler_for_session(session_id)
        _stop_scorer_for_session(session_id)
    except Exception:
        pass
    store.delete(session_id)
    return {"ok": True}


@app.post("/admin/sessions/{session_id}/token")
async def admin_issue_token(session_id: str):
    # issue short-lived token for the given session id
    s = store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    token = create_session_token(session_id, ttl_sec=300)
    return {"session_id": session_id, "token": token}


if __name__ == "__main__":
    uvicorn.run("voicecred.main:app", host="0.0.0.0", port=8000, reload=True)