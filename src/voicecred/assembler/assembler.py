from __future__ import annotations
from src.voicecred.utils.logger_util import get_logger, logging
logger=get_logger(__name__,logging.DEBUG)
import asyncio
from typing import Any, Dict
from src.voicecred.assembler.buffer import WindowBuffer
from src.voicecred.bus.bus import EventBus
from src.voicecred.schemas.events import FeatureFrameAssembled


class Assembler:
    """Assembler worker that consumes window_buffer partials and emits assembled envelopes and diffs.

    This is a small helper intended for in-process integration tests and local execution.
    """

    def __init__(self, bus: EventBus, buffer: WindowBuffer | None = None):
        self.bus = bus
        self.buffer = buffer or WindowBuffer()
        self._tasks: Dict[str, asyncio.Task] = {}

    async def start_for_session(self, session_id: str):
        # subscribe to window_buffer channel and start the processing loop
        q = self.bus.subscribe(session_id, "window_buffer")
        logger.debug("assembler subscribed to window_buffer for session %s -> q=%s", session_id, getattr(q, 'q', q))
        # process messages until cancelled
        try:
            while True:
                item = await q.get()
                logger.debug("assembler got item for session %s: %s", session_id, item)
                # item expected to be a dict with keys: type, window_id, feature/timestamp/stt/speaker/qc
                await self._handle_item(session_id, item)
        except asyncio.CancelledError:
            return

    async def _handle_item(self, session_id: str, item: Dict[str, Any]):
        typ = item.get("type")
        window_id = item.get("window_id") or str(item.get("timestamp_ms"))
        # map incoming item to buffer part_type and payload
        if typ == "feature_frame":
            payload = item.get("feature")
            # include qc and timestamp if present
            res = self.buffer.add_part(session_id, window_id, "feature", {**(payload or {}), "timestamp_ms": item.get("timestamp_ms", 0)})
        elif typ == "stt":
            res = self.buffer.add_part(session_id, window_id, "stt", item.get("stt"))
        elif typ == "speaker":
            res = self.buffer.add_part(session_id, window_id, "speaker", item.get("speaker"))
        elif typ == "qc":
            res = self.buffer.add_part(session_id, window_id, "qc", item.get("qc"))
        else:
            # unknown item — ignore
            return

        logger.debug("buffer result for session %s window %s -> %s", session_id, window_id, res)
        if res:
            # publish assembled envelope or patch to ui_out and a typed FeatureFrameAssembled event
            action = res.get("action")
            envelope = res.get("envelope")
            # publish to ui_out channel for clients to receive diffs/assembly results
            out_payload = {
                "action": action,
                "window_id": window_id,
                "revision": getattr(envelope, 'meta', {}).get('revision', None),
                "envelope": envelope.model_dump() if hasattr(envelope, 'model_dump') else envelope
            }
            ok = await self.bus.publish(session_id, "ui_out", out_payload, block=False)
            logger.debug("published ui_out (out_payload) for session %s ok=%s", session_id, ok)

            # send a typed event object on ui_out as well for richer handling
            if action == "assembled":
                evt = FeatureFrameAssembled(type="FeatureFrameAssembled", envelope=envelope)
                ok2 = await self.bus.publish(session_id, "ui_out", evt.model_dump(), block=False)
                logger.debug("published ui_out (typed evt) for session %s ok=%s", session_id, ok2)
