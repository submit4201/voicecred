from __future__ import annotations

import asyncio
from typing import Any, Dict, Tuple
import time

from src.voicecred.utils.logger_util import get_logger, logging
logger=get_logger(__name__,logging.DEBUG)
class Channel:
    def __init__(self, maxsize: int = 100):
        self.q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.maxsize = int(maxsize)
        self.dropped = 0
        self.created_at = time.time()

    @property
    def depth(self) -> int:
        return self.q.qsize()

    async def publish(self, item: Any, block: bool = False, timeout: float | None = None) -> bool:
        # Non-blocking publish tries to put without waiting; if full we drop
        try:
            if block:
                await asyncio.wait_for(self.q.put(item), timeout=timeout)
            else:
                self.q.put_nowait(item)
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            return False


class EventBus:
    """Tiny per-session event bus with typed channels and basic metrics.

    Channels are created per-session. Designed to be lightweight for tests and
    prototype wiring. Queues are bounded to support backpressure.
    """

    DEFAULT_CHANNELS = [
        "frames_in",
        "acoustic_q",
        "stt_q",
        "linguistic_q",
        "speaker_q",
        "window_buffer",
        "ui_out",
        "ops_events",
    ]

    def __init__(self, default_maxsize: int = 100):
        self.sessions: Dict[str, Dict[str, Channel]] = {}
        self.default_maxsize = int(default_maxsize)

    def _ensure_session(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {name: Channel(maxsize=self.default_maxsize) for name in self.DEFAULT_CHANNELS}

    def register_channel(self, session_id: str, channel_name: str, maxsize: int | None = None) -> Channel:
        self._ensure_session(session_id)
        if maxsize is None:
            maxsize = self.default_maxsize
        ch = Channel(maxsize=int(maxsize))
        self.sessions[session_id][channel_name] = ch
        return ch

    def subscribe(self, session_id: str, channel_name: str) -> asyncio.Queue:
        self._ensure_session(session_id)
        ch = self.sessions[session_id].get(channel_name)
        if ch is None:
            ch = self.register_channel(session_id, channel_name)
        return ch.q

    async def publish(self, session_id: str, channel_name: str, item: Any, block: bool = False, timeout: float | None = None) -> bool:
        self._ensure_session(session_id)
        ch = self.sessions[session_id].get(channel_name)
        if ch is None:
            ch = self.register_channel(session_id, channel_name)
        return await ch.publish(item, block=block, timeout=timeout)

    def metrics(self, session_id: str) -> Dict[str, Dict[str, int]]:
        """Return simple per-channel metrics for the session."""
        out: Dict[str, Dict[str, int]] = {}
        if session_id not in self.sessions:
            return out
        for name, ch in self.sessions[session_id].items():
            out[name] = {"queue_depth": ch.depth, "dropped": ch.dropped, "maxsize": ch.maxsize}
        return out
