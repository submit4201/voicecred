from __future__ import annotations
from src.voicecred.utils.logger_util import get_logger, logging
logger=get_logger(__name__,logging.DEBUG)
from typing import Any, Dict, Optional
import time
from collections import OrderedDict
from src.voicecred.schemas.envelope import EnvelopeV1
import logging

logger = logging.getLogger(__name__)


class WindowRecord:
    def __init__(self, session_id: str, window_id: str):
        self.session_id = session_id
        self.window_id = window_id
        self.parts: Dict[str, Any] = {}
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.assembled: Optional[EnvelopeV1] = None
        self.revision = 0

    def update_part(self, part_type: str, payload: Any):
        self.parts[part_type] = payload
        self.updated_at = time.time()

    def has_minimum(self) -> bool:
        # require qc and acoustic as minimum to assemble
        qc = self.parts.get("qc")
        acoustic = self.parts.get("feature") or self.parts.get("acoustic")
        return qc is not None and acoustic is not None


class WindowBuffer:
    """In-memory per-session window buffer for partials.

    Stores partials keyed by session_id and window_id. A simple LRU
    eviction policy (max_windows) is implemented to avoid unbounded growth.
    """

    def __init__(self, max_windows: int = 200):
        self.max_windows = int(max_windows)
        # map session_id -> OrderedDict(window_id -> WindowRecord)
        self.store: Dict[str, OrderedDict[str, WindowRecord]] = {}

    def _ensure_session(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = OrderedDict()

    def _evict_if_needed(self, session_id: str):
        d = self.store.get(session_id)
        if not d:
            return
        while len(d) > self.max_windows:
            # pop the oldest
            d.popitem(last=False)

    def add_part(self, session_id: str, window_id: str, part_type: str, payload: Any) -> Dict[str, Any] | None:
        """Add a partial. Returns a dict describing an assembled envelope or a patch.

        The return dict has one of two shapes:
          - {'action': 'assembled', 'envelope': EnvelopeV1}
          - {'action': 'patch', 'window_id': window_id, 'revision': rev, 'changed_fields': [..], 'envelope': EnvelopeV1}

        Returns None if nothing assembled/changed.
        """
        self._ensure_session(session_id)
        sessions = self.store[session_id]

        if window_id not in sessions:
            sessions[window_id] = WindowRecord(session_id, window_id)
        record = sessions[window_id]
        record.update_part(part_type, payload)
        # move to end (most recently updated)
        sessions.move_to_end(window_id)

        result = None

        # if we satisfy minimal requirements and no assembled envelope exists yet -> assemble
        if record.has_minimum() and record.assembled is None:
            # attempt to build EnvelopeV1 from available parts
            try:
                env_dict = self._build_envelope_dict(record)
                env = EnvelopeV1.model_validate(env_dict)
                record.assembled = env
                record.revision = 1
                result = {"action": "assembled", "envelope": env}
            except Exception:
                # invalid assembly; log for debugging
                logger.debug("failed to assemble envelope for %s/%s: %s", session_id, window_id, env_dict if 'env_dict' in locals() else None, exc_info=True)
                result = None

        # If already assembled and we got a new part that changes the envelope: create a patch
        elif record.assembled is not None:
            try:
                new_env_dict = self._build_envelope_dict(record)
                new_env = EnvelopeV1.model_validate(new_env_dict)
                # detect changed fields by shallow comparison
                changed = []
                old = record.assembled.model_dump()
                new = new_env.model_dump()
                for k, v in new.items():
                    if old.get(k) != v:
                        changed.append(k)
                if changed:
                    record.revision += 1
                    record.assembled = new_env
                    result = {"action": "patch", "window_id": window_id, "revision": record.revision, "changed_fields": changed, "envelope": new_env}
            except Exception:
                logger.debug("failed to build/validate patch for %s/%s", session_id, window_id, exc_info=True)
                # ignore invalid updates
                result = None

        # evict if session store too large
        self._evict_if_needed(session_id)
        return result

    def _build_envelope_dict(self, record: WindowRecord) -> Dict[str, Any]:
        # Basic mapping from parts to EnvelopeV1 fields. Conservative by design.
        parts = record.parts
        qc = parts.get("qc") or {}
        # prefer either a 'feature' with acoustic_named or acoustic fields
        feature = parts.get("feature") or {}

        # collect a sensible timestamp (prefer feature payload timestamp)
        ts = int(feature.get("timestamp_ms", parts.get("timestamp_ms", 0) or 0))

        # ensure linguistic payload contains expected keys (pydantic model requires keys present)
        ling = feature.get("linguistic") if isinstance(feature.get("linguistic"), dict) else parts.get("linguistic")
        if isinstance(ling, dict):
            ling_fixed = {
                "pronoun_ratio": ling.get("pronoun_ratio"),
                "article_ratio": ling.get("article_ratio"),
                "ttr": ling.get("ttr"),
                "avg_tokens_per_sentence": ling.get("avg_tokens_per_sentence"),
                "tokens": ling.get("tokens"),
                "speaking_rate": ling.get("speaking_rate"),
            }
        else:
            ling_fixed = None

        scoring = feature.get("scoring") if isinstance(feature.get("scoring"), dict) else None
        if isinstance(scoring, dict):
            scoring_fixed = {
                "score": scoring.get("score"),
                "ci_lo": scoring.get("ci_lo"),
                "ci_hi": scoring.get("ci_hi"),
            }
        else:
            scoring_fixed = None

        # make the assembled timestamp deterministic when feature provides a
        # timestamp_ms — this makes assembly order independent in tests
        assembled_at = ts if ts else int(record.created_at * 1000)

        envelope: Dict[str, Any] = {
            "version": "v1",
            "session_id": record.session_id,
            "window_id": record.window_id,
            "timestamp_ms": ts,
            "acoustic_named": feature.get("acoustic_named") if feature else None,
            "qc": qc,
            "stt": parts.get("stt") or (feature.get("stt") if feature else None),
            "speaker": parts.get("speaker") or (feature.get("speaker") if feature else None),
            "linguistic": ling_fixed,
            "scoring": scoring_fixed,
            "meta": {"assembled_at": int(assembled_at)},
        }
        # ensure required qc keys are present
        # envelope validation will ensure presence of required fields
        return envelope
