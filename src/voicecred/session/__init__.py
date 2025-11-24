from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Dict, Optional, List, Any
import math
import uuid
from src.voicecred.utils.logger_util import get_logger, logging
logger = get_logger(__name__, logging.DEBUG)

# Re-export session helpers and state models from this package
from .state import SessionStates, can_transition
from .baseline import compute_baseline_from_frames


@dataclass
class SessionState:
    session_id: str
    created_at: datetime
    last_seen: datetime
    state: str = "idle"  # idle|calibrating|scoring|finalized
    metadata: Dict[str, Any] = field(default_factory=dict)
    calib_frames: List[Dict] = field(default_factory=list)
    last_frames: List[Dict] = field(default_factory=list)
    # Store ASR results and assembled feature frames
    asr_results: List[Dict] = field(default_factory=list)
    feature_frames: List[Dict] = field(default_factory=list)
    transcripts: List[Dict] = field(default_factory=list)
    linguistic_frames: List[Dict] = field(default_factory=list)
    # baseline and scoring state
    baseline: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    scoring_state: Dict[str, Any] = field(default_factory=lambda: {
        "ema_alpha": 0.2,
        "ema_last": None,
        "recent_scores": [],
        "score_window": 20,
    })
    # opt-in flag: when True the session's baseline may be persisted in the
    # store's persistent_baselines (opt-in only; default False for privacy)
    persist_baseline: bool = False
    # calibration policy: 'test'| 'production' - controls default thresholds
    calibration_policy: str = 'test'


class InMemorySessionStore:
    """Simple in-process session store keyed by session_id.

    Not durable — for dev/prototype only. Sessions kept in a dict.
    """

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
        # opt-in persistent baseline storage (session_id -> baseline dict)
        self.persistent_baselines: Dict[str, Dict[str, Dict]] = {}
        # simple per-session rate limit state: {session_id: {"last_tokens": int, "last_ts": float}}
        self._rate_state: Dict[str, Dict] = {}

    def create_session(self, session_id: Optional[str] = None) -> SessionState:
        sid = session_id or str(uuid.uuid4())
        # use timezone-aware UTC datetimes to avoid future deprecation warnings
        now = datetime.now(timezone.utc)
        s = SessionState(session_id=sid, created_at=now, last_seen=now)
        self._sessions[sid] = s
        return s

    def get(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def update_last_seen(self, session_id: str):
        s = self.get(session_id)
        if s:
            s.last_seen = datetime.now(timezone.utc)

    def set_state(self, session_id: str, state: str):
        s = self.get(session_id)
        if s:
            s.state = state

    def add_calib_frame(self, session_id: str, frame: Dict):
        s = self.get(session_id)
        if s:
            # provide helpful trace for debugging when calibration frames are stored
            try:
                import logging
                logging.getLogger(__name__).debug("add_calib_frame: session=%s frame_keys=%s", session_id, list(frame.keys()))
            except Exception:
                pass
            s.calib_frames.append(frame)

    # We keep the existing in-store compute_and_store_baseline behavior but
    # delegate the core numeric baseline computation to compute_baseline_from_frames
    def compute_and_store_baseline(self, session_id: str, min_frames: int = 3, *, use_production_policy: bool = False, min_voiced_seconds: float | None = None) -> Dict[str, Dict]:
        s = self.get(session_id)
        if not s:
            return {}
        frames = s.calib_frames
        from src.voicecred.session.baseline import compute_baseline_from_frames

        total = len(frames)

        # If production policy is enabled, require a minimum total voiced_seconds
        if use_production_policy:
            total_voiced = 0.0
            for f in frames:
                qc = f.get("qc") if isinstance(f, dict) else None
                try:
                    total_voiced += float(qc.get("voiced_seconds", 0.0) or 0.0) if isinstance(qc, dict) else 0.0
                except Exception:
                    pass
            # Not enough voiced seconds to compute a production baseline
            if min_voiced_seconds is not None and total_voiced < float(min_voiced_seconds):
                s.metadata["baseline_filtering"] = {"total_calib_frames": total, "kept": total, "total_voiced_seconds": total_voiced, "insufficient_for_production": True}
                return {}

        # perform QC gating and baseline computation
        # We'll compute a kept_count to expose in metadata for debugging/tests
        kept = 0
        for f in frames:
            qc = f.get("qc") if isinstance(f, dict) else None
            accept = True if not isinstance(qc, dict) else False
            if isinstance(qc, dict):
                try:
                    sr = float(qc.get("speech_ratio", 0.0) or 0.0)
                except Exception:
                    sr = 0.0
                try:
                    vs = float(qc.get("voiced_seconds", 0.0) or 0.0)
                except Exception:
                    vs = 0.0
                if (not math.isnan(sr) and sr >= 0.2) or (not math.isnan(vs) and vs >= 0.05):
                    accept = True
                else:
                    try:
                        snr = float(qc.get("snr_db", float("nan")))
                    except Exception:
                        snr = float("nan")
                    if not math.isnan(snr) and snr >= 0.0:
                        accept = True
                    else:
                        try:
                            words = int(qc.get("words_in_window", 0) or 0)
                        except Exception:
                            words = 0
                        if words > 0:
                            accept = True
            if accept:
                kept += 1

        s.metadata["baseline_filtering"] = {"total_calib_frames": total, "kept": kept}

        baseline = compute_baseline_from_frames(frames, min_frames=min_frames)
        if baseline:
            s.baseline = baseline
            # clear calib frames after baseline computed
            s.calib_frames = []
            # persist baseline when opted in
            if getattr(s, "persist_baseline", False):
                try:
                    self.persistent_baselines[session_id] = dict(s.baseline)
                except Exception:
                    pass
        else:
            # insufficient after QC; record that fact in metadata
            if not baseline:
                s.metadata.setdefault("baseline_filtering", {})["insufficient_after_qc"] = True
            # keep calib frames for now (caller might collect more)
            pass
        return s.baseline

    def get_baseline(self, session_id: str) -> Dict[str, Dict]:
        s = self.get(session_id)
        if not s:
            return {}
        return s.baseline

    def has_valid_baseline(self, session_id: str, metric: str = None) -> bool:
        s = self.get(session_id)
        if not s:
            return False
        if metric is None:
            return bool(s.baseline)
        return metric in s.baseline and s.baseline[metric].get("count", 0) >= 1

    def add_frame(self, session_id: str, frame: Dict):
        s = self.get(session_id)
        if s:
            s.last_frames.append(frame)
            # keep small history
            if len(s.last_frames) > 500:
                s.last_frames = s.last_frames[-400:]

    def add_transcript(self, session_id: str, transcript: Dict):
        s = self.get(session_id)
        if s:
            s.transcripts.append(transcript)

    def add_linguistic(self, session_id: str, ling: Dict):
        s = self.get(session_id)
        if s:
            s.linguistic_frames.append(ling)

    def pop_frames(self, session_id: str) -> List[Dict]:
        s = self.get(session_id)
        if not s:
            return []
        frames = s.last_frames
        try:
            import logging
            logging.getLogger(__name__).debug("pop_frames: session=%s last_frames=%s calib_frames=%s state=%s", session_id, len(s.last_frames), len(s.calib_frames), s.state)
        except Exception:
            pass
        s.last_frames = []
        return frames

    def pop_calib_frames(self, session_id: str, max_n: int | None = None) -> List[Dict]:
        s = self.get(session_id)
        if not s:
            return []
        try:
            import logging
            logging.getLogger(__name__).debug("pop_calib_frames: session=%s last_frames=%s calib_frames=%s state=%s", session_id, len(s.last_frames), len(s.calib_frames), s.state)
        except Exception:
            pass
        if max_n is None:
            out = s.calib_frames
            s.calib_frames = []
            return out
        else:
            out = s.calib_frames[:max_n]
            s.calib_frames = s.calib_frames[max_n:]
            return out

    def add_asr_result(self, session_id: str, asr: Dict):
        s = self.get(session_id)
        if s:
            s.asr_results.append(asr)

    def add_feature_frame(self, session_id: str, frame: Dict):
        s = self.get(session_id)
        if s:
            s.feature_frames.append(frame)

    def _init_rate(self, session_id: str, capacity: int = 30, refill_per_sec: float = 1.0):
        if session_id not in self._rate_state:
            self._rate_state[session_id] = {
                "tokens": capacity,
                "capacity": capacity,
                "refill_per_sec": refill_per_sec,
                "last_ts": datetime.now(timezone.utc).timestamp(),
            }

    def allow_frame(self, session_id: str) -> bool:
        from datetime import datetime

        self._init_rate(session_id)
        rs = self._rate_state[session_id]
        now_ts = datetime.now(timezone.utc).timestamp()
        elapsed = now_ts - rs["last_ts"]
        refill = elapsed * rs["refill_per_sec"]
        rs["tokens"] = min(rs["capacity"], rs["tokens"] + refill)
        rs["last_ts"] = now_ts
        if rs["tokens"] >= 1:
            rs["tokens"] -= 1
            return True
        return False

    def finalize(self, session_id: str) -> Optional[Dict]:
        s = self.get(session_id)
        if not s:
            return None
        s.state = "finalized"
        snapshot = {"session_id": s.session_id, "created_at": s.created_at.isoformat(), "frames": len(s.last_frames), "calib_frames": len(s.calib_frames)}
        return snapshot

    def list(self):
        return list(self._sessions.keys())

    def delete(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]
