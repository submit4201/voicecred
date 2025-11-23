from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Dict, Optional, List, Any
import math
import uuid


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

    def compute_and_store_baseline(self, session_id: str, min_frames: int = 3, *, use_production_policy: bool = False, min_voiced_seconds: float | None = None) -> Dict[str, Dict]:
        """Compute median/MAD baselines across numeric metrics present in calib_frames.

        Returns the baseline dictionary stored on the session. Requires at least min_frames, otherwise returns empty baseline.
        """
        s = self.get(session_id)
        if not s:
            return {}
        frames = s.calib_frames
        # QC gating: filter out frames that are likely silent/low-quality so they
        # don't contaminate baselines. We accept a frame if it meets at least
        # one of these quality checks: speech_ratio >= min_speech_ratio OR
        # voiced_seconds >= min_voiced_seconds. These thresholds are conservative
        # defaults and can be adjusted to be stricter.
        # QC gating thresholds — per-frame acceptance
        min_speech_ratio = 0.2
        min_voiced_seconds = 0.05

        # Production policy defaults — stricter warm-up requirements (longer voiced duration)
        PROD_DEFAULT_MIN_FRAMES = 30
        PROD_DEFAULT_MIN_VOICED_SECONDS = 60.0

        # If caller asked to use a production policy, adjust the minimum frames and voiced-second
        # requirement unless explicitly overridden by the caller arguments.
        if use_production_policy:
            if min_frames is None:
                min_frames = PROD_DEFAULT_MIN_FRAMES
            if min_voiced_seconds is None:
                min_voiced_seconds = PROD_DEFAULT_MIN_VOICED_SECONDS
        filtered_frames = []
        filtered_out_indices = []
        for idx, f in enumerate(frames):
            qc = f.get("qc") if isinstance(f, dict) else None
            accept = True if not isinstance(qc, dict) else False
            if isinstance(qc, dict):
                # don't auto-accept frames just because linguistic fields exist;
                # rely on QC indicators (speech_ratio, voiced_seconds, snr_db)
                # use numeric QC fields to decide whether to accept the frame
                try:
                    sr = float(qc.get("speech_ratio", 0.0))
                except Exception:
                    sr = 0.0
                try:
                    vs = float(qc.get("voiced_seconds", 0.0))
                except Exception:
                    vs = 0.0

                # accept if speech ratio or voiced seconds exceed thresholds
                if (not math.isnan(sr) and sr >= min_speech_ratio) or (not math.isnan(vs) and vs >= min_voiced_seconds):
                    accept = True
                else:
                    # also allow frames with a sufficiently high SNR
                    try:
                        snr = float(qc.get("snr_db", float("nan")))
                    except Exception:
                        snr = float("nan")
                    if not math.isnan(snr) and snr >= 0.0:
                        accept = True

                    # If QC didn't indicate voiced audio but the frame contains
                    # linguistic evidence (e.g. ASR word counts in window),
                    # allow the frame for calibration. This supports cases
                    # where the client provided transcript_override (ASR)
                    # but the PCM buffer was empty (unit tests / integration
                    # scenarios). We prefer words_in_window as a safe signal
                    # of actual speech coming from the ASR/linguistic pipeline.
                    try:
                        words = int(qc.get("words_in_window", 0)) if qc.get("words_in_window", 0) is not None else 0
                    except Exception:
                        words = 0
                    if words > 0:
                        accept = True

            if accept:
                filtered_frames.append(f)
            else:
                filtered_out_indices.append(idx)

        # save metadata about filtering for debugging / tracking
        s.metadata.setdefault("baseline_filtering", {})
        s.metadata["baseline_filtering"]["total_calib_frames"] = len(frames)
        s.metadata["baseline_filtering"]["kept"] = len(filtered_frames)
        s.metadata["baseline_filtering"]["filtered_out_indices"] = filtered_out_indices

        frames = filtered_frames

        # When using a stricter voiced-second policy, allow baselining if the total voiced_seconds
        # across kept frames meets the production voiced duration requirement, or if the number
        # of kept frames meets the min_frames threshold. This gives two ways to qualify: many
        # short voiced frames (count) OR a sufficiently long voiced audio window.
        if use_production_policy and min_voiced_seconds is not None:
            total_voiced = 0.0
            for f in frames:
                qc = f.get("qc") if isinstance(f, dict) else None
                try:
                    vs = float(qc.get("voiced_seconds", 0.0)) if isinstance(qc, dict) else 0.0
                except Exception:
                    vs = 0.0
                if not math.isnan(vs):
                    total_voiced += vs
            enough_voiced = total_voiced >= float(min_voiced_seconds)
            enough_count = len(frames) >= (min_frames or 0)
            if not (enough_voiced or enough_count):
                # not enough quality calibration frames after QC gating
                s.calib_frames = []
                s.metadata.setdefault("baseline_filtering", {})
                s.metadata["baseline_filtering"]["insufficient_after_qc"] = True
                s.metadata["baseline_filtering"]["voiced_seconds_total"] = total_voiced
                s.metadata["baseline_filtering"]["kept"] = len(frames)
                return {}
        else:
            if len(frames) < min_frames:
                # not enough quality calibration frames after QC gating
                s.calib_frames = []
                s.metadata.setdefault("baseline_filtering", {})
                s.metadata["baseline_filtering"]["insufficient_after_qc"] = True
                return {}

        # collect numeric observations per metric path
        # track total observations and number of valid (non-NaN) values per metric
        metrics = {}

        def observe(key: str, value):
            metrics.setdefault(key, {"total": 0, "valid": 0, "vals": []})
            metrics[key]["total"] += 1
            try:
                v = float(value)
            except Exception:
                return
            # skip NaNs from valid vals
            import math
            if math.isnan(v):
                return
            metrics[key]["vals"].append(v)
            metrics[key]["valid"] += 1

        for f in frames:
            # acoustic block
            ac = f.get("acoustic") if isinstance(f, dict) else None
            if isinstance(ac, dict):
                for k, v in ac.items():
                    observe(f"acoustic.{k}", v)
            elif isinstance(ac, list):
                # Acoustic vector indices correspond to known features
                names = ["f0_mean", "f0_median", "f0_std", "rms", "zcr"]
                for idx, v in enumerate(ac):
                    key = names[idx] if idx < len(names) else f"idx_{idx}"
                    observe(f"acoustic.{key}", v)
            # linguistic block
            lg = f.get("linguistic") if isinstance(f, dict) else None
            if isinstance(lg, dict):
                for k, v in lg.items():
                    observe(f"linguistic.{k}", v)
            # derived list of small dicts
            for d in (f.get("derived") or []):
                if isinstance(d, dict):
                    for k, v in d.items():
                        observe(f"derived.{k}", v)
            # qc keys
            qc = f.get("qc") if isinstance(f, dict) else None
            if isinstance(qc, dict):
                for k, v in qc.items():
                    observe(f"qc.{k}", v)

        # compute median & mad for each metric
        from voicecred.utils import baseline as baseline_utils

        skipped = []
        total_metrics = len(metrics)
        for key, meta in metrics.items():
            vals = meta.get("vals", [])
            valid = meta.get("valid", 0)
            total = meta.get("total", 0)

            # require at least `min_frames` valid (non-NaN) measurements per metric
            if valid < min_frames:
                skipped.append(key)
                try:
                    import logging
                    logging.getLogger(__name__).warning("baseline skip: session=%s metric=%s valid=%s total=%s", session_id, key, valid, total)
                except Exception:
                    pass
                continue

            try:
                median, mad = baseline_utils.compute_median_mad(vals)
            except ValueError:
                skipped.append(key)
                continue

            s.baseline[key] = {"median": median, "mad": mad, "count": valid}
            try:
                import logging
                logging.getLogger(__name__).info("baseline computed: session=%s metric=%s median=%s mad=%s count=%s", session_id, key, median, mad, len(vals))
            except Exception:
                pass

        # record skipped metrics and set inconclusive flag when many metrics were skipped
        s.metadata["baseline_skipped_metrics"] = skipped
        if total_metrics > 0 and (len(skipped) / float(total_metrics)) >= 0.5:
            s.metadata["baseline_inconclusive"] = True
        else:
            s.metadata.pop("baseline_inconclusive", None)

        # persist baseline if session opted in
        if getattr(s, "persist_baseline", False):
            try:
                # store a shallow copy into the persistent_baselines map keyed by session id
                self.persistent_baselines[session_id] = dict(s.baseline)
                s.metadata.setdefault("baseline_filtering", {})
                s.metadata["baseline_filtering"]["persisted"] = True
            except Exception:
                pass

        # clear calib frames after baseline computed
        s.calib_frames = []
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

    # Persistence of assembled feature_frames is handled by add_feature_frame
    # (below) which writes into SessionState.feature_frames. The older
    # metadata-backed snapshot was removed to avoid confusion / duplicate
    # storage.

    def pop_frames(self, session_id: str) -> List[Dict]:
        """Pop and return all pending frames for a session.

        This is intended for batch processing consumer logic.
        """
        s = self.get(session_id)
        if not s:
            return []
        frames = s.last_frames
        # debugging: log counts of last_frames vs calib_frames so batch processor can reason about no-op
        try:
            import logging
            logging.getLogger(__name__).debug("pop_frames: session=%s last_frames=%s calib_frames=%s state=%s", session_id, len(s.last_frames), len(s.calib_frames), s.state)
        except Exception:
            pass
        s.last_frames = []
        return frames

    def pop_calib_frames(self, session_id: str, max_n: int | None = None) -> List[Dict]:
        """Pop and return calibration frames for a session.

        If max_n is provided, pop up to max_n frames; otherwise pop all.
        """
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

    # --- naive per-session rate limiting (token-bucket) ---
    def _init_rate(self, session_id: str, capacity: int = 30, refill_per_sec: float = 1.0):
        if session_id not in self._rate_state:
            self._rate_state[session_id] = {
                "tokens": capacity,
                "capacity": capacity,
                "refill_per_sec": refill_per_sec,
                "last_ts": datetime.now(timezone.utc).timestamp(),
            }

    def allow_frame(self, session_id: str) -> bool:
        """Return True if session can accept a frame under rate limit."""
        from datetime import datetime

        self._init_rate(session_id)
        rs = self._rate_state[session_id]
        now_ts = datetime.now(timezone.utc).timestamp()
        elapsed = now_ts - rs["last_ts"]
        # refill
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
        # snapshot object to return
        snapshot = {
            "session_id": s.session_id,
            "created_at": s.created_at.isoformat(),
            "frames": len(s.last_frames),
            "calib_frames": len(s.calib_frames),
        }
        return snapshot

    def list(self):
        return list(self._sessions.keys())

    def delete(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]
