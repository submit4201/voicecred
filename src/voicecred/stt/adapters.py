from __future__ import annotations

from typing import List, Dict, Any, Protocol
import time

from src.voicecred.utils.logger_util import get_logger, logging
logger=get_logger(__name__,logging.DEBUG)

class STTAdapter(Protocol):
    """Pluggable STT adapter interface.

    Implementations should provide a .transcribe(frames) method that returns a
    dictionary with at least: {
       "words": [ {"word": str, "start_ms": int, "end_ms": int, "confidence": float}, ...],
       "confidence": float  # overall ASR confidence in [0,1]
    }
    """


class MockSTTAdapter:
    """Simple deterministic mock adapter used in tests and local dev.

    If input is frames (list) and contains a key `transcript_override` returns that
    transcript; otherwise returns a short canned transcript.
    """

    def __init__(self, override: str | None = None, default_confidence: float = 0.95):
        self.override = override
        self.default_confidence = float(default_confidence)

    def transcribe(self, frames: List[Dict[str, Any]] | bytes | None = None) -> Dict[str, Any]:
        logger.debug("MockSTTAdapter.transcribe called: frames_type=%s frames_len=%s override=%s", type(frames), (len(frames) if isinstance(frames, list) else 'N/A'), bool(self.override))
        start = time.time()
        transcript = self.override
        if transcript is None and isinstance(frames, list) and frames:
            # look for an override in frame dicts
            for f in frames:
                if isinstance(f, dict) and f.get("transcript_override"):
                    transcript = f.get("transcript_override")
                    break

        if transcript is None:
            transcript = "hello world this is a mock transcript"

        # produce simplistic word timestamps and confidences
        words = []
        ts = 0
        for w in transcript.split():
            dur = 200
            words.append({"word": w, "start_ms": ts, "end_ms": ts + dur, "confidence": self.default_confidence})
            ts += dur

        res = {"words": words, "confidence": self.default_confidence, "raw": transcript}
        logger.debug("MockSTTAdapter.transcribe finished in %.3fms -> words=%s confidence=%s", (time.time()-start)*1000, len(words), res["confidence"])
        return res


class WhisperSTTAdapter:
    """Light-weight placeholder for a production STT adapter (eg. Whisper/VOSK).

    This class is intentionally a skeleton for tests and to demonstrate runtime
    adapter selection. A real implementation would call the model or cloud API
    and return the same dictionary shape as MockSTTAdapter.transcribe().
    """

    def __init__(self, model_name: str | None = None, device: str | None = None, default_confidence: float = 0.9):
        self.model_name = model_name or "small"
        self.device = device
        self.default_confidence = float(default_confidence)

        # try to lazily load OpenAI/whisper if available
        self._model = None
        try:
            import whisper as _whisper

            kwargs = {}
            if self.device:
                kwargs["device"] = self.device
            try:
                self._model = _whisper.load_model(self.model_name, **kwargs)
            except Exception:
                # loading may fail; keep _model as None and allow override-only behavior
                self._model = None
        except Exception:
            # whisper is not installed — keep as None
            self._model = None
        logger.debug("WhisperSTTAdapter init: model=%s device=%s model_loaded=%s", self.model_name, self.device, bool(self._model))

    def transcribe(self, frames: list | bytes | None = None) -> dict:
        logger.debug("WhisperSTTAdapter.transcribe called: frames_type=%s frames_len=%s", type(frames), (len(frames) if isinstance(frames, list) else 'N/A'))
        start = time.time()
        # Check overrides first (useful for tests / debug harnesses)
        transcript = None
        if isinstance(frames, list) and frames:
            for f in frames:
                if isinstance(f, dict) and f.get("transcript_override"):
                    transcript = f.get("transcript_override")
                    break

        # If override exists, produce mock-like deterministic output with default_confidence
        if transcript is not None:
            words = []
            ts = 0
            for w in str(transcript).split():
                dur = 200
                words.append({"word": w, "start_ms": ts, "end_ms": ts + dur, "confidence": self.default_confidence})
                ts += dur
            res = {"words": words, "confidence": float(self.default_confidence), "raw": transcript}
            logger.debug("WhisperSTTAdapter.transcribe override returned in %.3fms", (time.time()-start)*1000)
            return res

        # If whisper model loaded, and input is file-like or bytes — attempt to transcribe
        if self._model is not None:
            import tempfile, os

            def _call_model(input_source):
                logger.debug("WhisperSTTAdapter calling model.transcribe on %s", type(input_source))
                try:
                    out = self._model.transcribe(input_source)
                    raw = out.get("text") if isinstance(out, dict) else str(out)
                    words = []
                    ts = 0
                    for w in str(raw).split():
                        dur = 200
                        words.append({"word": w, "start_ms": ts, "end_ms": ts + dur, "confidence": float(self.default_confidence)})
                        ts += dur
                    res = {"words": words, "confidence": float(self.default_confidence), "raw": raw}
                    logger.debug("WhisperSTTAdapter model transcribe finished in %.3fms raw_len=%s", (time.time()-start)*1000, len(str(raw)))
                    return res
                except Exception as e:
                    raise RuntimeError(f"whisper transcribe failed: {e}")

            if isinstance(frames, (bytes, bytearray)):
                fd, path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                try:
                    with open(path, "wb") as f:
                        f.write(frames)
                    return _call_model(path)
                finally:
                    try:
                        os.remove(path)
                    except Exception:
                        pass

            if isinstance(frames, str):
                logger.debug("WhisperSTTAdapter.transcribe calling model on path string")
                return _call_model(frames)

            logger.error("WhisperSTTAdapter.transcribe: unsupported frames type for model (%s) and no override", type(frames))
            raise NotImplementedError("WhisperSTTAdapter requires either a file path/bytes or transcript_override when whisper isn't configured to accept raw arrays")


class RemoteSTTAdapter:
    """Adapter that sends frames to a remote HTTP endpoint (eg. a Docker-hosted LLM/STT service).

    The remote endpoint is expected to accept JSON with a `frames` list and return a
    JSON object compatible with MockSTTAdapter (keys: words, confidence, raw).
    """

    def __init__(self, endpoint: str | None = None, default_confidence: float = 0.9, timeout: int = 10):
        import os
        self.endpoint = endpoint or os.environ.get("REMOTE_STT_URL")
        self.default_confidence = float(default_confidence)
        self.timeout = int(timeout)

    def transcribe(self, frames: list | bytes | None = None) -> dict:
        if not self.endpoint:
            raise RuntimeError("RemoteSTTAdapter.endpoint not configured; set endpoint or REMOTE_STT_URL")
        try:
            import requests
        except Exception:
            raise RuntimeError("requests library required for RemoteSTTAdapter")

        payload = {"frames": []}
        if isinstance(frames, list):
            for f in frames:
                if isinstance(f, dict):
                    payload["frames"].append({"timestamp_ms": f.get("timestamp_ms"), "transcript_override": f.get("transcript_override")})
                else:
                    payload["frames"].append({"timestamp_ms": None})
        elif isinstance(frames, (bytes, bytearray)):
            import base64
            payload = {"audio_b64": base64.b64encode(bytes(frames)).decode("ascii")}

        resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        if not resp.ok:
            raise RuntimeError(f"Remote STT call failed: {resp.status_code} {resp.text}")
        data = resp.json()
        if "words" not in data:
            raw = data.get("text") or data.get("raw") or str(data)
            words = []
            ts = 0
            for w in str(raw).split():
                dur = 200
                words.append({"word": w, "start_ms": ts, "end_ms": ts + dur, "confidence": self.default_confidence})
                ts += dur
            return {"words": words, "confidence": float(data.get("confidence", self.default_confidence)), "raw": raw}
        return {"words": data.get("words", []), "confidence": float(data.get("confidence", self.default_confidence)), "raw": data.get("raw", data.get("text", ""))}


def create_stt_adapter(name: str | None = None, **kwargs):
    n = (name or "mock").strip().lower()
    if n in ("mock", "none"):
        return MockSTTAdapter(**kwargs)
    if n in ("whisper", "whisperstt"):
        return WhisperSTTAdapter(**kwargs)
    if n in ("remote", "http", "docker"):
        return RemoteSTTAdapter(**kwargs)
    raise ValueError(f"Unknown STT adapter name: {name}")
