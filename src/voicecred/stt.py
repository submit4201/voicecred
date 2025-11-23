from __future__ import annotations

from typing import List, Dict, Any, Protocol
import logging
import time

logger = logging.getLogger(__name__)


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
        """Transcribe the given frames.

        Behavior:
        - If any frame dict contains a 'transcript_override' key, return that transcript (deterministic for tests).
        - If whisper model is available and frames is a file path or bytes, attempt to transcribe via whisper.
        - Otherwise raise NotImplementedError when no override and no model is available.
        """
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
            # whisper.load_model returns an object with .transcribe() accepting either file path or numpy audio
            # Accept bytes -> write to temp file; accept str path -> pass-through
            import tempfile, os

            # helper to call model.transcribe safely
            def _call_model(input_source):
                logger.debug("WhisperSTTAdapter calling model.transcribe on %s", type(input_source))
                try:
                    out = self._model.transcribe(input_source)
                    # whisper returns dict with 'text' key
                    raw = out.get("text") if isinstance(out, dict) else str(out)
                    # very minimal tokenization -> words with equal confidences
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
                # write bytes to temp wav file and call model.transcribe
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

            # We don't yet support raw PCM list / numpy arrays here
            logger.error("WhisperSTTAdapter.transcribe: unsupported frames type for model (%s) and no override", type(frames))
            raise NotImplementedError("WhisperSTTAdapter requires either a file path/bytes or transcript_override when whisper isn't configured to accept raw arrays")


def create_stt_adapter(name: str | None = None, **kwargs):
    """Factory helper to create named STT adapters.

    Recognized names (case-insensitive): 'mock', 'whisper'
    If name is None or 'mock', returns MockSTTAdapter.
    """
    n = (name or "mock").strip().lower()
    if n in ("mock", "none"):
        return MockSTTAdapter(**kwargs)
    if n in ("whisper", "whisperstt"):
        return WhisperSTTAdapter(**kwargs)
    if n in ("remote", "http", "docker"):
        return RemoteSTTAdapter(**kwargs)
    raise ValueError(f"Unknown STT adapter name: {name}")


class RemoteSTTAdapter:
    """Adapter that sends frames to a remote HTTP endpoint (eg. a Docker-hosted LLM/STT service).

    The remote endpoint is expected to accept JSON with a `frames` list and return a
    JSON object compatible with MockSTTAdapter (keys: words, confidence, raw).

    Example constructor args:
      RemoteSTTAdapter(endpoint='http://localhost:5000/transcribe')

    If `requests` isn't available or the endpoint isn't reachable, this adapter will
    raise a RuntimeError when transcribe() is called.
    """

    def __init__(self, endpoint: str | None = None, default_confidence: float = 0.9, timeout: int = 10):
        import os
        self.endpoint = endpoint or os.environ.get("REMOTE_STT_URL")
        self.default_confidence = float(default_confidence)
        self.timeout = int(timeout)

    def transcribe(self, frames: list | bytes | None = None) -> dict:
        if not self.endpoint:
            raise RuntimeError("RemoteSTTAdapter.endpoint not configured; set endpoint or REMOTE_STT_URL")
        # try to import requests lazily
        try:
            import requests
        except Exception:
            raise RuntimeError("requests library required for RemoteSTTAdapter")

        # prepare payload. For convenience we send simplified frames structure
        payload = {"frames": []}
        if isinstance(frames, list):
            # include timestamps and any transcript_override
            for f in frames:
                if isinstance(f, dict):
                    payload["frames"].append({"timestamp_ms": f.get("timestamp_ms"), "transcript_override": f.get("transcript_override")})
                else:
                    payload["frames"].append({"timestamp_ms": None})
        elif isinstance(frames, (bytes, bytearray)):
            # send raw bytes as base64 to remote if desired
            import base64
            payload = {"audio_b64": base64.b64encode(bytes(frames)).decode("ascii")}

        resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        if not resp.ok:
            raise RuntimeError(f"Remote STT call failed: {resp.status_code} {resp.text}")
        data = resp.json()
        # ensure minimal keys
        if "words" not in data:
            # be permissive — if remote returns plain text, convert
            raw = data.get("text") or data.get("raw") or str(data)
            words = []
            ts = 0
            for w in str(raw).split():
                dur = 200
                words.append({"word": w, "start_ms": ts, "end_ms": ts + dur, "confidence": self.default_confidence})
                ts += dur
            return {"words": words, "confidence": float(data.get("confidence", self.default_confidence)), "raw": raw}
        return {"words": data.get("words", []), "confidence": float(data.get("confidence", self.default_confidence)), "raw": data.get("raw", data.get("text", ""))}


class SpeakerRecognitionAdapter(Protocol):
    """Interface for speaker recognition / diarization adapters.

    Implementations should provide a .recognize(frames_or_audio) method returning
    a dict with at least `segments`: [{'speaker': 'spk1', 'start_ms': int, 'end_ms': int}, ...]
    """


class MockSpeakerRecognitionAdapter:
    """Simple mock speaker recognizer for local testing.

    This returns a deterministic alternating speaker assignment based on word indices
    (or falling back to chunk index). It's purely for testing visualization and
    will never be used in production.
    """

    def __init__(self, speakers: list | None = None):
        self.speakers = list(speakers or ["spk_A", "spk_B"])[:2]

    def recognize(self, asr_result: dict | list | None = None) -> dict:
        # If asr_result contains words, assign speaker to words based on index
        segs = []
        if isinstance(asr_result, dict) and isinstance(asr_result.get("words"), list):
            words = asr_result.get("words")
            ts = 0
            for idx, w in enumerate(words):
                sp = self.speakers[idx % len(self.speakers)]
                start = w.get("start_ms", ts)
                end = w.get("end_ms", start + 200)
                segs.append({"speaker": sp, "start_ms": int(start), "end_ms": int(end)})
                ts = end
        else:
            # fallback: one segment covering entire sample
            segs.append({"speaker": self.speakers[0], "start_ms": 0, "end_ms": 1000})
        return {"segments": segs}
import os
from pyannote.audio import Pipeline
class PyannoteSpeakerAdapter:
    """Placeholder adapter that uses pyannote.audio if available.

    If pyannote is not installed, this adapter raises RuntimeError on recognize().
    """

    def __init__(self, model: str | None = None, revision: str | None = "main", token: str | None = None):
        # api key setup can be done via env var: export HUGGINGFACE_HUB_TOKEN='your_token_here'
        # for more info see: https://huggingface.co/docs/huggingface_hub/security-tokens
        # 
        self.model = model or "pyannote/speaker-diarization"
        # token can be passed directly, or via env var HF_API_KEY / HUGGINGFACE_HUB_TOKEN
        import os
        self.revision = revision
        self.token = token or os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self._pipeline = None
        try:
            from pyannote.audio import Pipeline
            kwargs = {}
            if self.revision is not None:
                kwargs["revision"] = self.revision
            if self.token:
                kwargs["token"] = self.token
            self._pipeline = Pipeline.from_pretrained(self.model, **kwargs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._pipeline = None
        print("Initializing PyannoteSpeakerAdapter with model:", self.model)
        print("Loading pyannote.audio pipeline...", self._pipeline)
        print("successfully initialized PyannoteSpeakerAdapter")


    def recognize(self, audio_path_or_frames) -> dict:
        if self._pipeline is None:
            raise RuntimeError("pyannote.audio pipeline not initialized — check installation, access and HF token/permissions")

        in_arg = audio_path_or_frames
        try:
            from typing import Any
            import torch
            if isinstance(in_arg, dict) and "waveform" in in_arg:
                # Handle both numpy arrays and torch.Tensors
                wf = in_arg["waveform"]
                # If it's already a torch Tensor, ensure dtype and shape are acceptable
                if isinstance(wf, torch.Tensor):
                    # expected shape (channels, time) and dtype float32
                    if wf.ndim == 1:
                        tensor = wf.unsqueeze(0)
                    elif wf.ndim == 2:
                        tensor = wf
                    else:
                        # If shape unexpected, try to transpose the last two dims
                        tensor = wf.permute(1, 0)
                    if tensor.dtype != torch.float32:
                        tensor = tensor.to(torch.float32)
                else:
                    import numpy as np
                    if isinstance(wf, np.ndarray):
                        if wf.ndim == 1:
                            tensor = torch.from_numpy(wf).unsqueeze(0)
                        elif wf.ndim == 2:
                            # soundfile and many readers return shape (time, channels)
                            # while pyannote expects (channels, time). If the first
                            # dimension is larger than the second, assume shape is
                            # (time, channels) and transpose.
                            if wf.shape[0] >= wf.shape[1]:
                                tensor = torch.from_numpy(wf.T)
                            else:
                                tensor = torch.from_numpy(wf)
                        else:
                            # best-effort conversion for unexpected shapes
                            tensor = torch.from_numpy(wf).unsqueeze(0)
                    else:
                        # leave unknown types for pipeline to handle
                        tensor = wf
                    # ensure float32 (models expect float32 tensors)
                    try:
                        tensor = tensor.to(torch.float32)
                    except Exception:
                        pass
                wf = in_arg["waveform"]
                import numpy as np
                if isinstance(wf, np.ndarray):
                    if wf.ndim == 1:
                        tensor = torch.from_numpy(wf).unsqueeze(0)
                    elif wf.ndim == 2:
                        # soundfile and many readers return shape (time, channels)
                        # while pyannote expects (channels, time). If the first
                        # dimension is larger than the second, assume shape is
                        # (time, channels) and transpose.
                        if wf.shape[0] >= wf.shape[1]:
                            tensor = torch.from_numpy(wf.T)
                        else:
                            tensor = torch.from_numpy(wf)
                    # ensure float32 (models expect float32 tensors)
                    try:
                        tensor = tensor.to(torch.float32)
                    except Exception:
                        pass
                    in_arg = {"waveform": tensor, "sample_rate": int(in_arg.get("sample_rate", 16000))}
        except Exception:
            pass

        # call pipeline with the audio argument only
        out = self._pipeline(in_arg)
        segs = []
        # pyannote.audio may return different output shapes depending on version:
        # - older versions: out.itertracks(yield_label=True)
        # - newer versions: out is DiarizeOutput and segments are in out.speaker_diarization
        if hasattr(out, 'itertracks'):
            for turn, _, speaker in out.itertracks(yield_label=True):
                segs.append({"speaker": speaker, "start_ms": int(turn.start * 1000), "end_ms": int(turn.end * 1000)})
        elif hasattr(out, 'speaker_diarization'):
            ann = out.speaker_diarization
            if hasattr(ann, 'itertracks'):
                for turn, _, speaker in ann.itertracks(yield_label=True):
                    segs.append({"speaker": speaker, "start_ms": int(turn.start * 1000), "end_ms": int(turn.end * 1000)})
            elif hasattr(ann, 'itersegments'):
                # itersegments doesn't support yield_label on some versions; fallback to itersegments and map labels
                for seg in ann.itersegments():
                    label = ann.label(seg) if hasattr(ann, 'label') else None
                    segs.append({"speaker": label, "start_ms": int(seg.start * 1000), "end_ms": int(seg.end * 1000)})
        return {"segments": segs}






def create_speaker_adapter(name: str | None = None, **kwargs):
    n = (name or "mock").strip().lower()
    if n in ("mock", "none"):
        return MockSpeakerRecognitionAdapter(**kwargs)
    if n in ("pyannote", "pyannote-audio"):
        return PyannoteSpeakerAdapter(**kwargs)
    raise ValueError(f"Unknown speaker recognition adapter name: {name}")
