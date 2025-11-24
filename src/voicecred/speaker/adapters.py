from __future__ import annotations

from typing import List, Dict, Protocol
from src.voicecred.utils.logger_util import get_logger, logging
logger=get_logger(__name__,logging.DEBUG)

class SpeakerRecognitionAdapter(Protocol):
    def recognize(self, frames_or_audio) -> Dict:
        ...


class MockSpeakerRecognitionAdapter:
    """Simple mock speaker recognizer for local testing.

    This returns a deterministic alternating speaker assignment based on word indices
    (or falling back to chunk index).
    """

    def __init__(self, speakers: List[str] | None = None):
        self.speakers = list(speakers or ["spk_A", "spk_B"])[:2]

    def recognize(self, asr_result: dict | list | None = None) -> dict:
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
            segs.append({"speaker": self.speakers[0], "start_ms": 0, "end_ms": 1000})
        return {"segments": segs}


class PyannoteSpeakerAdapter:
    """Adapter using pyannote.audio pipeline if available.

    If pyannote isn't installed the adapter will raise at runtime.
    """

    def __init__(self, model: str | None = None, revision: str | None = "main", token: str | None = None):
        import os
        from pyannote.audio import Pipeline

        self.model = model or "pyannote/speaker-diarization"
        self.revision = revision
        self.token = token or os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        kwargs = {}
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.token:
            kwargs["token"] = self.token
        self._pipeline = Pipeline.from_pretrained(self.model, **kwargs)

    def recognize(self, audio_path_or_frames) -> dict:
        if self._pipeline is None:
            raise RuntimeError("pyannote.audio pipeline not initialized — check installation, access and HF token/permissions")

        out = self._pipeline(audio_path_or_frames)
        segs = []
        if hasattr(out, 'itertracks'):
            for turn, _, speaker in out.itertracks(yield_label=True):
                segs.append({"speaker": speaker, "start_ms": int(turn.start * 1000), "end_ms": int(turn.end * 1000)})
        elif hasattr(out, 'speaker_diarization'):
            ann = out.speaker_diarization
            if hasattr(ann, 'itertracks'):
                for turn, _, speaker in ann.itertracks(yield_label=True):
                    segs.append({"speaker": speaker, "start_ms": int(turn.start * 1000), "end_ms": int(turn.end * 1000)})
            elif hasattr(ann, 'itersegments'):
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
