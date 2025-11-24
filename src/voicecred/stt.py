"""
Backward compatibility shim: keep import paths stable while the real
implementations live in src.voicecred.stt.adapters
"""

from .stt.adapters import (
    STTAdapter,
    MockSTTAdapter,
    WhisperSTTAdapter,
    RemoteSTTAdapter,
    create_stt_adapter,
)

__all__ = [
    "STTAdapter",
    "MockSTTAdapter",
    "WhisperSTTAdapter",
    "RemoteSTTAdapter",
    "create_stt_adapter",
]

# Backwards-compatible speaker adapters exported from the old stt module path
from src.voicecred.speaker.adapters import MockSpeakerRecognitionAdapter, PyannoteSpeakerAdapter, create_speaker_adapter
__all__.extend(["MockSpeakerRecognitionAdapter", "PyannoteSpeakerAdapter", "create_speaker_adapter"])
