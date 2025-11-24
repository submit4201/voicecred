from .adapters import STTAdapter, MockSTTAdapter, WhisperSTTAdapter, RemoteSTTAdapter, create_stt_adapter
# keep speaker adapters available via old import path (voicecred.stt.*) for backwards compatibility
from src.voicecred.speaker.adapters import MockSpeakerRecognitionAdapter, PyannoteSpeakerAdapter, create_speaker_adapter

__all__ = ["STTAdapter", "MockSTTAdapter", "WhisperSTTAdapter", "RemoteSTTAdapter", "create_stt_adapter", "MockSpeakerRecognitionAdapter", "PyannoteSpeakerAdapter", "create_speaker_adapter"]
