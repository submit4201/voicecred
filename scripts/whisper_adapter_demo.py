"""
Simple demo for the WhisperSTTAdapter / STT adapter factory.

This script shows how to select the STT adapter via the STT_ADAPTER env var
and call transcribe. For local GPU usage, install OpenAI whisper or a
local Whisper implementation and set STT_ADAPTER=whisper.

Examples:
  STT_ADAPTER=mock python scripts/whisper_adapter_demo.py
  STT_ADAPTER=whisper python scripts/whisper_adapter_demo.py

Note: the 'whisper' adapter will attempt to import the `whisper` package and
load a small model; if whisper is not available this demo will show how the
adapter falls back to raising NotImplementedError for non-overrides.
"""
import os
from voicecred.stt import create_stt_adapter

adapter_name = os.environ.get("STT_ADAPTER", "mock")
print(f"Creating STT adapter for: {adapter_name}")
stt = create_stt_adapter(adapter_name)

# Demo: prefer to use transcript_override so local env without models still works
frames = [{"timestamp_ms": 0, "pcm": [], "transcript_override": "demo override transcript"}]
print("Calling transcribe with transcript_override...")
print(stt.transcribe(frames))

# If you have whisper installed and want to try a path/bytes input, pass a file path or wav bytes
# e.g. stt.transcribe("/path/to/file.wav")
