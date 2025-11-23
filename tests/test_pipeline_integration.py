import pytest
import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Ensure repo root is on sys.path so imports like `voicecred.*` succeed
HERE = os.path.dirname(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.voicecred.session import InMemorySessionStore
from src.voicecred.acoustic import AcousticEngine
from src.voicecred.stt import MockSTTAdapter, MockSpeakerRecognitionAdapter
from src.voicecred.linguistic import LinguisticEngine
from src.voicecred.assembler import assemble_feature_frame


def pcm_from_float32(arr: np.ndarray) -> bytes:
    clipped = np.clip(arr, -1.0, 1.0)
    return (np.int16(clipped * 32767)).tobytes()


def test_pipeline_attaches_speaker_segments_background():
    """End-to-end-ish integration test: run mock STT + mock speaker recognition
    where speaker recognition runs in background and its results are attached
    into assembled feature frames in the QC block.
    """
    sr = 16000
    # create 1.0s of audio and split into 200ms frames -> 5 frames
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    sine = 0.2 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

    # split into 200ms chunks
    chunk_ms = 200
    samples_per_chunk = int((sr / 1000.0) * chunk_ms)
    pcm_chunks = []
    for i in range(0, len(sine), samples_per_chunk):
        seg = sine[i:i + samples_per_chunk]
        pcm_chunks.append(pcm_from_float32(seg))

    # create frames list used by the pipeline
    frames = []
    ts = 0
    for c in pcm_chunks:
        frames.append({"timestamp_ms": ts, "pcm": c})
        ts += chunk_ms

    store = InMemorySessionStore()
    s = store.create_session()
    sid = s.session_id

    acoustic = AcousticEngine(sample_rate=sr)
    stt = MockSTTAdapter()
    spk = MockSpeakerRecognitionAdapter(speakers=["S1", "S2"])  # fast mock
    ling = LinguisticEngine()

    # run full-file speaker recognition in background (simulating diarization)
    executor = ThreadPoolExecutor(max_workers=2)
    spk_future = executor.submit(lambda: spk.recognize(frames))

    # emulate per-batch STT + linguistic processing like the script harness
    acoustics = [acoustic.process_frame(f['pcm'], f['timestamp_ms']) for f in frames]

    feature_frames = []
    batch_size = 2
    for i in range(0, len(acoustics), batch_size):
        batch = acoustics[i:i + batch_size]
        frame_slice = frames[i:i + batch_size]

        # inject deterministic transcript override for mock
        if isinstance(frame_slice, list) and frame_slice:
            frame_slice[0] = dict(frame_slice[0])
            frame_slice[0]["transcript_override"] = "a b c d"

        asr = stt.transcribe(frame_slice)
        ling_res = ling.analyze(asr, timestamp_ms=(batch[0].timestamp_ms if batch else 0))

        spk_res = None
        if spk_future.done():
            spk_all = spk_future.result()
            spk_res = spk_all.get('segments') if isinstance(spk_all, dict) else None

        for r in batch:
            qc = dict(r.qc) if isinstance(r.qc, dict) else {}
            if spk_res:
                qc['speaker_segments'] = spk_res
            feat = assemble_feature_frame(sid, {"acoustic": r.acoustic, "qc": qc}, ling_res.linguistic, r.timestamp_ms)
            feature_frames.append(feat)

    # Expect at least one assembled frame to include speaker_segments in QC
    has_speaker = any(isinstance(f.get('qc', {}).get('speaker_segments'), list) and len(f['qc']['speaker_segments']) > 0 for f in feature_frames)
    assert has_speaker, "Expected speaker_segments to be attached to at least one feature frame"
