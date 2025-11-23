"""Simple local test: load download.wav, run it through the in-repo pipeline (acoustic -> stt -> linguistic -> assemble -> baseline -> score)

This is intended to let you test the pipeline end-to-end without running the HTTP server.
"""
import sys
import os
import wave

# Ensure local src/ is on PYTHONPATH so this script can be run directly
HERE = os.path.dirname(os.path.dirname(__file__))
# Add repo root so imports referencing `src.voicecred` will resolve
if HERE not in sys.path:
    sys.path.insert(0, HERE)
SRC = os.path.join(HERE, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from voicecred.session import InMemorySessionStore
from voicecred.acoustic import AcousticEngine
from voicecred.stt import MockSTTAdapter, create_stt_adapter, create_speaker_adapter
from voicecred.linguistic import LinguisticEngine
from voicecred.assembler import assemble_feature_frame
from voicecred.scorer import Scorer
from voicecred.utils import baseline as baseline_utils


def read_wav_as_int16(path):
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        nchan = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise RuntimeError(f"Only 16-bit WAV supported (sampwidth={sampwidth})")
    # if stereo, downmix to mono by taking every nchan-th sample
    if nchan == 1:
        pcm_bytes = frames
    else:
        # convert to int16 list and downmix
        import array
        arr = array.array('h')
        arr.frombytes(frames)
        # arr contains interleaved channels; take first channel
        mono = arr[::nchan]
        pcm_bytes = mono.tobytes()
    return pcm_bytes, sr


def chunk_bytes(pcm_bytes, sample_rate, chunk_ms=200):
    samples_per_chunk = int((sample_rate / 1000.0) * chunk_ms)
    bytes_per_sample = 2
    chunk_size = samples_per_chunk * bytes_per_sample
    out = []
    for i in range(0, len(pcm_bytes), chunk_size):
        out.append(pcm_bytes[i:i+chunk_size])
    return out


def main(wav_path="download.wav"):
    if not os.path.exists(wav_path):
        print("WAV not found at", wav_path)
        return

    store = InMemorySessionStore()
    s = store.create_session()
    sid = s.session_id

    acoustic = AcousticEngine()
    # use mock STT by default (works even when models are not installed)
    stt = create_stt_adapter(os.environ.get('STT_ADAPTER', 'mock'))
    ling = LinguisticEngine()
    scorer = Scorer()

    # optional speaker recognition adapter — run diarization in background
    speaker_adapter = create_speaker_adapter(os.environ.get('SPEAKER_ADAPTER', 'mock'))
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=2)
    speaker_future = None
    try:
        # If adapter supports file path recognition, run on full WAV in background
        if speaker_adapter and hasattr(speaker_adapter, 'recognize'):
            speaker_future = executor.submit(lambda: speaker_adapter.recognize(wav_path))
    except Exception:
        speaker_future = None

    pcm_bytes, sr = read_wav_as_int16(wav_path)
    chunks = chunk_bytes(pcm_bytes, sr, chunk_ms=200)

    print(f"Loaded {wav_path}: sample_rate={sr} chunk_count={len(chunks)}")

    frames = []
    ts = 0
    for c in chunks:
        frames.append({"timestamp_ms": ts, "pcm": c})
        ts += 200

    # acoustic processing
    acoustics = [acoustic.process_frame(f['pcm'], f['timestamp_ms']) for f in frames]
    print("Acoustic features extracted (first 3):")
    for a in acoustics[:3]:
        print(a)

    # STT & linguistic: process per-batch windows so linguistic features vary
    feature_frames = []
    batch_size = 4  # group 4 acoustic frames (e.g. 800ms windows) per STT/linguistic pass
    print(f"Running STT + linguistic analysis in batches of {batch_size} frames")
    for i in range(0, len(acoustics), batch_size):
        batch = acoustics[i:i + batch_size]
        if not batch:
            continue
        # pass the corresponding input frames (timestamps and pcm are available in frames list)
        # Map to the original frames slice (same indexes)
        # The STT adapter expects the list-of-frame dicts structure, so supply similar dicts
        frame_slice = frames[i:i + batch_size]
        # If using the mock STT adapter, inject a deterministic per-batch
        # `transcript_override` so the linguistic metrics vary across batches.
        # This prevents the mock's default static transcript from repeating
        # on every batch and yielding identical linguistic stats (which lead
        # to zero z-scores). For real adapters this won't be used.
        if hasattr(stt, '__class__') and stt.__class__.__name__ == 'MockSTTAdapter':
            # create a small pseudo-transcript that varies by batch index
            words = [f"w{(i + j) % 100}" for j in range(3 + (i % 5))]
            # attach an override to the first frame in the slice
            if isinstance(frame_slice, list) and frame_slice:
                frame_slice[0] = dict(frame_slice[0])
                frame_slice[0]["transcript_override"] = " ".join(words)
        asr = stt.transcribe(frame_slice)
        print("ASR batch summary:", {"raw_len": len(asr.get('raw') or ''), "confidence": asr.get('confidence')})
        ling_res = ling.analyze(asr, timestamp_ms=(batch[0].timestamp_ms if batch else 0))
        print("Linguistic keys batch:", list(ling_res.linguistic.keys()))

        # if background speaker diarization has finished, fetch results to attach
        spk_res = None
        if speaker_future and speaker_future.done():
            try:
                spk_all = speaker_future.result()
                spk_res = spk_all.get('segments') if isinstance(spk_all, dict) else None
            except Exception:
                spk_res = None

        for r in batch:
            qc = dict(r.qc) if isinstance(r.qc, dict) else {}
            if spk_res:
                qc['speaker_segments'] = spk_res
            feat = assemble_feature_frame(sid, {"acoustic": r.acoustic, "qc": qc}, ling_res.linguistic, r.timestamp_ms)
            feat.setdefault('normalized', {})
            feature_frames.append(feat)

    print(f"Assembled {len(feature_frames)} feature frames. Computing baseline (if enough frames)...")
    # compute baseline if enough
    if len(feature_frames) >= 3:
        s.calib_frames = feature_frames
        b = store.compute_and_store_baseline(sid, min_frames=3)
        print("Baseline metrics count:", len(b))
    else:
        b = {}
        print("Not enough frames for baseline; collected", len(feature_frames))

    # score frames
    scored = []
    for feat in feature_frames:
        if b:
            for m, mstat in b.items():
                val = scorer._lookup_metric(feat, m)
                if val is not None:
                    feat['normalized'][m] = baseline_utils.z_score(val, mstat.get('median', 0.0), mstat.get('mad', 0.0))
            scres = scorer.compute(feat, b)
            feat['score'] = float(scres.score)
            feat['ci'] = [float(x) for x in scres.ci]
            feat['explain'] = scres.explain
        scored.append(feat)

    print("Scored frames (first 3):")
    import json
    print(json.dumps(scored[:3], indent=2))


if __name__ == '__main__':
    main('download.wav')
