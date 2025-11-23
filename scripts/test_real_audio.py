"""Simple local test: load download.wav, run it through the in-repo pipeline (acoustic -> stt -> linguistic -> assemble -> baseline -> score)

This is intended to let you test the pipeline end-to-end without running the HTTP server.
"""
import sys
import os
import wave

# Ensure local src/ is on PYTHONPATH so this script can be run directly
HERE = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(HERE, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from src.voicecred.session import InMemorySessionStore
from src.voicecred.acoustic import AcousticEngine
from src.voicecred.stt import MockSTTAdapter, create_stt_adapter
from src.voicecred.linguistic import LinguisticEngine
from src.voicecred.assembler import assemble_feature_frame
from src.voicecred.scorer import Scorer
from src.voicecred.utils import baseline as baseline_utils


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

    # STT: pass frames to adapter
    asr = stt.transcribe(frames)
    print("ASR result summary:", {"raw_len": len(asr.get('raw') or ''), "confidence": asr.get('confidence')})

    # linguistic
    ling_res = ling.analyze(asr, timestamp_ms=acoustics[0].timestamp_ms if acoustics else 0)
    print("Linguistic results keys:", list(ling_res.linguistic.keys()))

    # assemble feature frames
    feature_frames = []
    for r in acoustics:
        feat = assemble_feature_frame(sid, {"acoustic": r.acoustic, "qc": r.qc}, ling_res.linguistic, r.timestamp_ms)
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
