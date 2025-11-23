"""Small evaluation harness to run a short end-to-end pipeline on synthetic inputs.

This script is intentionally lightweight and uses in-repo components to produce a
trace of scored frames for quick manual checks and unit tests.

Usage (pythonic):
    from voicecred.scripts.eval_pipeline import run_demo
    trace = run_demo(n_silent=2, n_voiced=2)

The function returns a list of scored frames with fields {timestamp_ms, acoustic, linguistic, derived, normalized, score, ci, explain}
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple
import math

from src.voicecred.session import InMemorySessionStore
from src.voicecred.acoustic import AcousticEngine
from src.voicecred.stt import MockSTTAdapter
from src.voicecred.linguistic import LinguisticEngine
from src.voicecred.assembler import assemble_feature_frame
from src.voicecred.scorer import Scorer
from src.voicecred.utils import baseline as baseline_utils


def make_silence_pcm(length_ms: int = 200, sample_rate: int = 16000) -> List[int]:
    # short silent PCM as int16 zeros (length_ms milliseconds)
    n = int(sample_rate * (length_ms / 1000.0))
    return [0] * n


def make_simple_voiced_pcm(freq_hz: float = 220.0, length_ms: int = 200, sample_rate: int = 16000) -> List[int]:
    # very small synthetic sine wave as int16 list (mono)
    import math
    n = int(sample_rate * (length_ms / 1000.0))
    pcm = []
    amp = 10000
    for i in range(n):
        v = int(amp * math.sin(2.0 * math.pi * freq_hz * (i / sample_rate)))
        pcm.append(v)
    return pcm


def run_demo(n_silent: int = 2, n_voiced: int = 2) -> List[Dict[str, Any]]:
    """Run a quick pipeline: create session -> produce acoustic frames -> ASR/linguistic -> assemble -> baseline -> normalize -> score.

    Returns: list of assembled & scored frames
    """
    store = InMemorySessionStore()
    aengine = AcousticEngine()
    stt = MockSTTAdapter(override=None, default_confidence=0.95)
    ling = LinguisticEngine()
    scorer = Scorer()

    s = store.create_session()
    sid = s.session_id

    frames = []
    ts = 0
    # produce silence frames
    for _ in range(n_silent):
        pcm = make_silence_pcm()
        frm = {"timestamp_ms": ts, "pcm": pcm}
        frames.append(frm)
        ts += 200
    # produce voiced frames
    for i in range(n_voiced):
        pcm = make_simple_voiced_pcm(freq_hz=220.0 + i * 100)
        frm = {"timestamp_ms": ts, "pcm": pcm, "transcript_override": f"voiced {i}"}
        frames.append(frm)
        ts += 200

    scored = []

    # emulate the pipeline for each frame batched
    batch = frames
    # 1) acoustic
    acoustics = []
    for f in batch:
        res = aengine.process_frame(f["pcm"], f["timestamp_ms"])
        acoustics.append(res)

    # 2) run STT on batch
    asr_res = stt.transcribe(batch)

    # 3) linguistic
    ling_res = ling.analyze(asr_res, timestamp_ms=acoustics[0].timestamp_ms if acoustics else 0)

    # 4) assemble frames and add derived/synth fields
    feature_frames = []
    for r in acoustics:
        feat = assemble_feature_frame(sid, {"acoustic": r.acoustic, "qc": r.qc}, ling_res.linguistic, r.timestamp_ms)
        feat.setdefault("normalized", {})
        feature_frames.append(feat)

    # 5) compute baseline from these frames (if we have enough), then normalize & score
    if len(feature_frames) >= 2:
        # temporarily store as calib_frames and compute baseline
        s.calib_frames = list(feature_frames)
        b = store.compute_and_store_baseline(sid, min_frames=2)
    else:
        b = {}

    for feat in feature_frames:
        if b:
            for m, mstat in b.items():
                val = scorer._lookup_metric(feat, m)
                if val is not None:
                    feat["normalized"][m] = baseline_utils.z_score(val, mstat.get("median", 0.0), mstat.get("mad", 0.0))
            scres = scorer.compute(feat, b)
            feat["score"] = float(scres.score)
            feat["ci"] = [float(x) for x in scres.ci]
            feat["explain"] = scres.explain

        scored.append(feat)

    return scored


def _compute_auc_from_scores(scores_and_labels: List[Tuple[float, int]]) -> float:
    """Compute approximate AUC from (prob, label) pairs using ROC trapezoid integration.

    Simple pure-Python implementation (no sklearn dependency). Accepts prob in [0,1].
    """
    if not scores_and_labels:
        return float('nan')

    # sort descending by score
    pairs = sorted(scores_and_labels, key=lambda x: x[0], reverse=True)
    # total positives/negatives
    P = sum(1 for _, l in pairs if l == 1)
    N = sum(1 for _, l in pairs if l == 0)
    if P == 0 or N == 0:
        return float('nan')

    # Build ROC points by sweeping threshold across unique prediction values
    thresholds = sorted(set(p for p, _ in pairs), reverse=True)
    roc = [(0.0, 0.0)]
    for t in thresholds:
        tp = sum(1 for s, l in pairs if s >= t and l == 1)
        fp = sum(1 for s, l in pairs if s >= t and l == 0)
        tpr = tp / float(P)
        fpr = fp / float(N)
        roc.append((fpr, tpr))
    roc.append((1.0, 1.0))

    # integrate using trapezoidal rule sorted by fpr
    roc_sorted = sorted(roc, key=lambda x: x[0])
    auc = 0.0
    for i in range(1, len(roc_sorted)):
        x0, y0 = roc_sorted[i - 1]
        x1, y1 = roc_sorted[i]
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return float(auc)


def _brier_score(probs: List[float], labels: List[int]) -> float:
    if not probs:
        return float('nan')
    n = len(probs)
    total = 0.0
    for p, l in zip(probs, labels):
        total += (p - float(l)) ** 2
    return float(total / n)


def _calibration_curve(probs: List[float], labels: List[int], bins: int = 10) -> List[Tuple[float, float, int]]:
    """Return calibration bins as list of tuples (bin_mid, avg_pred, avg_label_count).

    Uses equal-width bins between 0 and 1.
    """
    if not probs:
        return []
    buck = [[] for _ in range(bins)]
    for p, l in zip(probs, labels):
        # clamp to [0,1]
        pv = min(max(float(p), 0.0), 1.0)
        idx = int(pv * bins)
        if idx == bins:
            idx = bins - 1
        buck[idx].append((pv, l))

    out = []
    for i, items in enumerate(buck):
        mid = (i + 0.5) / bins
        if not items:
            out.append((mid, float('nan'), 0))
            continue
        avg_pred = sum(p for p, _ in items) / len(items)
        avg_label = sum(l for _, l in items) / len(items)
        out.append((mid, avg_pred, avg_label))
    return out


def run_evaluation(n_calib: int = 4, n_good: int = 10, n_bad: int = 10) -> Dict[str, Any]:
    """Run a labeled evaluation using synthetic frames.

    - n_calib: number of calibration (baseline) frames (voiced)
    - n_good: number of positive (good/voiced) evaluation frames
    - n_bad: number of negative (bad/silent) evaluation frames

    Returns a dict with keys: trace (frames), probs, labels, auc, brier, calibration
    """
    # reuse run_demo primitives
    store = InMemorySessionStore()
    aengine = AcousticEngine()
    stt = MockSTTAdapter(override=None, default_confidence=0.95)
    ling = LinguisticEngine()
    scorer = Scorer()

    s = store.create_session()
    sid = s.session_id

    frames = []
    ts = 0
    # calibration frames (voiced)
    for i in range(n_calib):
        pcm = make_simple_voiced_pcm(freq_hz=120.0 + (i * 10))
        frames.append({"pcm": pcm, "timestamp_ms": ts, "transcript_override": f"calib {i}"})
        ts += 200

    eval_frames = []
    labels = []
    # good frames (voiced) -> label 1
    for i in range(n_good):
        pcm = make_simple_voiced_pcm(freq_hz=200.0 + (i * 5))
        eval_frames.append({"pcm": pcm, "timestamp_ms": ts, "transcript_override": f"good {i}"})
        labels.append(1)
        ts += 200

    # bad frames (silent) -> label 0
    for i in range(n_bad):
        pcm = make_silence_pcm()
        eval_frames.append({"pcm": pcm, "timestamp_ms": ts, "transcript_override": f"bad {i}"})
        labels.append(0)
        ts += 200

    # Build calib plus eval list, but we'll compute baseline from calib only
    all_frames = frames + eval_frames

    # 1) acoustic on calibration frames
    acoustics = [aengine.process_frame(f["pcm"], f["timestamp_ms"]) for f in frames]
    # 2) asr
    asr_res_calib = stt.transcribe(frames)
    ling_res_calib = ling.analyze(asr_res_calib, timestamp_ms=acoustics[0].timestamp_ms if acoustics else 0)

    # assemble calibration feature frames and store as calib_frames
    calib_feature_frames = []
    for r in acoustics:
        feat = assemble_feature_frame(sid, {"acoustic": r.acoustic, "qc": r.qc}, ling_res_calib.linguistic, r.timestamp_ms)
        calib_feature_frames.append(feat)

    s.calib_frames = calib_feature_frames
    store.compute_and_store_baseline(sid, min_frames=max(2, n_calib))

    # evaluate: process eval frames in batches
    acoustics_eval = [aengine.process_frame(f["pcm"], f["timestamp_ms"]) for f in eval_frames]
    asr_res_eval = stt.transcribe(eval_frames)
    ling_res_eval = ling.analyze(asr_res_eval, timestamp_ms=acoustics_eval[0].timestamp_ms if acoustics_eval else 0)

    trace = []
    probs = []
    for r, f, l in zip(acoustics_eval, eval_frames, labels):
        feat = assemble_feature_frame(sid, {"acoustic": r.acoustic, "qc": r.qc}, ling_res_eval.linguistic, r.timestamp_ms)
        # compute normalized & score
        b = store.get_baseline(sid)
        if b:
            for m, mstat in b.items():
                val = scorer._lookup_metric(feat, m)
                if val is not None:
                    feat.setdefault("normalized", {})[m] = baseline_utils.z_score(val, mstat.get("median", 0.0), mstat.get("mad", 0.0))
            scres = scorer.compute(feat, b)
            feat["score"] = float(scres.score)
            # simple prob = score / 100
            prob = max(0.0, min(1.0, float(feat["score"]) / 100.0))
        else:
            prob = 0.5
        probs.append(prob)
        feat["prob"] = prob
        feat["label"] = l
        trace.append(feat)

    # compute metrics
    auc = _compute_auc_from_scores(list(zip(probs, labels)))
    brier = _brier_score(probs, labels)
    calibration = _calibration_curve(probs, labels, bins=10)

    return {"trace": trace, "probs": probs, "labels": labels, "auc": auc, "brier": brier, "calibration": calibration}


if __name__ == "__main__":
    import json

    trace = run_demo()
    print(json.dumps(trace, indent=2)[:4000])
