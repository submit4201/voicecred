from __future__ import annotations

from typing import Dict, List, Any, Optional
from src.voicecred.utils.baseline import compute_median_mad
import math


def compute_baseline_from_frames(
    frames: List[Dict[str, Any]],
    min_frames: int | None = 3,
    qc_gate_min_speech_ratio: float = 0.2,
    qc_gate_min_voiced_seconds: float = 0.05,
    *,
    use_production_policy: bool = False,
    min_voiced_seconds: float | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute median/MAD baseline across provided frames.

    frames: list of feature dicts with keys like 'acoustic', 'linguistic', 'derived', 'qc'.
    Returns mapping metric -> {median,mad,count} for metrics meeting min_frames.
    """
    # if frames list empty -> nothing to compute
    if not frames:
        return {}
    
    if min_frames is not None and len(frames) < min_frames:
        return {}

    # If production policy is enabled, require a minimum total voiced_seconds
    if use_production_policy:
        total_voiced = 0.0        for f in frames:
            qc = f.get("qc") if isinstance(f, dict) else None
            try:
                total_voiced += float(qc.get("voiced_seconds", 0.0) or 0.0) if isinstance(qc, dict) else 0.0
            except Exception:
                pass
        # Not enough voiced seconds to compute a production baseline
        if min_voiced_seconds is not None and total_voiced < float(min_voiced_seconds):
            return {}

    # QC-gating similar to store: keep frames that meet at least one QC condition
    kept = []
    for f in frames:
        qc = f.get("qc") if isinstance(f, dict) else None
        accept = False if isinstance(qc, dict) else True
        if isinstance(qc, dict):
            try:
                sr = float(qc.get("speech_ratio", 0.0))
            except Exception:
                sr = 0.0
            try:
                vs = float(qc.get("voiced_seconds", 0.0))
            except Exception:
                vs = 0.0
            if (not math.isnan(sr) and sr >= qc_gate_min_speech_ratio) or (not math.isnan(vs) and vs >= qc_gate_min_voiced_seconds):
                accept = True
            else:
                try:
                    snr = float(qc.get("snr_db", float("nan")))
                except Exception:
                    snr = float("nan")
                if not math.isnan(snr) and snr >= 0.0:
                    accept = True
                else:
                    try:
                        words = int(qc.get("words_in_window", 0) or 0)
                    except Exception:
                        words = 0
                    if words > 0:
                        accept = True
        if accept:
            kept.append(f)

    if min_frames is not None and len(kept) < min_frames:
        return {}

    # aggregate numeric values per metric
    metrics = {}

    def observe(key: str, v):
        metrics.setdefault(key, []).append(v)

    for f in kept:
        ac = f.get("acoustic") if isinstance(f, dict) else None
        if isinstance(ac, dict):
            for k, v in ac.items():
                observe(f"acoustic.{k}", v)
        elif isinstance(ac, list):
            names = ["f0_mean", "f0_median", "f0_std", "rms", "zcr"]
            for idx, v in enumerate(ac):
                key = names[idx] if idx < len(names) else f"idx_{idx}"
                observe(f"acoustic.{key}", v)

        lg = f.get("linguistic") if isinstance(f, dict) else None
        if isinstance(lg, dict):
            for k, v in lg.items():
                observe(f"linguistic.{k}", v)

        for d in (f.get("derived") or []):
            if isinstance(d, dict):
                for k, v in d.items():
                    observe(f"derived.{k}", v)

        qc = f.get("qc") if isinstance(f, dict) else None
        if isinstance(qc, dict):
            for k, v in qc.items():
                observe(f"qc.{k}", v)

    baseline = {}
    for k, vals in metrics.items():
        # ensure we have enough numeric values
        numeric = []
        for v in vals:
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isnan(fv):
                continue
            numeric.append(fv)
        if min_frames is not None and len(numeric) < min_frames:
            continue
        median, mad = compute_median_mad(numeric)
        baseline[k] = {"median": median, "mad": mad, "count": len(numeric)}

    return baseline
