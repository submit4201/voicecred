from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import math

from src.voicecred.utils import baseline as baseline_utils


@dataclass
class ScoreResult:
    score: float
    ci: Tuple[float, float]
    contributions: Dict[str, float]
    z_scores: Dict[str, float]
    explain: Dict[str, Any]


class Scorer:
    """Simple explainable scorer using baseline-normalized z-scores.

    The scorer computes per-metric z-scores using median/MAD baselines, applies a
    weight table to compute contributions, sums into a raw value and maps to a
    0-100 index. The class also provides simple EMA smoothing helpers and CI
    estimation from a recent scores buffer.
    """

    DEFAULT_WEIGHTS = {
        # acoustic: f0_mean, f0_median, f0_std, rms, zcr
        "acoustic.f0_mean": 0.3,
        "acoustic.f0_median": 0.2,
        "acoustic.f0_std": 0.8,
        "acoustic.rms": 0.6,
        "acoustic.zcr": 0.1,
        # linguistic
        "linguistic.ttr": 0.4,
        "linguistic.avg_word_length": 0.3,
        "derived.avg_word_length": 0.3,
        "derived.combined_asr_quality": -0.5,  # higher ASR quality reduces uncertainty/tension
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None, scale: float = 2.0, ci_method: str = "analytic"):
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.scale = float(scale)
        # CI estimation method: 'analytic' (fast) or 'bootstrap' (empirical from recent window)
        self.ci_method = str(ci_method)

    def _lookup_metric(self, frame: Dict[str, Any], metric: str) -> Optional[float]:
        """Resolve metric string like 'acoustic.rms' or 'linguistic.ttr' from a frame dict."""
        parts = metric.split(".")
        current = frame
        try:
            not_found = False
            for p in parts:
                if isinstance(current, dict):
                    current = current.get(p)
                    if current is None:
                        not_found = True
                        break
                else:
                    not_found = True
                    break
            if not not_found and isinstance(current, (int, float)):
                return float(current)
        except Exception:
            pass
        # fallback: acoustic.* metrics may be stored under acoustic_named
        if metric.startswith("acoustic."):
            alt_metric = metric.replace("acoustic.", "acoustic_named.")
            parts = alt_metric.split(".")
            try:
                current = frame
                for p in parts:
                    if isinstance(current, dict):
                        current = current.get(p)
                    else:
                        current = None
                        break
                if isinstance(current, (int, float)):
                    return float(current)
            except Exception:
                pass
        # Special handling for 'derived.' metrics where frame['derived'] is
        # a list of small dicts (e.g. [{'avg_word_length': 4.2}, {'pause_ratio':0.1}])
        # Try to find the key inside list entries when direct lookup fails.
        if metric.startswith("derived."):
            # key after 'derived.'
            k = metric.split(".", 1)[1]
            try:
                derived = frame.get("derived")
                if isinstance(derived, list):
                    for d in derived:
                        if isinstance(d, dict) and k in d:
                            v = d.get(k)
                            if isinstance(v, (int, float)):
                                return float(v)
            except Exception:
                pass

        return None

    def compute(self, frame: Dict[str, Any], baseline: Dict[str, Dict[str, float]]) -> ScoreResult:
        """Compute a score result for a single frame using a baseline dict.

        baseline: mapping metric -> {median,mad,count}
        """
        z_scores: Dict[str, float] = {}
        contributions: Dict[str, float] = {}

        # compute z scores for all metrics present in baseline
        for metric, mstat in baseline.items():
            val = self._lookup_metric(frame, metric)
            if val is None:
                continue
            z = baseline_utils.z_score(val, mstat.get("median", 0.0), mstat.get("mad", 0.0))
            z_scores[metric] = z
            w = float(self.weights.get(metric, 0.0))
            contributions[metric] = w * z

        # raw aggregated value
        raw = sum(contributions.values()) if contributions else 0.0

        # map raw -> 0..100 using scaled tanh
        score = 50.0 * (1.0 + math.tanh(raw / max(1e-6, self.scale)))

        # CI estimation: default to analytic propagation for speed & low compute cost.
        # analytic: assume standardized z-scores ~ unit variance and independent, so
        # var(raw) = sum(w_i^2 * var(z_i)) ≈ sum(w_i^2).
        if self.ci_method == "analytic":
            # compute standard deviation of the raw aggregated value
            if contributions:
                var_raw = sum((float(self.weights.get(k, 0.0)) ** 2) for k in contributions.keys())
                sd_raw = math.sqrt(var_raw) if var_raw > 0.0 else 0.0
            else:
                sd_raw = 0.0
            # convert raw +/- scaled sd into score domain
            raw_lo = raw - 1.96 * sd_raw
            raw_hi = raw + 1.96 * sd_raw
            mean_score = score
            lo = max(0.0, 50.0 * (1.0 + math.tanh(raw_lo / max(1e-6, self.scale))))
            hi = min(100.0, 50.0 * (1.0 + math.tanh(raw_hi / max(1e-6, self.scale))))
        else:
            # fallback to bootstrap-like behavior using recent scores if present
            recent = list(frame.get("_recent_scores", []))
            if recent:
                mean_raw = sum(recent) / len(recent)
                var = sum((r - mean_raw) ** 2 for r in recent) / max(1, len(recent) - 1)
                sd = math.sqrt(var) if var >= 0 else 0.0
                # map raw mean to score domain
                mean_score = 50.0 * (1.0 + math.tanh(mean_raw / max(1e-6, self.scale)))
                lo = max(0.0, mean_score - 1.96 * sd)
                hi = min(100.0, mean_score + 1.96 * sd)
            else:
                lo = max(0.0, score - 5.0)
                hi = min(100.0, score + 5.0)

        explain = {
            "raw": raw,
            "weights_used": {k: v for k, v in self.weights.items() if k in contributions},
            "top_factors": sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5],
        }

        return ScoreResult(score=score, ci=(lo, hi), contributions=contributions, z_scores=z_scores, explain=explain)

    def update_ema(self, scoring_state: Dict[str, Any], raw: float) -> float:
        """Update EMA in scoring_state (in-place) & return updated EMA value."""
        alpha = float(scoring_state.get("ema_alpha", 0.2))
        last = scoring_state.get("ema_last")
        if last is None:
            new = raw
        else:
            new = alpha * raw + (1 - alpha) * float(last)
        scoring_state["ema_last"] = new
        # maintain recent_scores buffer
        buf: List[float] = scoring_state.get("recent_scores", [])
        buf.append(raw)
        # keep only last window
        window = int(scoring_state.get("score_window", 20) or 20)
        while len(buf) > window:
            buf.pop(0)
        scoring_state["recent_scores"] = buf
        return new
