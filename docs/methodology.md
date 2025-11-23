# VoiceCred methodology — baseline, normalization, and scoring

This document summarizes the math and rationale behind the baseline normalization and scoring system implemented in this repository.

## Goals

- Provide robust per-session baseline statistics for acoustic / linguistic metrics using median and MAD (median absolute deviation).
- Normalize incoming measurements relative to a session baseline in a way that is robust to outliers and non-Gaussian distributions.
- Produce an explainable per-frame score (0–100) aggregating multiple metrics, smoothed by EMA, and provide a simple confidence / uncertainty estimate.

## Baseline estimation (median + MAD)

- For each numeric metric collected during session calibration we compute:
  - median = median(values)
  - MAD = median(|values - median|)

- Why median/MAD?
  - Median and MAD are robust to outliers and heavy-tailed distributions compared to mean/std.
  - MAD estimates spread without being unduly influenced by a few extreme values.

- NaN handling
  - NaN values are ignored when computing both the median and MAD — they do not contribute to the baseline.
  - If a metric contains too few non-NaN values the metric is skipped for baseline (treated as unreliable).
  - The session stores a list of skipped metrics (metadata.baseline_skipped_metrics) and sets metadata.baseline_inconclusive when 50% or more metrics were skipped.

## Convert MAD → sigma

- For approximately normal data, MAD can be converted to an estimate of standard deviation by multiplying by 1.4826:

  sigma ≈ 1.4826 × MAD

This conversion gives a scale factor suitable for z-score-style normalization while retaining robust estimation.

## Robust z-score

- For an observation x, the robust z-score is defined as:

  z = (x - median) / sigma

  where sigma is computed from MAD as above. If sigma is zero we fall back to a tiny positive constant (1e-9) to avoid division by zero.

- NaN handling for z-scores
  - If an input observation is NaN the z-score for that position is preserved as NaN.

## Mapping z → 0..100 index

- Each metric's contribution to the final score is computed as a scaled tanh of the z-score (or another monotonic mapping) then weighted and summed.

- Example mapping used in the Scorer:
  - contribution = weight × (50 × (1 + tanh(scale × z)))

  This maps negative z to values under 50 and positive z to values above 50, producing an interpretable index from 0..100.

## Exponential Moving Average (EMA)

- The final per-session score is optionally smoothed using EMA to reduce frame-to-frame jitter.

- EMA update rule:

  ema_new = alpha × score + (1 - alpha) × ema_old

  where alpha is the smoothing factor (0 < alpha ≤ 1). Smaller alpha → heavier smoothing.

## Simple Confidence / CI estimate

- A light-weight confidence interval / uncertainty is computed from the recent window of scores (e.g., last N scores) using empirical variance or standard error of the mean.

- This is intentionally simple — it provides a quick, interpretable signal on whether recent scores are stable or noisy.

## Caveats and guidance

- Silent audio and non-speech
  - Many acoustic features become NaN with silent or zero-valued PCM. Tests in this repo use synthetic silent frames which trigger NaNs.
  - NaNs are intentionally ignored during baseline estimation; however if too many frames are NaN the baseline will be marked inconclusive so calibration should be repeated or calibrated with better data.

- Calibration quality

## Production hardening and calibration policies

- This repo supports a two-tiered calibration policy to make testing fast and production more conservative:
  - Test-mode (default in unit tests / dev): minimal warm-up for rapid feedback — for example, 3 calibration frames.
  - Production-mode (opt-in): stricter warm-up requirements — e.g., at least 30 calibration frames OR a total voiced duration of 60+ seconds across calibration frames. The pipeline accepts the baseline when either condition is met.

- QC gates remain in place (speech_ratio, voiced_seconds, snr_db, and ASR-derived evidence such as a positive words_in_window) to avoid including silent/noisy frames in the baseline.

## Baseline persistence and privacy

- Baseline storage is session-scoped by default for privacy (the session's baseline is not persisted beyond the session lifetime).
- Opt-in persistence: sessions can opt into storing a persistent baseline across sessions (e.g., research or personalization workflows) — this is strictly opt-in and requires a clear consent flag. When enabled the store will persist the baseline to a designated persistent store. Make sure to implement appropriate access controls and privacy notices before enabling persistence in production.

### Quick example: opt-in persistence and production calibration

Here's a tiny example showing how a session can opt into persistent baselines and how to request the stricter "production" calibration policy when computing the baseline.

```python
from voicecred.session import InMemorySessionStore

store = InMemorySessionStore()
session = store.create_session()

# Client (or test) supplies calibration frames
session.calib_frames = [ ... ]  # assembled feature frames collected during warm-up

# Opt-in to persist the baseline for this session (privacy-sensitive)
session.persist_baseline = True

# Compute baseline using the production policy (stricter thresholds):
# - Requires min_frames (default 30) OR total voiced seconds (default 60s) across kept frames
baseline = store.compute_and_store_baseline(session.session_id, use_production_policy=True)

# Access persisted baseline store (for opt-in sessions)
persisted = store.persistent_baselines.get(session.session_id)
```

Notes:

- session.persist_baseline must be set explicitly to True for the store to persist the baseline for that session.
- For quick unit tests and dev flows, omit use_production_policy and use the default fast warm-up thresholds (e.g., 3 frames).

## Confidence intervals (CI)

- Default CI estimation in the Scorer is analytic propagation from per-metric contributions for low-latency, low-compute CI estimates. This assumes per-feature standardized z-scores and independent contributions as a conservative approximation: var(raw) ≈ sum(weights^2).
- Optionally a bootstrap-like method (empirical variance from recent scores) can be used behind a configuration flag when heavier compute or richer uncertainty estimates are desired.

## Next steps / Improvements

- Add explicit QC gates during calibration to only include frames with speech_ratio and voiced_seconds above thresholds.
- Provide a documented, reproducible evaluation harness with labeled datasets and calibration validation metrics (AUC, Brier score, calibration curves).
- Document best practices and recommended sizes for calibration windows for each metric.

---

If you'd like, I can add this doc to the top-level README and/or expand the evaluation harness to demonstrate these behaviors with synthetic labeled data.
