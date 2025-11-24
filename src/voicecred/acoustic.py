from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
from src.voicecred.utils.logger_util import get_logger, logging
logger=get_logger(__name__,logging.DEBUG)
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math

#mpy is optional at import time for fast startup of tests that don't use it
import numpy as np

import parselmouth



@dataclass
class AcousticResult:
    timestamp_ms: int
    acoustic: List[float]
    qc: Dict[str, Any]

def pcm_int16_to_float32(pcm: bytes | List[int] | "np.ndarray") -> "np.ndarray":
    logger.debug("Converting PCM int16 to float32.")
    # Provide a safe fallback when numpy is not installed (do not crash at import time)
    if np is None:
        if isinstance(pcm, bytes):
            from array import array
            arr16 = array("h")
            arr16.frombytes(pcm)
            logger.debug("Converted bytes PCM to float32 using pure Python.")
            return [x / 32768.0 for x in arr16]
        else:
            logger.debug("Converted list PCM to float32 using pure Python.")
            return [float(x) / 32768.0 for x in list(pcm)]

    if isinstance(pcm, bytes):
        arr = np.frombuffer(pcm, dtype=np.int16)
    else:
        arr = np.asarray(pcm, dtype=np.int16)
    logger.debug("Converted PCM to float32 using numpy.")
    return (arr.astype(np.float32) / 32768.0)

def estimate_snr_db(signal: "np.ndarray") -> float:
    logger.debug("Estimating SNR (dB).")
    if np is None:
        if not signal:
            logger.warning("Empty signal for SNR estimation.")
            return float("nan")
        vals = [float(s) for s in signal]
        power = sum(v * v for v in vals) / max(1, len(vals))
        mid = sorted([abs(v) for v in vals])[len(vals) // 2]
        noise_power = max(mid * mid, 1e-12)
        snr = 10.0 * (math.log10(max(power, 1e-12) / noise_power))
        logger.debug(f"SNR (pure Python): {snr}")
        return float(snr)

    if signal.size == 0:
        logger.warning("Empty numpy signal for SNR estimation.")
        return float("nan")
    eps = 1e-9
    power = np.mean(signal.astype(np.float64) ** 2)
    win = 1024
    if signal.size < win:
        noise_power = np.median(signal.astype(np.float64) ** 2) + eps
    else:
        frames = signal[: (signal.size // win) * win].reshape(-1, win)
        energy = np.mean(frames ** 2, axis=1)
        noise_power = max(np.median(energy), eps)
    snr = 10.0 * np.log10(max(power, eps) / noise_power)
    logger.debug(f"SNR (numpy): {snr}")
    return float(snr)

def speech_ratio_energy(signal: "np.ndarray", threshold_db: float = -40.0) -> float:
    logger.debug("Estimating speech ratio energy.")
    if np is None:
        if not signal:
            logger.warning("Empty signal for speech ratio estimation.")
            return 0.0
        mean_sq = sum(float(s) ** 2 for s in signal) / max(1, len(signal))
        rms_db = 10.0 * math.log10(mean_sq + 1e-12)
        ratio = 1.0 if rms_db > threshold_db else 0.0
        logger.debug(f"Speech ratio (pure Python): {ratio}")
        return ratio

    if signal.size == 0:
        logger.warning("Empty numpy signal for speech ratio estimation.")
        return 0.0
    win = 512
    pad = (-signal.size) % win
    if pad:
        signal = np.pad(signal, (0, pad))
    frames = signal.reshape(-1, win)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    ratio = float((db > threshold_db).sum() / db.size)
    logger.debug(f"Speech ratio (numpy): {ratio}")
    return ratio

class AcousticEngine:
    """Small acoustic engine that extracts a compact set of acoustic features."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        logger.info(f"AcousticEngine initialized with sample_rate={sample_rate}")

    def process_frame(self, pcm: bytes | List[int] | np.ndarray, timestamp_ms: int) -> AcousticResult:
        logger.debug(f"Processing frame at timestamp {timestamp_ms}")
        import math

        arr = pcm_int16_to_float32(pcm)
        if np is None:
            vals = [float(x) for x in arr]
            rms = float(math.sqrt(sum(v * v for v in vals) / max(1, len(vals)) + 1e-12))
            if len(vals) < 2:
                zcr = 0.0
            else:
                zc = sum(1 for i in range(len(vals) - 1) if vals[i] * vals[i + 1] < 0)
                zcr = float(zc / max(1, len(vals) - 1))
            snr_db = estimate_snr_db(vals)
            speech_ratio = speech_ratio_energy(vals)
        else:
            rms = float(np.sqrt(np.mean(arr ** 2) + 1e-12))
            zcr = float(((arr[:-1] * arr[1:]) < 0).sum() / max(1, arr.size - 1))
            snr_db = estimate_snr_db(arr)
            speech_ratio = speech_ratio_energy(arr)

        f0_mean = float("nan")
        f0_median = float("nan")
        f0_std = float("nan")
        voiced_seconds = 0.0

        if parselmouth is not None and (np is not None and isinstance(arr, np.ndarray) and arr.size > 50):
            try:
                snd = parselmouth.Sound(arr, self.sample_rate)
                pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
                f0_values = pitch.selected_array['frequency']
                if np is not None:
                    f0_values = np.asarray(f0_values, dtype=float)
                    f0_values = f0_values[f0_values > 0]
                else:
                    f0_values = [float(x) for x in f0_values if x and x > 0]
                if (np is not None and getattr(f0_values, "size", 0) > 0) or (np is None and len(f0_values) > 0):
                    if np is not None:
                        f0_mean = float(np.mean(f0_values))
                        f0_median = float(np.median(f0_values))
                        f0_std = float(np.std(f0_values))
                        voiced_seconds = float(f0_values.size * 0.01)
                    else:
                        f0_mean = float(sum(f0_values) / len(f0_values))
                        f0_median = float(sorted(f0_values)[len(f0_values) // 2])
                        meanv = f0_mean
                        f0_std = float((sum((x - meanv) ** 2 for x in f0_values) / max(1, len(f0_values))) ** 0.5)
                        voiced_seconds = float(len(f0_values) * 0.01)
                    logger.debug(f"Pitch stats: mean={f0_mean}, median={f0_median}, std={f0_std}, voiced_seconds={voiced_seconds}")
                else:
                    logger.warning("No voiced frames detected by parselmouth, using fallback.")
                    raise RuntimeError("no voiced frames from parselmouth")
            except Exception as e:
                logger.error(f"Parselmouth pitch extraction failed: {e}")
                try:
                    if np is not None and isinstance(arr, np.ndarray) and arr.size >= 32:
                        N = arr.size
                        freqs = np.fft.rfftfreq(N, d=1.0 / self.sample_rate)
                        mags = np.abs(np.fft.rfft(arr * np.hanning(N)))
                        band_mask = (freqs >= 50) & (freqs <= 1000)
                        if band_mask.any():
                            band_freqs = freqs[band_mask]
                            band_mags = mags[band_mask]
                            peak_idx = int(np.argmax(band_mags))
                            peak_freq = float(band_freqs[peak_idx])
                            f0_mean = peak_freq
                            f0_median = peak_freq
                            f0_std = 0.0
                            voiced_seconds = float(N / self.sample_rate)
                            logger.debug(f"FFT fallback pitch: {peak_freq}")
                except Exception as e2:
                    logger.error(f"FFT fallback pitch extraction failed: {e2}")

        acoustic_vector = [f0_mean, f0_median, f0_std, rms, zcr]
        qc = {
            "snr_db": snr_db,
            "speech_ratio": speech_ratio,
            "voiced_seconds": voiced_seconds,
        }
        logger.info(f"Acoustic features extracted for timestamp {timestamp_ms}: {acoustic_vector}, QC: {qc}")
        return AcousticResult(timestamp_ms=timestamp_ms, acoustic=acoustic_vector, qc=qc)

    def process_batch(self, frames: List[Dict[str, Any]]) -> List[AcousticResult]:
        logger.info(f"Processing batch of {len(frames)} frames.")
        out: List[AcousticResult] = []
        for f in frames:
            pcm = f.get("pcm")
            ts = int(f.get("timestamp_ms", 0))
            out.append(self.process_frame(pcm, ts))
        logger.info("Batch processing complete.")
        return out

    async def process_batch_async(self, frames: List[Dict[str, Any]]) -> List[AcousticResult]:
        import asyncio
        logger.info(f"Async batch processing of {len(frames)} frames started.")
        result = await asyncio.to_thread(self.process_batch, frames)
        logger.info("Async batch processing complete.")
        return result

def acoustic_batch_to_json(results: List[AcousticResult]) -> List[Dict[str, Any]]:
    logger.debug("Converting acoustic batch results to JSON serializable format.")
    out = []
    for r in results:
        def _is_nan(v):
            try:
                return isinstance(v, float) and math.isnan(v)
            except Exception:
                return False

        out.append({
            "timestamp_ms": r.timestamp_ms,
            "acoustic": [float(x) if (x is not None and not _is_nan(x)) else None for x in r.acoustic],
            "qc": r.qc,
        })
    logger.debug("Conversion to JSON format complete.")
    return out
