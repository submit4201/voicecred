"""Robust baseline helpers: median, MAD and z-score helpers.

These helpers implement median/MAD based normalization which is robust to outliers.
"""
from typing import Iterable, Tuple, List
from collections import deque
import math
import logging

logger = logging.getLogger(__name__)


def compute_median_mad(values: Iterable[float]) -> Tuple[float, float]:
    """Compute the median and MAD (median absolute deviation) for values.

    Returns (median, mad). If input is empty raises ValueError.
    MAD is computed as the median of absolute deviations from the median.
    """
    # convert to float and drop NaNs so baseline ignores invalid measurements
    vals_raw = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        # skip NaN values
        if math.isnan(fv):
            continue
        vals_raw.append(fv)
    vals = sorted(vals_raw)
    logger.debug("compute_median_mad: raw_values=%s -> filtered_sorted=%s", list(values), vals)
    logger.debug("compute_median_mad: values=%s -> sorted=%s", list(values), vals)
    n = len(vals)
    if n == 0:
        raise ValueError("compute_median_mad requires at least one non-NaN numeric value")

    # median
    mid = n // 2
    if n % 2 == 1:
        median = vals[mid]
    else:
        median = 0.5 * (vals[mid - 1] + vals[mid])

    # median absolute deviation
    deviations = [abs(v - median) for v in vals]
    deviations.sort()
    if n % 2 == 1:
        mad = deviations[mid]
    else:
        mad = 0.5 * (deviations[mid - 1] + deviations[mid])

    logger.debug("compute_median_mad: median=%s mad=%s", median, mad)
    return float(median), float(mad)


def mad_to_sigma(mad: float) -> float:
    """Convert MAD to approximate standard deviation (sigma) using constant 1.4826.

    For normally-distributed data: sigma ≈ 1.4826 * MAD.
    """
    res = float(mad) * 1.4826
    logger.debug("mad_to_sigma: mad=%s -> sigma=%s", mad, res)
    return res


def z_score(x: float, median: float, mad: float, small_value: float = 1e-1) -> float:
    """Compute robust z-score for x using median & MAD.

    When MAD is zero, fall back to small_value to avoid division by zero.
    """
    sigma = mad_to_sigma(mad)
    if sigma <= 0:
        sigma = small_value
    z = float((x - median) / sigma)
    logger.debug("z_score: x=%s median=%s mad=%s sigma=%s z=%s", x, median, mad, sigma, z)
    return z


def normalize_sequence(values: Iterable[float]) -> Tuple[List[float], float, float]:
    """Return sequence of z-scores for input values and the computed (median, mad).

    Useful for creating normalized arrays for scoring downstream.
    """
    # Preserve original positions, but compute median/mad on non-NaN values only
    raw_vals = []
    for v in values:
        try:
            rv = float(v)
        except Exception:
            rv = math.nan
        raw_vals.append(rv)

    # compute baseline on values that are not NaN
    valid_vals = [v for v in raw_vals if not math.isnan(v)]
    median, mad = compute_median_mad(valid_vals)

    # produce z for the original sequence; NaNs remain NaN in the output
    zseq = [z_score(v, median, mad) if not math.isnan(v) else float("nan") for v in raw_vals]
    logger.debug("normalize_sequence: median=%s mad=%s zseq=%s", median, mad, zseq)
    logger.debug("normalize_sequence: median=%s mad=%s zseq=%s", median, mad, zseq)
    return zseq, median, mad


class RollingMedianMAD:
    """Simple rolling window median/MAD helper.

    This implementation keeps a fixed-size window of the last N numeric
    observations. add() appends a new value, and get_stats() returns
    (median, mad, count) computed on the window.

    Note: This is a straightforward, correct but not optimized implementation
    (recomputes median/MAD on the window on each update). For small windows
    this is perfectly acceptable in test/experiment code.
    """

    def __init__(self, window: int = 100):
        if window <= 0:
            raise ValueError("window must be > 0")
        self.window = int(window)
        self._dq = deque(maxlen=self.window)

    def add(self, value: float):
        try:
            fv = float(value)
        except Exception:
            # ignore non-numeric values
            return
        # ignore NaNs -- they should not contaminate rolling stats
        if math.isnan(fv):
            return
        self._dq.append(fv)

    def get_stats(self) -> Tuple[float, float, int]:
        if not self._dq:
            raise ValueError("no values in rolling window")
        median, mad = compute_median_mad(list(self._dq))
        return median, mad, len(self._dq)

    def reset(self):
        self._dq.clear()
