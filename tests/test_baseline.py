import pytest
from voicecred.utils import baseline
import math

def test_compute_median_mad_odd():
    vals = [1, 2, 3]
    median, mad = baseline.compute_median_mad(vals)
    assert median == 2.0
    assert mad == 1.0


def test_compute_median_mad_even():
    vals = [1, 2, 3, 4]
    median, mad = baseline.compute_median_mad(vals)
    assert median == 2.5
    # deviations from median: [1.5, 0.5, 0.5, 1.5] -> median = 1.0
    assert mad == 1.0


def test_mad_to_sigma_and_z():
    mad = 1.0
    sigma = baseline.mad_to_sigma(mad)
    assert abs(sigma - 1.4826) < 1e-6

    z = baseline.z_score(3.0, median=2.0, mad=1.0)
    # (3 - 2) / sigma
    assert abs(z - (1.0 / sigma)) < 1e-6


def test_normalize_sequence_and_edgecases():
    vals = [10, 12, 14, 16]
    zseq, median, mad = baseline.normalize_sequence(vals)
    print(f"Normalized sequence: {zseq}, median: {median}, mad: {mad}")
    assert median == 13.0
    assert mad == 2.0
    # check length
    assert len(zseq) == 4

    # zero-length should raise
    try:
        baseline.compute_median_mad([])
        print('Caught expected ValueError for empty input')
        assert False, "Expected ValueError for empty input"
    except ValueError:
        print('Caught expected ValueError for empty input')
        pass


def test_rolling_median_mad():
    from voicecred.utils.baseline import RollingMedianMAD

    r = RollingMedianMAD(window=3)
    r.add(1)
    r.add(2)
    r.add(3)
    median, mad, count = r.get_stats()
    assert median == 2.0
    assert mad == 1.0
    assert count == 3

    # push a new value -> window should be [2,3,6]
    r.add(6)
    median2, mad2, count2 = r.get_stats()
    assert median2 == 3.0
    assert mad2 == 1.0
    assert count2 == 3

    # reset should clear window
    r.reset()
    try:
        r.get_stats()
        assert False, "Expected ValueError after resetting empty window"
    except ValueError:
        pass


def test_compute_median_mad_ignores_nans():
    vals = [1, float('nan'), 3]
    median, mad = baseline.compute_median_mad(vals)
    assert median == 2.0
    assert mad == 1.0


def test_compute_median_mad_all_nans_raises():
    with pytest.raises(ValueError):
        baseline.compute_median_mad([float('nan'), float('nan')])


def test_normalize_sequence_preserves_nans():
    vals = [10, float('nan'), 14]
    zseq, median, mad = baseline.normalize_sequence(vals)
    assert math.isnan(zseq[1])
    assert median == 12.0
    assert mad == 2.0
