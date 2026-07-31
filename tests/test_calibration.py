"""Tests for calibration metrics.

Calibration is easy to get subtly wrong — off-by-one bin edges silently drop the
most confident predictions, which is exactly the region that matters for a medical
model. These tests pin the behaviour against cases with known answers.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.explain import expected_calibration_error, reliability_bins

SEED = 0


def test_perfectly_calibrated_model_scores_near_zero():
    """If a model is right exactly as often as it claims, ECE should vanish."""
    rng = np.random.RandomState(SEED)
    confidences = rng.uniform(0.05, 1.0, 20_000)
    correct = (rng.uniform(size=20_000) < confidences).astype(float)

    assert expected_calibration_error(confidences, correct) < 0.02


def test_overconfident_model_is_penalised():
    """Claims 95% confidence, right 60% of the time -> ECE about 0.35."""
    rng = np.random.RandomState(SEED)
    confidences = np.full(5_000, 0.95)
    correct = (rng.uniform(size=5_000) < 0.60).astype(float)

    assert expected_calibration_error(confidences, correct) == pytest.approx(0.35, abs=0.05)


def test_underconfident_model_is_also_penalised():
    """Calibration error is symmetric — being too humble is still miscalibrated."""
    rng = np.random.RandomState(SEED)
    confidences = np.full(5_000, 0.40)
    correct = (rng.uniform(size=5_000) < 0.90).astype(float)

    assert expected_calibration_error(confidences, correct) == pytest.approx(0.50, abs=0.05)


def test_confidence_of_exactly_one_is_not_dropped():
    """The top bin must be closed on the right, or the most confident predictions vanish."""
    confidences = np.ones(100)
    correct = np.ones(100)

    bins = reliability_bins(confidences, correct)
    assert bins["n"].sum() == 100, "confidence == 1.0 fell outside every bin"
    assert expected_calibration_error(confidences, correct) == pytest.approx(0.0)


def test_bins_partition_the_data_exactly_once():
    rng = np.random.RandomState(SEED)
    confidences = rng.uniform(0, 1, 1_000)
    correct = (rng.uniform(size=1_000) < confidences).astype(float)

    assert reliability_bins(confidences, correct)["n"].sum() == 1_000


def test_reliability_bins_have_expected_shape():
    rng = np.random.RandomState(SEED)
    confidences = rng.uniform(0, 1, 500)
    correct = (rng.uniform(size=500) < confidences).astype(float)

    bins = reliability_bins(confidences, correct, n_bins=10)
    assert len(bins) == 10
    assert set(bins.columns) == {"bin_low", "bin_high", "n", "mean_confidence", "accuracy"}


def test_empty_input_returns_nan_rather_than_crashing():
    assert np.isnan(expected_calibration_error(np.array([]), np.array([])))
