"""Tests for the controlled-contamination sweep.

These guard the arithmetic the new results rest on. A bug here would not crash
anything — it would quietly produce a plausible-looking curve with the wrong slope,
which is the worst kind of bug for a project whose entire argument is about
trustworthy numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contamination import contaminate, fit_slope

@pytest.fixture
def test_df():
    """A test partition with a deliberately uneven class mix, like the real one."""
    rows = []
    for cls, n in [("nv", 60), ("mel", 20), ("bkl", 15), ("bcc", 5)]:
        for i in range(n):
            rows.append({"image_id": f"test_{cls}_{i}", "lesion_id": f"L_test_{cls}_{i}",
                         "dx": cls, "group": f"L_test_{cls}_{i}", "split": "test"})
    return pd.DataFrame(rows)


@pytest.fixture
def donors():
    rows = []
    for cls, n in [("nv", 200), ("mel", 60), ("bkl", 50), ("bcc", 20)]:
        for i in range(n):
            rows.append({"image_id": f"train_{cls}_{i}", "lesion_id": f"L_train_{cls}_{i}",
                         "dx": cls, "group": f"L_train_{cls}_{i}", "split": "train"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Contamination mechanics
# --------------------------------------------------------------------------------------

def test_zero_rate_is_a_no_op(test_df, donors):
    out, achieved = contaminate(test_df, donors, 0.0, "dx", np.random.default_rng(0))
    assert achieved == 0.0
    assert len(out) == len(test_df)
    assert set(out["image_id"]) == set(test_df["image_id"])


def test_test_set_size_is_preserved(test_df, donors):
    """Substitution, not addition — otherwise the denominator moves with the treatment."""
    for rate in (0.1, 0.2, 0.4):
        out, _ = contaminate(test_df, donors, rate, "dx", np.random.default_rng(1))
        assert len(out) == len(test_df), f"size changed at rate {rate}"


def test_class_balance_is_preserved(test_df, donors):
    """Balanced accuracy is sensitive to class mix, so the mix must not be the variable."""
    before = test_df["dx"].value_counts().sort_index()
    for rate in (0.1, 0.25, 0.4):
        out, _ = contaminate(test_df, donors, rate, "dx", np.random.default_rng(2))
        after = out["dx"].value_counts().sort_index()
        pd.testing.assert_series_equal(before, after, check_names=False)


def test_achieved_rate_tracks_the_target(test_df, donors):
    for rate in (0.1, 0.2, 0.3, 0.4):
        out, achieved = contaminate(test_df, donors, rate, "dx", np.random.default_rng(3))
        assert abs(achieved - rate) < 0.02, f"target {rate}, achieved {achieved}"


def test_substituted_images_come_from_the_donor_pool(test_df, donors):
    out, _ = contaminate(test_df, donors, 0.3, "dx", np.random.default_rng(4))
    added = set(out["image_id"]) - set(test_df["image_id"])
    assert added, "nothing was substituted in"
    assert added.issubset(set(donors["image_id"]))


def test_no_donor_is_used_twice(test_df, donors):
    """A duplicated donor would score the same image twice and inflate the effect."""
    out, _ = contaminate(test_df, donors, 0.4, "dx", np.random.default_rng(5))
    assert out["image_id"].is_unique


def test_small_donor_pool_degrades_gracefully(test_df):
    """Falls short of the target rather than sampling with replacement or crashing."""
    tiny = pd.DataFrame([{"image_id": "train_nv_0", "lesion_id": "L0",
                          "dx": "nv", "group": "L0", "split": "train"}])
    out, achieved = contaminate(test_df, tiny, 0.4, "dx", np.random.default_rng(6))
    assert len(out) == len(test_df)
    assert achieved < 0.4
    assert out["image_id"].is_unique


def test_draws_differ_between_seeds(test_df, donors):
    """Error bars are only meaningful if repeats actually resample."""
    a, _ = contaminate(test_df, donors, 0.3, "dx", np.random.default_rng(7))
    b, _ = contaminate(test_df, donors, 0.3, "dx", np.random.default_rng(8))
    assert set(a["image_id"]) != set(b["image_id"])


# --------------------------------------------------------------------------------------
# Slope fitting
# --------------------------------------------------------------------------------------

def test_slope_recovers_a_known_line():
    rates = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    values = 0.65 + 0.25 * rates          # +0.025 per 10pp by construction
    fit = fit_slope(rates, values)
    assert fit["slope_per_10pp"] == pytest.approx(0.025, abs=1e-9)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_flat_curve_has_zero_slope():
    rates = np.array([0.0, 0.1, 0.2, 0.3])
    fit = fit_slope(rates, np.full_like(rates, 0.70))
    assert fit["slope_per_10pp"] == pytest.approx(0.0, abs=1e-9)
