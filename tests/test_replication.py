"""Replication statistics.

The paired analysis is what turns "we saw a gap once" into "the gap survives being
run again". These tests check it can find a real effect *and*, just as importantly,
that it can fail to find one -- a test that always reports significance proves nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.replicate import paired_gaps, paired_test


def _runs(naive_vals, grouped_vals, backbone="efficientnet_b0"):
    rows = []
    for i, (n, g) in enumerate(zip(naive_vals, grouped_vals)):
        seed = 42 + i
        rows.append({"split": "naive", "seed": seed, "backbone": backbone,
                     "balanced_accuracy": n, "mel_recall": n})
        rows.append({"split": "grouped", "seed": seed, "backbone": backbone,
                     "balanced_accuracy": g, "mel_recall": g})
    return pd.DataFrame(rows)


def test_gaps_are_paired_within_seed():
    runs = _runs([0.75, 0.76, 0.74], [0.65, 0.67, 0.64])
    gaps = paired_gaps(runs)
    assert len(gaps) == 3
    assert gaps["gap"].to_numpy() == pytest.approx([0.10, 0.09, 0.10])


def test_unpaired_seed_is_dropped():
    """A seed whose second arm crashed must not contribute half an observation."""
    runs = _runs([0.75, 0.76], [0.65, 0.67])
    runs = pd.concat([runs, pd.DataFrame([{
        "split": "naive", "seed": 99, "backbone": "efficientnet_b0",
        "balanced_accuracy": 0.80, "mel_recall": 0.80}])], ignore_index=True)
    gaps = paired_gaps(runs)
    assert set(gaps["seed"]) == {42, 43}


def test_consistent_gap_is_significant():
    stats = paired_test(np.array([0.10, 0.09, 0.11, 0.10, 0.095]))
    assert stats["n_pairs"] == 5
    assert stats["mean_gap"] == pytest.approx(0.099, abs=1e-3)
    assert stats["t_p_value"] < 0.001
    assert stats["cohens_d"] > 5


def test_noise_around_zero_is_not_significant():
    """The test must be capable of failing to find an effect, or it proves nothing."""
    stats = paired_test(np.array([0.01, -0.012, 0.004, -0.008, 0.002]))
    assert stats["t_p_value"] > 0.05


def test_single_pair_reports_no_spread():
    stats = paired_test(np.array([0.10]))
    assert stats["n_pairs"] == 1
    assert np.isnan(stats["std_gap"])
    assert "t_p_value" not in stats


