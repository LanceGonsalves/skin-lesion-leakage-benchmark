"""Tests for evaluation metrics and class-imbalance handling.

Deliberately torch-free so they run in CI without installing a deep-learning stack.
The headline test is `test_balanced_accuracy_exposes_majority_class_predictor`: it
encodes why this project reports balanced accuracy at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import bootstrap_ci, compute_metrics, per_class_report
from src.models.dataset import compute_class_weights, sampler_weights
from src.models.train import balanced_accuracy

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
# Approximate HAM10000 proportions.
PROPORTIONS = [0.033, 0.051, 0.110, 0.011, 0.111, 0.669, 0.015]
SEED = 42


@pytest.fixture
def y_true() -> np.ndarray:
    rng = np.random.RandomState(SEED)
    return rng.choice(len(CLASSES), size=1700, p=PROPORTIONS)


# --------------------------------------------------------------------------------------
# Why balanced accuracy
# --------------------------------------------------------------------------------------

def test_balanced_accuracy_exposes_majority_class_predictor(y_true):
    """A model that only ever predicts `nv` should look good on accuracy, bad on balanced."""
    always_nv = np.full_like(y_true, CLASSES.index("nv"))
    metrics = compute_metrics(y_true, always_nv, CLASSES)

    assert metrics["accuracy"] > 0.6, "sanity: nv is the majority class"
    assert metrics["balanced_accuracy"] == pytest.approx(1 / len(CLASSES), abs=0.01), (
        "balanced accuracy must collapse to 1/n_classes for a single-class predictor"
    )


def test_perfect_predictor_scores_one(y_true):
    metrics = compute_metrics(y_true, y_true, CLASSES)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)


def test_training_balanced_accuracy_matches_sklearn(y_true):
    """train.py computes balanced accuracy itself; it must not drift from sklearn."""
    rng = np.random.RandomState(SEED + 1)
    y_pred = rng.choice(len(CLASSES), size=len(y_true), p=PROPORTIONS)

    assert balanced_accuracy(y_true, y_pred, len(CLASSES)) == pytest.approx(
        compute_metrics(y_true, y_pred, CLASSES)["balanced_accuracy"], abs=1e-9
    )


def test_balanced_accuracy_ignores_absent_classes():
    """A class with no true examples should be skipped, not scored as zero recall."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    # Class 2 never appears; perfect prediction should still be 1.0.
    assert balanced_accuracy(y_true, y_pred, n_classes=3) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Per-class reporting
# --------------------------------------------------------------------------------------

def test_per_class_report_covers_every_class(y_true):
    rng = np.random.RandomState(SEED + 2)
    y_pred = rng.choice(len(CLASSES), size=len(y_true), p=PROPORTIONS)
    report = per_class_report(y_true, y_pred, CLASSES)

    assert list(report["class"]) == CLASSES
    assert report["support"].sum() == len(y_true)
    assert ((report[["precision", "recall", "f1"]] >= 0).all().all())
    assert ((report[["precision", "recall", "f1"]] <= 1).all().all())


# --------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------

def test_bootstrap_ci_brackets_point_estimate(y_true):
    rng = np.random.RandomState(SEED + 3)
    y_pred = rng.choice(len(CLASSES), size=len(y_true), p=PROPORTIONS)

    point = compute_metrics(y_true, y_pred, CLASSES)
    cis = bootstrap_ci(y_true, y_pred, CLASSES, n_iterations=300, seed=SEED)

    for metric, (low, high) in cis.items():
        assert low <= point[metric] <= high, f"{metric} point estimate outside its CI"
        assert low < high


def test_bootstrap_is_deterministic_given_seed(y_true):
    rng = np.random.RandomState(SEED + 4)
    y_pred = rng.choice(len(CLASSES), size=len(y_true), p=PROPORTIONS)

    first = bootstrap_ci(y_true, y_pred, CLASSES, n_iterations=200, seed=SEED)
    second = bootstrap_ci(y_true, y_pred, CLASSES, n_iterations=200, seed=SEED)
    assert first == second


# --------------------------------------------------------------------------------------
# Class imbalance
# --------------------------------------------------------------------------------------

def test_class_weights_are_inverse_frequency_and_mean_one():
    labels = pd.Series(["nv"] * 670 + ["mel"] * 111 + ["df"] * 11 + ["bcc"] * 51 +
                       ["bkl"] * 110 + ["akiec"] * 33 + ["vasc"] * 15)
    weights = compute_class_weights(labels, CLASSES)

    assert weights.mean() == pytest.approx(1.0, abs=1e-5), "weights should normalise to mean 1"
    # Rarer class must get the larger weight.
    assert weights[CLASSES.index("df")] > weights[CLASSES.index("nv")]
    assert (weights > 0).all()


def test_absent_class_does_not_produce_infinite_weight():
    labels = pd.Series(["nv"] * 100 + ["mel"] * 10)   # five classes missing
    weights = compute_class_weights(labels, CLASSES)

    assert np.isfinite(weights).all()
    assert (weights > 0).all()


def test_sampler_weights_align_with_labels():
    labels = pd.Series(["nv", "df", "nv", "mel"])
    weights = sampler_weights(labels, CLASSES)

    assert len(weights) == len(labels)
    # The rare class gets sampled more often than the common one.
    assert weights[1] > weights[0]
    assert weights[0] == pytest.approx(weights[2]), "same label -> same weight"
