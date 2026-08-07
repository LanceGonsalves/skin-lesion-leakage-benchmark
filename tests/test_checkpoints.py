"""Checkpoint naming.

A replication sweep trains the same split many times. If every run wrote to the same
filename the sweep would silently overwrite the headline model and every later
evaluation would score the wrong weights -- with no error anywhere.
"""

from __future__ import annotations

from src.models.build import checkpoint_name


def test_default_run_keeps_the_original_filename():
    """Existing commands, figures and README references must keep resolving."""
    assert checkpoint_name("grouped", "efficientnet_b0", 42,
                           "efficientnet_b0", 42) == "grouped_best.pt"
    assert checkpoint_name("naive", "efficientnet_b0", 42,
                           "efficientnet_b0", 42) == "naive_best.pt"


def test_a_different_seed_gets_its_own_file():
    name = checkpoint_name("grouped", "efficientnet_b0", 43, "efficientnet_b0", 42)
    assert name != "grouped_best.pt"
    assert "43" in name


def test_a_different_backbone_gets_its_own_file():
    name = checkpoint_name("grouped", "resnet18", 42, "efficientnet_b0", 42)
    assert name != "grouped_best.pt"
    assert "resnet18" in name


def test_sweep_runs_never_collide():
    names = {
        checkpoint_name(split, backbone, seed, "efficientnet_b0", 42)
        for split in ("naive", "grouped")
        for backbone in ("efficientnet_b0", "resnet18")
        for seed in (42, 43, 44)
    }
    # 2 splits x 2 backbones x 3 seeds, all distinct
    assert len(names) == 12
