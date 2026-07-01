"""Regression tests for experiment metadata and current-run naming."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_README = ROOT / "experiments" / "README.md"
CURRENT_RUNS = ROOT / "experiments" / "CURRENT_RUNS.md"
CURRENT_DIR = ROOT / "experiments" / "current"

DEPRECATED_TOKENS = [
    "Tree-Ring",
    "Tree-Ring-like",
    "treering_like",
    "rotation_attack_treering_like",
    "qualityfix",
    "vae_footprint_two_pair",
    "vae_rotcorr_two_pair_alpha015",
    "vae_roundtrip_two_pair_alpha015",
]

APPROVED_CURRENT_DIRS = {
    "current/rotation_sync_same_size_zfill",
    "current/rotation_sync_same_size_zfill_vae",
    "current/embedding_process_same_size_zfill_alpha015",
    "current/anchor_imperceptibility_two_pair_alpha015",
    "current/gaussian_shading_canonical_rotbind_alpha015_real10",
    "current/gaussian_shading_quality_rotbind_alpha015_real10",
}


def read_experiment_metadata() -> str:
    """Return the current experiment metadata documents as one string."""
    return EXPERIMENTS_README.read_text() + "\n" + CURRENT_RUNS.read_text()


def test_experiment_readme_has_no_deprecated_current_references() -> None:
    text = read_experiment_metadata()

    for token in DEPRECATED_TOKENS:
        assert token not in text


def test_current_runs_lists_only_approved_current_directories() -> None:
    text = CURRENT_RUNS.read_text()
    current_dirs = set(re.findall(r"`(current/[^`]+)`", text))

    assert current_dirs == APPROVED_CURRENT_DIRS
    assert "archive_debug/deprecated_before_rotation_and_quality_fix/" in text


def test_experiment_readme_declares_standard_rotation_attack() -> None:
    text = EXPERIMENTS_README.read_text()

    assert "literature-aligned same-size zero-fill rotation attack" in text
    assert "attack interpolation = nearest" in text
    assert "expand=False" in text
    assert "attack fill = 0" in text
    assert "correction interpolation = bilinear" in text
    assert "correction fill = 0" in text
    assert "This setup is a benchmark rotation definition rather than a method-specific setting." in text


def test_current_directory_contains_no_deprecated_names() -> None:
    deprecated_name_parts = [
        "treering",
        "qualityfix",
        "smoke",
        "vae_footprint_two_pair",
        "vae_rotcorr",
        "vae_roundtrip",
    ]
    approved_names = {name.removeprefix("current/") for name in APPROVED_CURRENT_DIRS}
    current_names = {path.name for path in CURRENT_DIR.iterdir() if path.is_dir()}

    for name in current_names:
        for token in deprecated_name_parts:
            assert token not in name
    assert current_names <= approved_names
