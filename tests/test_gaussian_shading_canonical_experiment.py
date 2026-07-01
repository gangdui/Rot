"""Tests for the Gaussian Shading canonical RotBind necessity experiment."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from rotbind_anchor.gaussian_shading_adapter import load_gaussian_shading_pipeline
from rotbind_anchor.eval_gaussian_shading_rotbind_canonical import (
    RESULT_FIELDS,
    amplification,
    compute_pair_metrics,
    cosine,
    main,
    mse,
    rel_mse,
    summarize_rows,
)


def test_pixel_cosine_and_amplification_metrics() -> None:
    base = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    variant = np.asarray([2.0, 4.0, 6.0], dtype=np.float32)

    assert mse(variant, base) == pytest.approx((1.0 + 4.0 + 9.0) / 3.0)
    assert rel_mse(variant, base) == pytest.approx(mse(variant, base) / np.mean(base**2))
    assert cosine(base, variant) == pytest.approx(1.0)
    assert amplification(8.0, 2.0) == pytest.approx(4.0)

    metrics = compute_pair_metrics("toy", base, variant)
    assert metrics["toy_mse"] == pytest.approx(mse(variant, base))
    assert metrics["toy_rel_mse"] == pytest.approx(rel_mse(variant, base))
    assert metrics["toy_cos"] == pytest.approx(1.0)


def test_summary_aggregates_mean_median_and_success_rates() -> None:
    rows = [
        {
            "pixel_mse_anchor": 1.0,
            "pixel_mse_clean": 0.25,
            "detector_success_base": 1,
            "detector_success_anchor": 0,
            "detector_success_clean": 1,
        },
        {
            "pixel_mse_anchor": 3.0,
            "pixel_mse_clean": 0.75,
            "detector_success_base": 1,
            "detector_success_anchor": 1,
            "detector_success_clean": 1,
        },
    ]

    summary = summarize_rows(rows)[0]

    assert summary["mean_pixel_mse_anchor"] == pytest.approx(2.0)
    assert summary["median_pixel_mse_anchor"] == pytest.approx(2.0)
    assert summary["mean_pixel_mse_clean"] == pytest.approx(0.5)
    assert summary["median_pixel_mse_clean"] == pytest.approx(0.5)
    assert summary["success_rate_base"] == pytest.approx(1.0)
    assert summary["success_rate_anchor"] == pytest.approx(0.5)
    assert summary["success_rate_clean"] == pytest.approx(1.0)


def test_adapter_missing_configuration_raises_clear_error() -> None:
    args = Namespace(gs_model_path=None, gs_config=None, gs_key=None)

    with pytest.raises((ValueError, NotImplementedError), match="Gaussian Shading.*gs"):
        load_gaussian_shading_pipeline(args)


def test_fake_pipeline_smoke_writes_results_summary_and_readme(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    img = np.stack(
        [
            xx / 63.0,
            yy / 63.0,
            0.5 + 0.25 * np.sin(2.0 * np.pi * (xx + yy) / 63.0),
        ],
        axis=-1,
    )
    Image.fromarray((np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(
        image_dir / "fake_gs.png"
    )
    outdir = tmp_path / "out"

    rc = main(
        [
            "--image-dir",
            str(image_dir),
            "--outdir",
            str(outdir),
            "--alpha",
            "0.15",
            "--size",
            "64",
            "--max-images",
            "1",
            "--num-angles",
            "180",
            "--num-r",
            "32",
            "--use-fake-gs-pipeline",
        ]
    )

    assert rc == 0
    results_path = outdir / "rotbind_gs_canonical_results.csv"
    summary_path = outdir / "summary.csv"
    readme_path = outdir / "README.md"
    grid_path = outdir / "example_grid.png"
    assert results_path.exists()
    assert summary_path.exists()
    assert readme_path.exists()
    assert grid_path.exists()

    import csv

    with results_path.open() as f:
        row = next(csv.DictReader(f))
    for field in RESULT_FIELDS:
        assert field in row
    assert float(row["pixel_mse_clean"]) < float(row["pixel_mse_anchor"])
    assert row["score_higher_is_better"] == "True"
