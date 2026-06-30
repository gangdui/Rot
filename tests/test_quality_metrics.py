"""Tests for matched-geometry PSNR/SSIM quality metrics."""

from argparse import Namespace
from pathlib import Path

import numpy as np

from rotbind_anchor.eval_rotbind_anchor import (
    ADDITIONAL_QUALITY_SUMMARY_FIELDS,
    SUMMARY_FIELDS,
    evaluate_one,
    psnr,
    simple_ssim,
    summarize,
)
from rotbind_anchor.rotbind_anchor import make_ring_pair_mask


QUALITY_FIELDS = [
    "psnr_anchor_rotated",
    "ssim_anchor_rotated",
    "psnr_corr_anchor_vs_corr_original",
    "ssim_corr_anchor_vs_corr_original",
    "psnr_clean_predcorr_vs_roundtrip_pred",
    "ssim_clean_predcorr_vs_roundtrip_pred",
    "psnr_oraclecorr_anchor_vs_roundtrip_oracle",
    "ssim_oraclecorr_anchor_vs_roundtrip_oracle",
    "psnr_clean_oraclecorr_vs_roundtrip_oracle",
    "ssim_clean_oraclecorr_vs_roundtrip_oracle",
    "psnr_clean_canonical",
    "ssim_clean_canonical",
    "psnr_postattack_clean_to_original",
    "ssim_postattack_clean_to_original",
]


def test_summary_reports_mean_and_median_for_matched_geometry_quality_fields() -> None:
    row = {
        "alpha": 0.15,
        "attack_rotation_backend": "torchvision",
        "attack_rotation_interpolation": "nearest",
        "attack_rotation_fill": 0.0,
        "correction_rotation_backend": "torchvision",
        "correction_rotation_interpolation": "bilinear",
        "correction_rotation_fill": 0.0,
        "rotation_error_deg": 0.1,
        "psnr_anchor": 40.0,
        "ssim_anchor": 0.99,
        "psnr_clean": 12.0,
        "ssim_clean": 0.5,
        "runtime_ms": 1.0,
    }
    for idx, field in enumerate(QUALITY_FIELDS):
        row[field] = float(idx + 1)

    summary = summarize([row], method="two_pair")[0]

    for field in QUALITY_FIELDS:
        assert summary[f"mean_{field}"] == row[field]
        assert summary[f"median_{field}"] == row[field]


def test_legacy_clean_quality_aliases_postattack_metric() -> None:
    row = {
        "alpha": 0.15,
        "attack_rotation_backend": "torchvision",
        "attack_rotation_interpolation": "nearest",
        "attack_rotation_fill": 0.0,
        "correction_rotation_backend": "torchvision",
        "correction_rotation_interpolation": "bilinear",
        "correction_rotation_fill": 0.0,
        "rotation_error_deg": 0.1,
        "psnr_anchor": 40.0,
        "ssim_anchor": 0.99,
        "psnr_clean": 12.0,
        "ssim_clean": 0.5,
        "psnr_postattack_clean_to_original": 12.0,
        "ssim_postattack_clean_to_original": 0.5,
        "runtime_ms": 1.0,
    }
    for field in QUALITY_FIELDS:
        row.setdefault(field, 1.0)

    assert row["psnr_clean"] == row["psnr_postattack_clean_to_original"]
    assert row["ssim_clean"] == row["ssim_postattack_clean_to_original"]


def test_summary_fields_cover_all_additional_quality_fields() -> None:
    for field in ADDITIONAL_QUALITY_SUMMARY_FIELDS:
        assert f"mean_{field}" in SUMMARY_FIELDS
        assert f"median_{field}" in SUMMARY_FIELDS


def test_evaluate_one_matched_geometry_quality_pairs() -> None:
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    x = xx / 63.0
    y = yy / 63.0
    img = np.stack(
        [
            0.2 + 0.6 * x,
            0.2 + 0.6 * y,
            0.5 + 0.25 * np.sin(2.0 * np.pi * (x + y)),
        ],
        axis=-1,
    ).astype(np.float32)
    modulation_grid, metadata = make_ring_pair_mask(64, 64, key=0, num_angles=180)
    args = Namespace(
        num_r=32,
        no_refine_peak=False,
        attack_rotation_backend="torchvision",
        attack_rotation_interpolation="nearest",
        attack_rotation_fill=0.0,
        correction_rotation_backend="torchvision",
        correction_rotation_interpolation="bilinear",
        correction_rotation_fill=0.0,
    )

    row, artifacts = evaluate_one(
        img,
        image_id="synthetic",
        image_path=Path("synthetic.png"),
        alpha=0.15,
        rotation_gt_deg=15.0,
        args=args,
        modulation_grid=modulation_grid,
        metadata=metadata,
        vae_encoder=None,
    )

    x_original = artifacts["x"]
    x_anchor = artifacts["x_anchor"]
    x_rot_original = artifacts["x_rot_original"]
    x_roundtrip_pred = artifacts["x_roundtrip_pred"]
    x_roundtrip_oracle = artifacts["x_roundtrip_oracle"]
    x_att = artifacts["x_att"]
    x_corr = artifacts["x_corr"]
    x_clean = artifacts["x_clean"]
    x_oracle_corr = artifacts["x_oracle_corr"]
    x_clean_oracle = artifacts["x_clean_oracle"]

    assert np.isclose(row["psnr_anchor"], psnr(x_original, x_anchor))
    assert np.isclose(row["psnr_anchor_rotated"], psnr(x_rot_original, x_att))
    assert np.isclose(row["psnr_corr_anchor_vs_corr_original"], psnr(x_roundtrip_pred, x_corr))
    assert np.isclose(row["psnr_clean_predcorr_vs_roundtrip_pred"], psnr(x_roundtrip_pred, x_clean))
    assert np.isclose(
        row["psnr_oraclecorr_anchor_vs_roundtrip_oracle"],
        psnr(x_roundtrip_oracle, x_oracle_corr),
    )
    assert np.isclose(
        row["psnr_clean_oraclecorr_vs_roundtrip_oracle"],
        psnr(x_roundtrip_oracle, x_clean_oracle),
    )
    assert np.isclose(row["ssim_corr_anchor_vs_corr_original"], simple_ssim(x_roundtrip_pred, x_corr))
    assert np.isclose(row["ssim_clean_predcorr_vs_roundtrip_pred"], simple_ssim(x_roundtrip_pred, x_clean))
    assert row["psnr_clean"] == row["psnr_postattack_clean_to_original"]
    assert row["ssim_clean"] == row["ssim_postattack_clean_to_original"]
