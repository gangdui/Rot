"""Tests for matched-geometry PSNR/SSIM quality metrics."""

from rotbind_anchor.eval_rotbind_anchor import summarize


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
