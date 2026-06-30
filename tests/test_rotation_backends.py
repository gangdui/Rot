"""Tests for benchmark-aligned same-size zero-fill rotation attack configuration."""

import numpy as np

from rotbind_anchor.eval_rotbind_anchor import summarize
from rotbind_anchor.rotbind_anchor import rotate_image_torchvision_keep_size


def test_torchvision_rotation_preserves_rgb_shape_dtype_and_range() -> None:
    img = np.zeros((17, 19, 3), dtype=np.float32)
    img[4:12, 6:14, 0] = 1.0
    img[8, 9, 1] = 0.5

    rotated = rotate_image_torchvision_keep_size(img, 17.0, interpolation="nearest", fill=0.0)

    assert rotated.shape == img.shape
    assert rotated.dtype == np.float32
    assert float(rotated.min()) >= 0.0
    assert float(rotated.max()) <= 1.0


def test_eval_parser_defaults_to_same_size_zfill_attack_and_bilinear_correction() -> None:
    from rotbind_anchor.eval_rotbind_anchor import parse_args

    args = parse_args(["--image", "dummy.png", "--outdir", "out"])

    assert args.attack_rotation_backend == "torchvision"
    assert args.attack_rotation_interpolation == "nearest"
    assert args.attack_rotation_fill == 0.0
    assert args.correction_rotation_backend == "torchvision"
    assert args.correction_rotation_interpolation == "bilinear"
    assert args.correction_rotation_fill == 0.0


def test_summary_records_rotation_attack_and_correction_settings() -> None:
    rows = [
        {
            "alpha": 0.15,
            "rotation_error_deg": 0.25,
            "psnr_anchor": 40.0,
            "psnr_clean": 38.0,
            "ssim_anchor": 0.99,
            "ssim_clean": 0.98,
            "runtime_ms": 1.0,
            "attack_rotation_backend": "torchvision",
            "attack_rotation_interpolation": "nearest",
            "attack_rotation_fill": 0.0,
            "correction_rotation_backend": "torchvision",
            "correction_rotation_interpolation": "bilinear",
            "correction_rotation_fill": 0.0,
        }
    ]

    summary = summarize(rows, method="two_pair")

    assert summary[0]["attack_rotation_backend"] == "torchvision"
    assert summary[0]["attack_rotation_interpolation"] == "nearest"
    assert summary[0]["correction_rotation_interpolation"] == "bilinear"
