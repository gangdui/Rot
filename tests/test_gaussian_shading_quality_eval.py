"""Tests for paper-aligned Gaussian Shading + RotBind quality evaluation."""

from __future__ import annotations

import csv
import json
import math
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from rotbind_anchor.eval_gaussian_shading_quality import (
    RESULT_FIELDS,
    import_open_clip,
    build_metadata_lookup,
    download_open_clip_checkpoint,
    load_metadata_jsonl,
    main,
    resolve_clip_pretrained,
    summarize_rows,
)


def save_rgb(path: Path, img: np.ndarray) -> None:
    """Save a float RGB image in [0, 1]."""
    Image.fromarray((np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)


def make_toy_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Create a tiny image directory plus Gaussian Shading-style metadata."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    for idx in range(2):
        img = np.stack(
            [
                xx / 63.0,
                yy / 63.0,
                0.35 + 0.2 * idx + 0.15 * np.sin(2.0 * np.pi * (xx + yy) / 63.0),
            ],
            axis=-1,
        )
        save_rgb(image_dir / f"{idx:06d}.png", img)

    metadata_path = tmp_path / "metadata.jsonl"
    rows = [
        {
            "image_id": "000000",
            "image_path": str(image_dir / "000000.png"),
            "prompt": "a high quality photo of a cat",
            "seed": 123,
        },
        {
            "image_id": "000001",
            "image_path": str(image_dir / "000001.png"),
            "prompt": "a high quality photo of a dog",
            "seed": 124,
        },
    ]
    metadata_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return image_dir, metadata_path


def test_metadata_reading_preserves_prompt_seed_and_lookup(tmp_path: Path) -> None:
    image_dir, metadata_path = make_toy_dataset(tmp_path)

    rows = load_metadata_jsonl(metadata_path)
    lookup = build_metadata_lookup(rows)

    assert rows[0]["prompt"] == "a high quality photo of a cat"
    assert rows[0]["seed"] == 123
    assert lookup["000000"]["prompt"] == "a high quality photo of a cat"
    assert lookup["000000.png"]["seed"] == 123
    assert lookup[str((image_dir / "000001.png").resolve())]["prompt"].endswith("dog")


def test_open_clip_import_can_use_vendored_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code_root = tmp_path / "vendored"
    package = code_root / "open_clip"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "def create_model_and_transforms(*args, **kwargs):\n"
        "    return None, None, None\n"
    )
    monkeypatch.delitem(sys.modules, "open_clip", raising=False)

    module = import_open_clip(code_root)

    assert hasattr(module, "create_model_and_transforms")
    assert str(code_root) in str(Path(module.__file__).as_posix())


def test_clip_pretrained_resolution_refuses_uncached_remote_weights() -> None:
    fake_open_clip = SimpleNamespace(
        pretrained=SimpleNamespace(
            get_pretrained_cfg=lambda model, tag: {
                "hf_hub": "laion/CLIP-ViT-g-14-laion2B-s34B-b88K/"
            }
        )
    )

    with pytest.raises(RuntimeError, match="not available in the local cache"):
        resolve_clip_pretrained(
            fake_open_clip,
            "ViT-g-14",
            "laion2b_s34b_b88k",
            allow_download=False,
        )


def test_clip_download_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_hf_hub_download(*args: object, **kwargs: object) -> str:
        raise RuntimeError("SSL EOF while reading")

    fake_hub = SimpleNamespace(hf_hub_download=failing_hf_hub_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    with pytest.raises(RuntimeError, match="CLIP weight download failed"):
        download_open_clip_checkpoint(
            SimpleNamespace(pretrained=SimpleNamespace(download_pretrained=lambda cfg: "")),
            {"hf_hub": "laion/CLIP-ViT-g-14-laion2B-s34B-b88K/"},
        )


def test_summary_ignores_inf_for_finite_mean_psnr_and_keeps_reference_fid_nan() -> None:
    rows = [
        {
            "pixel_mse_anchor": 1.0,
            "pixel_mse_clean": 0.0,
            "psnr_anchor": 30.0,
            "psnr_clean": float("inf"),
            "ssim_anchor": 0.9,
            "ssim_clean": 1.0,
            "lpips_anchor": float("nan"),
            "lpips_clean": float("nan"),
            "clip_score_baseline": 0.2,
            "clip_score_anchor": 0.19,
            "clip_score_clean": 0.2,
            "clip_score_delta_anchor": -0.01,
            "clip_score_delta_clean": 0.0,
        },
        {
            "pixel_mse_anchor": 3.0,
            "pixel_mse_clean": 0.01,
            "psnr_anchor": 40.0,
            "psnr_clean": 50.0,
            "ssim_anchor": 0.8,
            "ssim_clean": 0.99,
            "lpips_anchor": float("nan"),
            "lpips_clean": float("nan"),
            "clip_score_baseline": 0.4,
            "clip_score_anchor": 0.39,
            "clip_score_clean": 0.4,
            "clip_score_delta_anchor": -0.01,
            "clip_score_delta_clean": 0.0,
        },
    ]

    summary = summarize_rows(rows, fid_metrics={})[0]

    assert summary["num_images"] == 2
    assert summary["mean_pixel_mse_anchor"] == pytest.approx(2.0)
    assert math.isinf(summary["mean_psnr_clean"])
    assert summary["finite_mean_psnr_clean"] == pytest.approx(50.0)
    assert summary["finite_mean_psnr_anchor"] == pytest.approx(35.0)
    assert math.isnan(summary["fid_baseline_vs_reference"])


def test_fake_clip_smoke_writes_quality_outputs_and_complete_fields(tmp_path: Path) -> None:
    image_dir, metadata_path = make_toy_dataset(tmp_path)
    outdir = tmp_path / "quality_out"

    rc = main(
        [
            "--image-dir",
            str(image_dir),
            "--metadata",
            str(metadata_path),
            "--outdir",
            str(outdir),
            "--alpha",
            "0.15",
            "--method",
            "two_pair",
            "--key",
            "0",
            "--size",
            "64",
            "--max-images",
            "2",
            "--num-angles",
            "180",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--use-fake-clip",
            "--save-variant-images",
        ]
    )

    assert rc == 0
    results_path = outdir / "quality_results.csv"
    summary_path = outdir / "quality_summary.csv"
    readme_path = outdir / "README.md"
    grid_path = outdir / "example_quality_grid.png"
    assert results_path.exists()
    assert summary_path.exists()
    assert readme_path.exists()
    assert grid_path.exists()
    assert (outdir / "images" / "baseline" / "000000.png").exists()
    assert (outdir / "images" / "anchor" / "000000.png").exists()
    assert (outdir / "images" / "clean" / "000000.png").exists()

    with results_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    for field in RESULT_FIELDS:
        assert field in rows[0]
    assert rows[0]["prompt"] == "a high quality photo of a cat"
    assert rows[0]["seed"] == "123"
    assert rows[0]["lpips_anchor"] == "nan"
    assert rows[0]["lpips_clean"] == "nan"
    assert float(rows[0]["clip_score_delta_anchor"]) == pytest.approx(
        float(rows[0]["clip_score_anchor"]) - float(rows[0]["clip_score_baseline"])
    )

    with summary_path.open() as f:
        summary = next(csv.DictReader(f))
    assert summary["num_images"] == "2"
    assert summary["fid_baseline_vs_reference"] == "nan"
    assert "finite_mean_psnr_clean" in summary
    assert "RotBind incremental paired quality" in readme_path.read_text()
