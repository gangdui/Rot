"""Basic RotBind anchor evaluation script.

This script evaluates only the pixel-frequency anchor module. It does not use
VAE encoding, DDIM inversion, or any original latent watermark detector.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rotbind_anchor.rotbind_anchor import (  # noqa: E402
    circular_angle_error,
    circular_correlation_shift,
    detect_rotbind_angle,
    embed_rotbind_anchor_rgb,
    extract_metadata_ring_difference_signature,
    fft_polar_log_magnitude,
    make_negative_anchor_rgb,
    make_ring_pair_mask,
    remove_rotbind_anchor_rgb,
    rotate_image_keep_size,
    rotate_image_torchvision_keep_size,
    shift_to_attack_angle,
    wrap_angle_signed,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MATCHED_GEOMETRY_QUALITY_FIELDS = [
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
]
LEGACY_POSTATTACK_QUALITY_FIELDS = [
    "psnr_postattack_clean_to_original",
    "ssim_postattack_clean_to_original",
]
ADDITIONAL_QUALITY_SUMMARY_FIELDS = MATCHED_GEOMETRY_QUALITY_FIELDS + LEGACY_POSTATTACK_QUALITY_FIELDS
RESULT_FIELDS = [
    "image_id",
    "image_path",
    "alpha",
    "attack_rotation_backend",
    "attack_rotation_interpolation",
    "attack_rotation_fill",
    "correction_rotation_backend",
    "correction_rotation_interpolation",
    "correction_rotation_fill",
    "rotation_gt_deg",
    "rotation_gt_mod180_deg",
    "rotation_hat_deg",
    "rotation_hat_display_deg",
    "rotation_error_deg",
    "corr_shift_deg",
    "corr_shift_display_deg",
    "corr_shift_full_deg",
    "corr_shift_full_deg_int",
    "peak_refine_delta",
    "corr_period_deg",
    "best_score",
    "top2_score",
    "corr_margin",
    "ambiguity_resolved",
    "candidate_score_0",
    "candidate_score_180",
    "diff_corr_shift_deg",
    "diff_rotation_hat_deg",
    "diff_rotation_error_deg",
    "diff_rot_corr_shift_deg",
    "diff_rot_rotation_hat_deg",
    "diff_rot_rotation_error_deg",
    # Legacy angle fields kept temporarily for old analysis scripts.
    "theta_gt",
    "raw_theta_shift",
    "theta_attack_hat",
    "theta_hat_raw",
    "theta_hat",
    "angle_sign",
    "angle_error",
    "angle_error_mod180",
    "diff_theta_hat",
    "diff_angle_error",
    "diff_best_score",
    "diff_corr_margin",
    "diff_rot_theta_shift",
    "diff_rot_attack_hat",
    "diff_rot_attack_error",
    "diff_rot_theta_hat",
    "diff_rot_angle_error",
    "diff_rot_best_score",
    "diff_rot_corr_margin",
    "psnr_anchor",
    "ssim_anchor",
    *MATCHED_GEOMETRY_QUALITY_FIELDS,
    *LEGACY_POSTATTACK_QUALITY_FIELDS,
    # Legacy/post-attack full-image quality; dominated by rotation roundtrip artifacts.
    "psnr_clean",
    "ssim_clean",
    "vae_mse_anchor",
    "vae_mse_clean",
    "vae_mse_symmetric",
    "vae_mse_clean_canonical",
    "vae_mse_clean_predcorr",
    "vae_mse_clean_oraclecorr",
    "vae_mse_symmetric_canonical",
    "vae_mse_symmetric_predcorr",
    "vae_mse_symmetric_oraclecorr",
    "vae_mse_roundtrip_oracle",
    "vae_mse_anchor_removed_extra",
    "vae_raw_mse_anchor",
    "vae_raw_mse_clean_canonical",
    "vae_raw_mse_clean_predcorr",
    "vae_raw_mse_clean_oraclecorr",
    "vae_raw_mse_roundtrip_oracle",
    "vae_rel_mse_anchor",
    "vae_rel_mse_clean_canonical",
    "vae_rel_mse_clean_predcorr",
    "vae_rel_mse_clean_oraclecorr",
    "vae_rel_mse_roundtrip_oracle",
    "vae_rel_mse_anchor_removed_extra",
    "vae_cos_anchor",
    "vae_cos_clean",
    "vae_cos_symmetric",
    "vae_cos_clean_canonical",
    "vae_cos_clean_predcorr",
    "vae_cos_clean_oraclecorr",
    "vae_cos_symmetric_canonical",
    "vae_cos_symmetric_predcorr",
    "vae_cos_symmetric_oraclecorr",
    "vae_cos_roundtrip_oracle",
    "runtime_ms",
]
SUMMARY_FIELDS = [
    "method",
    "alpha",
    "attack_rotation_backend",
    "attack_rotation_interpolation",
    "attack_rotation_fill",
    "correction_rotation_backend",
    "correction_rotation_interpolation",
    "correction_rotation_fill",
    "num_samples",
    "mean_rotation_error_deg",
    "median_rotation_error_deg",
    "max_rotation_error_deg",
    "failure_rate_rotation_error_gt_1deg",
    "failure_rate_rotation_error_gt_3deg",
    "mean_psnr_anchor",
    "mean_psnr_anchor_finite",
    "median_psnr_anchor",
    "num_inf_psnr_anchor",
    "mean_ssim_anchor",
    "median_ssim_anchor",
    "mean_psnr_clean",
    "mean_psnr_clean_finite",
    "median_psnr_clean",
    "num_inf_psnr_clean",
    "mean_ssim_clean",
    "median_ssim_clean",
    *[
        stat_name
        for field_name in ADDITIONAL_QUALITY_SUMMARY_FIELDS
        for stat_name in (f"mean_{field_name}", f"median_{field_name}")
    ],
    "mean_runtime_ms",
]
DIAGNOSTIC_SUMMARY_FIELDS = [
    "diagnostic_alpha",
    "num_images",
    "mean_diff_rotation_error_deg",
    "median_diff_rotation_error_deg",
    "max_diff_rotation_error_deg",
    "failure_rate_diff_rotation_error_gt_3deg",
    "mean_diff_best_score",
    "mean_diff_corr_margin",
]


def parse_float_list(text: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def load_rgb_image(path: Path, size: int, no_resize: bool = False) -> np.ndarray:
    """Load an image as float32 RGB in [0, 1]."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if not no_resize:
            im = im.resize((int(size), int(size)), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def save_rgb_image(path: Path, img: np.ndarray) -> None:
    """Save a float RGB image in [0, 1]."""
    arr = np.clip(img, 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


def discover_images(args: argparse.Namespace) -> list[Path]:
    """Return image paths from --image or --image-dir."""
    paths: list[Path] = []
    if args.image:
        paths.append(Path(args.image))
    if args.image_dir:
        root = Path(args.image_dir)
        if not root.exists():
            raise FileNotFoundError(f"image directory not found: {root}")
        paths.extend(sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS))
    if args.max_images is not None:
        paths = paths[: int(args.max_images)]
    if not paths:
        raise ValueError("no input images found; use --image or --image-dir")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"input image not found: {missing[0]}")
    return paths


def psnr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Compute PSNR for [0, 1] images."""
    mse = float(np.mean((x.astype(np.float32) - y.astype(np.float32)) ** 2))
    if mse <= eps:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def simple_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Compute a simple global SSIM averaged over RGB channels."""
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    c1 = 0.01**2
    c2 = 0.03**2
    vals = []
    for c in range(x64.shape[2]):
        a = x64[..., c]
        b = y64[..., c]
        mu_a = float(a.mean())
        mu_b = float(b.mean())
        var_a = float(((a - mu_a) ** 2).mean())
        var_b = float(((b - mu_b) ** 2).mean())
        cov = float(((a - mu_a) * (b - mu_b)).mean())
        num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)
        den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
        vals.append(num / den if den != 0 else 1.0)
    return float(np.mean(vals))


class DiffusersVaeEncoder:
    """Small deterministic wrapper around diffusers AutoencoderKL.encode."""

    def __init__(self, model: Any, device: str = "cpu", use_scaling_factor: bool = True) -> None:
        self.model = model
        self.device = device
        self.scaling_factor = float(getattr(model.config, "scaling_factor", 0.18215))
        self.use_scaling_factor = bool(use_scaling_factor)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = "auto",
        local_files_only: bool = False,
        subfolder: str | None = None,
    ) -> "DiffusersVaeEncoder":
        """Load a diffusers AutoencoderKL model."""
        import torch
        from diffusers import AutoencoderKL

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if subfolder:
            kwargs["subfolder"] = subfolder
        model = AutoencoderKL.from_pretrained(model_name_or_path, **kwargs)
        model = model.to(device)
        model.eval()
        return cls(model, device=device)

    def encode_images_raw(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode RGB float images in [0, 1] into flattened raw latent means."""
        import torch

        batch = np.stack(images, axis=0).astype(np.float32)
        batch = np.clip(batch, 0.0, 1.0) * 2.0 - 1.0
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            encoded = self.model.encode(tensor)
            latent = encoded.latent_dist.mean
        return latent.detach().float().cpu().numpy().reshape(len(images), -1).astype(np.float32)

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode images into flattened Stable Diffusion scaled latents by default."""
        latents = self.encode_images_raw(images)
        if self.use_scaling_factor:
            latents = latents * self.scaling_factor
        return latents.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Return finite cosine similarity between two flattened vectors."""
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= eps:
        return 0.0
    return float(np.dot(av, bv) / denom)


def _encode_images_raw_if_available(vae_encoder: Any, images: list[np.ndarray]) -> np.ndarray:
    """Return raw latent means when the encoder exposes them, otherwise use encode_images."""
    if hasattr(vae_encoder, "encode_images_raw"):
        return vae_encoder.encode_images_raw(images)
    return vae_encoder.encode_images(images)


def compute_vae_footprint_metrics(
    vae_encoder: Any,
    x_original: np.ndarray,
    x_anchor: np.ndarray,
    x_clean: np.ndarray,
    x_minus: np.ndarray,
) -> dict[str, float]:
    """Compute VAE latent MSE and cosine footprint metrics."""
    z_original, z_anchor, z_clean, z_minus = vae_encoder.encode_images(
        [x_original, x_anchor, x_clean, x_minus]
    )
    raw_original, raw_anchor, raw_clean, raw_minus = _encode_images_raw_if_available(
        vae_encoder,
        [x_original, x_anchor, x_clean, x_minus],
    )
    z_symmetric = (z_anchor + z_minus) / 2.0
    raw_symmetric = (raw_anchor + raw_minus) / 2.0

    def mse(z: np.ndarray, z0: np.ndarray = z_original) -> float:
        return float(np.mean((np.asarray(z, dtype=np.float32) - z0) ** 2))

    def raw_mse(z: np.ndarray) -> float:
        return mse(z, raw_original)

    def rel_mse(z: np.ndarray) -> float:
        return mse(z) / (float(np.mean(np.asarray(z_original, dtype=np.float32) ** 2)) + 1e-12)

    return {
        "vae_mse_anchor": mse(z_anchor),
        "vae_mse_clean": mse(z_clean),
        "vae_mse_symmetric": mse(z_symmetric),
        "vae_raw_mse_anchor": raw_mse(raw_anchor),
        "vae_raw_mse_clean_canonical": raw_mse(raw_clean),
        "vae_raw_mse_clean_predcorr": float("nan"),
        "vae_raw_mse_clean_oraclecorr": float("nan"),
        "vae_raw_mse_roundtrip_oracle": float("nan"),
        "vae_rel_mse_anchor": rel_mse(z_anchor),
        "vae_rel_mse_clean_canonical": rel_mse(z_clean),
        "vae_rel_mse_clean_predcorr": float("nan"),
        "vae_rel_mse_clean_oraclecorr": float("nan"),
        "vae_rel_mse_roundtrip_oracle": float("nan"),
        "vae_rel_mse_anchor_removed_extra": float("nan"),
        "vae_cos_anchor": cosine_similarity(z_anchor, z_original),
        "vae_cos_clean": cosine_similarity(z_clean, z_original),
        "vae_cos_symmetric": cosine_similarity(z_symmetric, z_original),
    }


def compute_rotated_corrected_vae_metrics(
    vae_encoder: Any,
    x_original: np.ndarray,
    x_plus_canonical: np.ndarray,
    x_plus_predcorr: np.ndarray,
    x_plus_oraclecorr: np.ndarray,
    x_clean_canonical: np.ndarray,
    x_clean_predcorr: np.ndarray,
    x_clean_oraclecorr: np.ndarray,
    x_minus_canonical: np.ndarray,
    x_minus_predcorr: np.ndarray,
    x_minus_oraclecorr: np.ndarray,
    x_roundtrip_oracle: np.ndarray,
) -> dict[str, float]:
    """Compute VAE metrics for canonical, predicted-corrected, and oracle-corrected variants."""
    images = [
        x_original,
        x_plus_canonical,
        x_plus_predcorr,
        x_plus_oraclecorr,
        x_clean_canonical,
        x_clean_predcorr,
        x_clean_oraclecorr,
        x_minus_canonical,
        x_minus_predcorr,
        x_minus_oraclecorr,
        x_roundtrip_oracle,
    ]
    (
        z_original,
        z_plus_canonical,
        z_plus_predcorr,
        z_plus_oraclecorr,
        z_clean_canonical,
        z_clean_predcorr,
        z_clean_oraclecorr,
        z_minus_canonical,
        z_minus_predcorr,
        z_minus_oraclecorr,
        z_roundtrip_oracle,
    ) = vae_encoder.encode_images(images)
    (
        raw_original,
        raw_plus_canonical,
        raw_plus_predcorr,
        raw_plus_oraclecorr,
        raw_clean_canonical,
        raw_clean_predcorr,
        raw_clean_oraclecorr,
        raw_minus_canonical,
        raw_minus_predcorr,
        raw_minus_oraclecorr,
        raw_roundtrip_oracle,
    ) = _encode_images_raw_if_available(vae_encoder, images)
    z_symmetric_canonical = (z_plus_canonical + z_minus_canonical) / 2.0
    z_symmetric_predcorr = (z_plus_predcorr + z_minus_predcorr) / 2.0
    z_symmetric_oraclecorr = (z_plus_oraclecorr + z_minus_oraclecorr) / 2.0

    def mse(z: np.ndarray, z0: np.ndarray = z_original) -> float:
        return float(np.mean((np.asarray(z, dtype=np.float32) - z0) ** 2))

    def raw_mse(z: np.ndarray) -> float:
        return mse(z, raw_original)

    z_original_energy = float(np.mean(np.asarray(z_original, dtype=np.float32) ** 2)) + 1e-12

    def rel_mse(z: np.ndarray) -> float:
        return mse(z) / z_original_energy

    metrics = {
        "vae_mse_anchor": mse(z_plus_canonical),
        "vae_mse_clean_canonical": mse(z_clean_canonical),
        "vae_mse_clean_predcorr": mse(z_clean_predcorr),
        "vae_mse_clean_oraclecorr": mse(z_clean_oraclecorr),
        "vae_mse_symmetric_canonical": mse(z_symmetric_canonical),
        "vae_mse_symmetric_predcorr": mse(z_symmetric_predcorr),
        "vae_mse_symmetric_oraclecorr": mse(z_symmetric_oraclecorr),
        "vae_mse_roundtrip_oracle": mse(z_roundtrip_oracle),
        "vae_raw_mse_anchor": raw_mse(raw_plus_canonical),
        "vae_raw_mse_clean_canonical": raw_mse(raw_clean_canonical),
        "vae_raw_mse_clean_predcorr": raw_mse(raw_clean_predcorr),
        "vae_raw_mse_clean_oraclecorr": raw_mse(raw_clean_oraclecorr),
        "vae_raw_mse_roundtrip_oracle": raw_mse(raw_roundtrip_oracle),
        "vae_rel_mse_anchor": rel_mse(z_plus_canonical),
        "vae_rel_mse_clean_canonical": rel_mse(z_clean_canonical),
        "vae_rel_mse_clean_predcorr": rel_mse(z_clean_predcorr),
        "vae_rel_mse_clean_oraclecorr": rel_mse(z_clean_oraclecorr),
        "vae_rel_mse_roundtrip_oracle": rel_mse(z_roundtrip_oracle),
        "vae_cos_anchor": cosine_similarity(z_plus_canonical, z_original),
        "vae_cos_clean_canonical": cosine_similarity(z_clean_canonical, z_original),
        "vae_cos_clean_predcorr": cosine_similarity(z_clean_predcorr, z_original),
        "vae_cos_clean_oraclecorr": cosine_similarity(z_clean_oraclecorr, z_original),
        "vae_cos_symmetric_canonical": cosine_similarity(z_symmetric_canonical, z_original),
        "vae_cos_symmetric_predcorr": cosine_similarity(z_symmetric_predcorr, z_original),
        "vae_cos_symmetric_oraclecorr": cosine_similarity(z_symmetric_oraclecorr, z_original),
        "vae_cos_roundtrip_oracle": cosine_similarity(z_roundtrip_oracle, z_original),
    }
    metrics["vae_mse_anchor_removed_extra"] = (
        metrics["vae_mse_clean_oraclecorr"] - metrics["vae_mse_roundtrip_oracle"]
    )
    metrics["vae_rel_mse_anchor_removed_extra"] = (
        metrics["vae_rel_mse_clean_oraclecorr"] - metrics["vae_rel_mse_roundtrip_oracle"]
    )
    metrics["vae_mse_clean"] = metrics["vae_mse_clean_canonical"]
    metrics["vae_mse_symmetric"] = metrics["vae_mse_symmetric_canonical"]
    metrics["vae_cos_clean"] = metrics["vae_cos_clean_canonical"]
    metrics["vae_cos_symmetric"] = metrics["vae_cos_symmetric_canonical"]
    return metrics


def nan_vae_metrics() -> dict[str, float]:
    """Return NaN placeholders for VAE metrics when disabled."""
    return {
        "vae_mse_anchor": float("nan"),
        "vae_mse_clean": float("nan"),
        "vae_mse_symmetric": float("nan"),
        "vae_mse_clean_canonical": float("nan"),
        "vae_mse_clean_predcorr": float("nan"),
        "vae_mse_clean_oraclecorr": float("nan"),
        "vae_mse_symmetric_canonical": float("nan"),
        "vae_mse_symmetric_predcorr": float("nan"),
        "vae_mse_symmetric_oraclecorr": float("nan"),
        "vae_mse_roundtrip_oracle": float("nan"),
        "vae_mse_anchor_removed_extra": float("nan"),
        "vae_raw_mse_anchor": float("nan"),
        "vae_raw_mse_clean_canonical": float("nan"),
        "vae_raw_mse_clean_predcorr": float("nan"),
        "vae_raw_mse_clean_oraclecorr": float("nan"),
        "vae_raw_mse_roundtrip_oracle": float("nan"),
        "vae_rel_mse_anchor": float("nan"),
        "vae_rel_mse_clean_canonical": float("nan"),
        "vae_rel_mse_clean_predcorr": float("nan"),
        "vae_rel_mse_clean_oraclecorr": float("nan"),
        "vae_rel_mse_roundtrip_oracle": float("nan"),
        "vae_rel_mse_anchor_removed_extra": float("nan"),
        "vae_cos_anchor": float("nan"),
        "vae_cos_clean": float("nan"),
        "vae_cos_symmetric": float("nan"),
        "vae_cos_clean_canonical": float("nan"),
        "vae_cos_clean_predcorr": float("nan"),
        "vae_cos_clean_oraclecorr": float("nan"),
        "vae_cos_symmetric_canonical": float("nan"),
        "vae_cos_symmetric_predcorr": float("nan"),
        "vae_cos_symmetric_oraclecorr": float("nan"),
        "vae_cos_roundtrip_oracle": float("nan"),
    }


def normalize_angle(theta: float) -> float:
    """Normalize angle to [0, 360)."""
    return float(theta % 360.0)


def apply_angle_sign(theta_shift: float, angle_sign: str, angle_period: float = 180.0) -> float:
    """Deprecated: convert a correlation shift into an image rotation angle."""
    if angle_sign == "raw":
        return shift_to_attack_angle(theta_shift, angle_period=angle_period)
    if angle_sign == "neg":
        return float(theta_shift % angle_period)
    raise ValueError("angle_sign must be raw or neg")


def rotate_with_config(
    img: np.ndarray,
    angle: float,
    backend: str,
    interpolation: str,
    fill: float,
) -> np.ndarray:
    """Rotate an image using an experiment-configured backend."""
    if backend == "torchvision":
        return rotate_image_torchvision_keep_size(
            img,
            angle,
            interpolation=interpolation,
            fill=fill,
        )
    if backend == "scipy":
        return rotate_image_keep_size(img, angle, mode="reflect", order=1)
    raise ValueError("rotation backend must be 'torchvision' or 'scipy'")


def make_synthetic_images(size: int) -> list[np.ndarray]:
    """Create simple synthetic calibration images."""
    n = int(size)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    x = xx / max(n - 1, 1)
    y = yy / max(n - 1, 1)

    rng = np.random.default_rng(1234)
    smooth = rng.random((max(n // 8, 4), max(n // 8, 4), 3), dtype=np.float32)
    smooth_img = np.asarray(
        Image.fromarray((smooth * 255).astype(np.uint8)).resize((n, n), Image.BILINEAR),
        dtype=np.float32,
    )
    smooth_img = smooth_img / 255.0

    checker = (((xx // max(n // 8, 1)) + (yy // max(n // 8, 1))) % 2).astype(np.float32)
    checker_img = np.stack([0.25 + 0.5 * x, 0.25 + 0.5 * y, 0.2 + 0.6 * checker], axis=-1)

    lowfreq = np.stack(
        [
            0.5 + 0.25 * np.sin(2.0 * np.pi * (x + 0.2 * y)),
            0.5 + 0.25 * np.cos(2.0 * np.pi * (0.3 * x + y)),
            0.5 + 0.2 * np.sin(2.0 * np.pi * (x + y)),
        ],
        axis=-1,
    )
    return [
        np.clip(smooth_img, 0.0, 1.0).astype(np.float32),
        np.clip(checker_img, 0.0, 1.0).astype(np.float32),
        np.clip(lowfreq, 0.0, 1.0).astype(np.float32),
    ]


def calibrate_angle_sign(args: argparse.Namespace) -> tuple[str, dict[str, float]]:
    """Deprecated compatibility stub; RotBind now always uses raw convention."""
    return "raw", {"raw_mean_error": float("nan"), "neg_mean_error": float("nan")}


def ambiguity_fields(info: dict[str, Any]) -> tuple[bool, float, float]:
    """Extract ambiguity score fields from detector info."""
    amb = info.get("ambiguity") if isinstance(info, dict) else None
    if not isinstance(amb, dict):
        return False, float("nan"), float("nan")
    return True, float(amb.get("score1", float("nan"))), float(amb.get("score2", float("nan")))


def diagnostic_diff_feature(
    img: np.ndarray,
    x_anchor: np.ndarray,
    metadata: dict[str, Any],
    num_r: int,
    expected_theta: float = 0.0,
    refine_peak: bool = True,
) -> dict[str, float]:
    """Correlate PolarFeature(x_anchor)-PolarFeature(x) against the angular code."""
    pos_bands = metadata["pos_bands"]
    neg_bands = metadata["neg_bands"]
    all_bands = pos_bands + neg_bands
    rmin = min(float(b[0]) for b in all_bands)
    rmax = max(float(b[1]) for b in all_bands)
    polar_anchor, info = fft_polar_log_magnitude(
        x_anchor,
        rmin,
        rmax,
        num_r=int(num_r),
        num_angles=int(metadata["num_angles"]),
        normalize="per_radius",
    )
    polar_original, _ = fft_polar_log_magnitude(
        img,
        rmin,
        rmax,
        num_r=int(num_r),
        num_angles=int(metadata["num_angles"]),
        normalize="per_radius",
    )
    diff_feature = polar_anchor - polar_original
    signature, _ = extract_metadata_ring_difference_signature(diff_feature, info["r_values"], metadata)
    angle_period = 180.0 if bool(metadata.get("pi_periodic", True)) else 360.0
    corr_shift_deg, best_score, _, corr_info = circular_correlation_shift(
        signature,
        metadata["angular_code"],
        angle_period=angle_period,
        refine_peak=refine_peak,
    )
    rotation_hat_deg = shift_to_attack_angle(corr_shift_deg, angle_period=angle_period)
    rotation_error_deg = circular_angle_error(rotation_hat_deg, expected_theta, period=angle_period)
    return {
        "diff_corr_shift_deg": float(corr_shift_deg),
        "diff_rotation_hat_deg": float(rotation_hat_deg),
        "diff_rotation_error_deg": rotation_error_deg,
        "diff_theta_shift": float(corr_shift_deg),
        "diff_attack_hat": float(rotation_hat_deg),
        "diff_attack_error": rotation_error_deg,
        "diff_theta_hat": float(rotation_hat_deg),
        "diff_angle_error": rotation_error_deg,
        "diff_best_score": float(best_score),
        "diff_corr_margin": float(corr_info.get("corr_margin", float("nan"))),
    }


def evaluate_one(
    img: np.ndarray,
    image_id: str,
    image_path: Path,
    alpha: float,
    rotation_gt_deg: float,
    args: argparse.Namespace,
    modulation_grid: np.ndarray,
    metadata: dict[str, Any],
    vae_encoder: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one image/alpha/angle sample and return CSV row plus artifacts."""
    start = time.perf_counter()
    refine_peak = not bool(args.no_refine_peak)
    x_anchor = embed_rotbind_anchor_rgb(img, modulation_grid, alpha)
    x_clean_canonical = remove_rotbind_anchor_rgb(x_anchor, modulation_grid, alpha)
    x_minus_canonical = make_negative_anchor_rgb(x_anchor, modulation_grid, alpha)
    diff_info = diagnostic_diff_feature(img, x_anchor, metadata, int(args.num_r), refine_peak=refine_peak)
    x_att = rotate_with_config(
        x_anchor,
        rotation_gt_deg,
        args.attack_rotation_backend,
        args.attack_rotation_interpolation,
        float(args.attack_rotation_fill),
    )
    x_rot_original = rotate_with_config(
        img,
        rotation_gt_deg,
        args.attack_rotation_backend,
        args.attack_rotation_interpolation,
        float(args.attack_rotation_fill),
    )
    x_roundtrip_oracle = rotate_with_config(
        x_rot_original,
        -float(rotation_gt_deg),
        args.correction_rotation_backend,
        args.correction_rotation_interpolation,
        float(args.correction_rotation_fill),
    )
    x_roundtrip_oracle = np.clip(x_roundtrip_oracle, 0.0, 1.0).astype(np.float32)
    diff_rot_info = diagnostic_diff_feature(
        np.clip(x_rot_original, 0.0, 1.0).astype(np.float32),
        np.clip(x_att, 0.0, 1.0).astype(np.float32),
        metadata,
        int(args.num_r),
        expected_theta=float(rotation_gt_deg),
        refine_peak=refine_peak,
    )
    angle_period = 180.0 if bool(metadata.get("pi_periodic", True)) else 360.0
    rotation_hat_deg, best_score, score_curve, info = detect_rotbind_angle(
        np.clip(x_att, 0.0, 1.0).astype(np.float32),
        metadata,
        num_r=int(args.num_r),
        resolve_ambiguity=False,
        refine_peak=refine_peak,
    )
    corr_shift_deg = float(info.get("corr_shift_deg", float("nan")))
    rotation_error_deg = circular_angle_error(rotation_hat_deg, rotation_gt_deg, period=angle_period)
    x_roundtrip_pred = rotate_with_config(
        x_rot_original,
        -float(rotation_hat_deg),
        args.correction_rotation_backend,
        args.correction_rotation_interpolation,
        float(args.correction_rotation_fill),
    )
    x_roundtrip_pred = np.clip(x_roundtrip_pred, 0.0, 1.0).astype(np.float32)
    x_corr = rotate_with_config(
        x_att,
        -rotation_hat_deg,
        args.correction_rotation_backend,
        args.correction_rotation_interpolation,
        float(args.correction_rotation_fill),
    )
    x_corr = np.clip(x_corr, 0.0, 1.0).astype(np.float32)
    x_clean = remove_rotbind_anchor_rgb(x_corr, modulation_grid, alpha)
    x_minus = make_negative_anchor_rgb(x_corr, modulation_grid, alpha)
    x_oracle_corr = rotate_with_config(
        x_att,
        -float(rotation_gt_deg),
        args.correction_rotation_backend,
        args.correction_rotation_interpolation,
        float(args.correction_rotation_fill),
    )
    x_oracle_corr = np.clip(x_oracle_corr, 0.0, 1.0).astype(np.float32)
    x_clean_oracle = remove_rotbind_anchor_rgb(x_oracle_corr, modulation_grid, alpha)
    x_minus_oracle = make_negative_anchor_rgb(x_oracle_corr, modulation_grid, alpha)
    vae_metrics = (
        compute_rotated_corrected_vae_metrics(
            vae_encoder,
            img,
            x_anchor,
            x_corr,
            x_oracle_corr,
            x_clean_canonical,
            x_clean,
            x_clean_oracle,
            x_minus_canonical,
            x_minus,
            x_minus_oracle,
            x_roundtrip_oracle,
        )
        if vae_encoder is not None
        else nan_vae_metrics()
    )
    runtime_ms = (time.perf_counter() - start) * 1000.0

    ambiguity_resolved, cand0, cand180 = ambiguity_fields(info)
    psnr_postattack_clean_to_original = psnr(img, x_clean)
    ssim_postattack_clean_to_original = simple_ssim(img, x_clean)
    row: dict[str, Any] = {
        "image_id": image_id,
        "image_path": str(image_path),
        "alpha": float(alpha),
        "attack_rotation_backend": args.attack_rotation_backend,
        "attack_rotation_interpolation": args.attack_rotation_interpolation,
        "attack_rotation_fill": float(args.attack_rotation_fill),
        "correction_rotation_backend": args.correction_rotation_backend,
        "correction_rotation_interpolation": args.correction_rotation_interpolation,
        "correction_rotation_fill": float(args.correction_rotation_fill),
        "rotation_gt_deg": float(rotation_gt_deg),
        "rotation_gt_mod180_deg": float(rotation_gt_deg % 180.0),
        "rotation_hat_deg": float(rotation_hat_deg),
        "rotation_hat_display_deg": wrap_angle_signed(rotation_hat_deg, period=angle_period),
        "rotation_error_deg": rotation_error_deg,
        "corr_shift_deg": corr_shift_deg,
        "corr_shift_display_deg": wrap_angle_signed(corr_shift_deg, period=angle_period),
        "corr_shift_full_deg": float(info.get("corr_shift_full_deg", float("nan"))),
        "corr_shift_full_deg_int": float(info.get("corr_shift_full_deg_int", float("nan"))),
        "peak_refine_delta": float(info.get("peak_refine_delta", float("nan"))),
        "corr_period_deg": float(info.get("corr_period_deg", angle_period)),
        "best_score": float(best_score),
        "top2_score": float(info.get("top2_score", float("nan"))),
        "corr_margin": float(info.get("corr_margin", float("nan"))),
        "ambiguity_resolved": ambiguity_resolved,
        "candidate_score_0": cand0,
        "candidate_score_180": cand180,
        "diff_corr_shift_deg": diff_info["diff_corr_shift_deg"],
        "diff_rotation_hat_deg": diff_info["diff_rotation_hat_deg"],
        "diff_rotation_error_deg": diff_info["diff_rotation_error_deg"],
        "diff_rot_corr_shift_deg": diff_rot_info["diff_corr_shift_deg"],
        "diff_rot_rotation_hat_deg": diff_rot_info["diff_rotation_hat_deg"],
        "diff_rot_rotation_error_deg": diff_rot_info["diff_rotation_error_deg"],
        # Legacy angle fields kept temporarily for old analysis scripts.
        "theta_gt": float(rotation_gt_deg),
        "raw_theta_shift": corr_shift_deg,
        "theta_attack_hat": float(rotation_hat_deg),
        "theta_hat_raw": corr_shift_deg,
        "theta_hat": float(rotation_hat_deg),
        "angle_sign": "raw",
        "angle_error": rotation_error_deg,
        "angle_error_mod180": circular_angle_error(rotation_hat_deg, rotation_gt_deg, period=180.0),
        "diff_theta_hat": diff_info["diff_theta_hat"],
        "diff_angle_error": diff_info["diff_angle_error"],
        "diff_best_score": diff_info["diff_best_score"],
        "diff_corr_margin": diff_info["diff_corr_margin"],
        "diff_rot_theta_shift": diff_rot_info["diff_theta_shift"],
        "diff_rot_attack_hat": diff_rot_info["diff_attack_hat"],
        "diff_rot_attack_error": diff_rot_info["diff_attack_error"],
        "diff_rot_theta_hat": diff_rot_info["diff_attack_hat"],
        "diff_rot_angle_error": diff_rot_info["diff_attack_error"],
        "diff_rot_best_score": diff_rot_info["diff_best_score"],
        "diff_rot_corr_margin": diff_rot_info["diff_corr_margin"],
        "psnr_anchor": psnr(img, x_anchor),
        "ssim_anchor": simple_ssim(img, x_anchor),
        "psnr_anchor_rotated": psnr(x_rot_original, x_att),
        "ssim_anchor_rotated": simple_ssim(x_rot_original, x_att),
        "psnr_corr_anchor_vs_corr_original": psnr(x_roundtrip_pred, x_corr),
        "ssim_corr_anchor_vs_corr_original": simple_ssim(x_roundtrip_pred, x_corr),
        "psnr_clean_predcorr_vs_roundtrip_pred": psnr(x_roundtrip_pred, x_clean),
        "ssim_clean_predcorr_vs_roundtrip_pred": simple_ssim(x_roundtrip_pred, x_clean),
        "psnr_oraclecorr_anchor_vs_roundtrip_oracle": psnr(x_roundtrip_oracle, x_oracle_corr),
        "ssim_oraclecorr_anchor_vs_roundtrip_oracle": simple_ssim(x_roundtrip_oracle, x_oracle_corr),
        "psnr_clean_oraclecorr_vs_roundtrip_oracle": psnr(x_roundtrip_oracle, x_clean_oracle),
        "ssim_clean_oraclecorr_vs_roundtrip_oracle": simple_ssim(x_roundtrip_oracle, x_clean_oracle),
        "psnr_clean_canonical": psnr(img, x_clean_canonical),
        "ssim_clean_canonical": simple_ssim(img, x_clean_canonical),
        "psnr_postattack_clean_to_original": psnr_postattack_clean_to_original,
        "ssim_postattack_clean_to_original": ssim_postattack_clean_to_original,
        "psnr_clean": psnr_postattack_clean_to_original,
        "ssim_clean": ssim_postattack_clean_to_original,
        **vae_metrics,
        "runtime_ms": runtime_ms,
    }
    artifacts = {
        "x": img,
        "x_anchor": x_anchor,
        "x_att": np.clip(x_att, 0.0, 1.0).astype(np.float32),
        "x_rot_original": np.clip(x_rot_original, 0.0, 1.0).astype(np.float32),
        "x_roundtrip_pred": x_roundtrip_pred,
        "x_roundtrip_oracle": x_roundtrip_oracle,
        "x_corr": x_corr,
        "x_clean": x_clean,
        "x_oracle_corr": x_oracle_corr,
        "x_clean_oracle": x_clean_oracle,
        "x_minus": x_minus,
        "score_curve": score_curve,
        "rotation_gt_deg": rotation_gt_deg,
        "rotation_hat_deg": rotation_hat_deg,
        "rotation_error_deg": rotation_error_deg,
        "corr_shift_deg": corr_shift_deg,
        "attack_rotation_backend": args.attack_rotation_backend,
        "attack_rotation_interpolation": args.attack_rotation_interpolation,
        "attack_rotation_fill": float(args.attack_rotation_fill),
        "correction_rotation_backend": args.correction_rotation_backend,
        "correction_rotation_interpolation": args.correction_rotation_interpolation,
        "correction_rotation_fill": float(args.correction_rotation_fill),
        # Legacy aliases.
        "theta_gt": rotation_gt_deg,
        "theta_hat": rotation_hat_deg,
        "alpha": alpha,
    }
    return row, artifacts


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV with stable field order."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, float("nan")) for field in fields})


def finite_stats(values: np.ndarray) -> tuple[float, float, int]:
    """Return finite mean, finite median, and number of infinite values."""
    finite = values[np.isfinite(values)]
    num_inf = int(np.isinf(values).sum())
    if finite.size == 0:
        return float("nan"), float("nan"), num_inf
    return float(np.nanmean(finite)), float(np.nanmedian(finite)), num_inf


def summarize(rows: list[dict[str, Any]], method: str = "unknown") -> list[dict[str, Any]]:
    """Aggregate result rows by alpha."""
    summary = []
    alphas = sorted({float(row["alpha"]) for row in rows})
    for alpha in alphas:
        group = [row for row in rows if float(row["alpha"]) == alpha]

        def arr(name: str) -> np.ndarray:
            return np.asarray([float(row[name]) for row in group], dtype=np.float64)

        rotation_error = arr("rotation_error_deg")
        psnr_anchor = arr("psnr_anchor")
        psnr_clean = arr("psnr_clean")
        psnr_anchor_finite_mean, psnr_anchor_median, psnr_anchor_num_inf = finite_stats(psnr_anchor)
        psnr_clean_finite_mean, psnr_clean_median, psnr_clean_num_inf = finite_stats(psnr_clean)
        ssim_anchor = arr("ssim_anchor")
        ssim_clean = arr("ssim_clean")
        summary_row = {
            "method": method,
            "alpha": alpha,
            "attack_rotation_backend": group[0].get("attack_rotation_backend", "unknown"),
            "attack_rotation_interpolation": group[0].get("attack_rotation_interpolation", "unknown"),
            "attack_rotation_fill": float(group[0].get("attack_rotation_fill", float("nan"))),
            "correction_rotation_backend": group[0].get("correction_rotation_backend", "unknown"),
            "correction_rotation_interpolation": group[0].get("correction_rotation_interpolation", "unknown"),
            "correction_rotation_fill": float(group[0].get("correction_rotation_fill", float("nan"))),
            "num_samples": len(group),
            "mean_rotation_error_deg": float(np.nanmean(rotation_error)),
            "median_rotation_error_deg": float(np.nanmedian(rotation_error)),
            "max_rotation_error_deg": float(np.nanmax(rotation_error)),
            "failure_rate_rotation_error_gt_1deg": float(np.nanmean(rotation_error > 1.0)),
            "failure_rate_rotation_error_gt_3deg": float(np.nanmean(rotation_error > 3.0)),
            "mean_psnr_anchor": float(np.nanmean(psnr_anchor)),
            "mean_psnr_anchor_finite": psnr_anchor_finite_mean,
            "median_psnr_anchor": psnr_anchor_median,
            "num_inf_psnr_anchor": psnr_anchor_num_inf,
            "mean_ssim_anchor": float(np.nanmean(ssim_anchor)),
            "median_ssim_anchor": float(np.nanmedian(ssim_anchor)),
            "mean_psnr_clean": float(np.nanmean(psnr_clean)),
            "mean_psnr_clean_finite": psnr_clean_finite_mean,
            "median_psnr_clean": psnr_clean_median,
            "num_inf_psnr_clean": psnr_clean_num_inf,
            "mean_ssim_clean": float(np.nanmean(ssim_clean)),
            "median_ssim_clean": float(np.nanmedian(ssim_clean)),
            "mean_runtime_ms": float(np.nanmean(arr("runtime_ms"))),
        }
        for field_name in ADDITIONAL_QUALITY_SUMMARY_FIELDS:
            values = np.asarray(
                [float(row.get(field_name, float("nan"))) for row in group],
                dtype=np.float64,
            )
            if np.all(np.isnan(values)):
                summary_row[f"mean_{field_name}"] = float("nan")
                summary_row[f"median_{field_name}"] = float("nan")
            else:
                summary_row[f"mean_{field_name}"] = float(np.nanmean(values))
                summary_row[f"median_{field_name}"] = float(np.nanmedian(values))
        summary.append(summary_row)
    return summary


def evaluate_diagnostic_alphas(
    image_paths: list[Path],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Evaluate diff-feature diagnostics for the requested large alpha values."""
    rows: list[dict[str, Any]] = []
    for image_path in image_paths:
        img = load_rgb_image(image_path, size=int(args.size), no_resize=bool(args.no_resize))
        H, W = img.shape[:2]
        modulation_grid, metadata = make_ring_pair_mask(
            H,
            W,
            num_angles=int(args.num_angles),
            key=int(args.key),
            method=args.method,
            num_ring_pairs=int(args.num_ring_pairs),
            angular_bin_mode=args.angular_bin_mode,
        )
        for alpha in args.diagnostic_alphas:
            x_anchor = embed_rotbind_anchor_rgb(img, modulation_grid, float(alpha))
            diff_info = diagnostic_diff_feature(
                img,
                x_anchor,
                metadata,
                int(args.num_r),
                refine_peak=not bool(args.no_refine_peak),
            )
            rows.append(
                {
                    "diagnostic_alpha": float(alpha),
                    "image_path": str(image_path),
                    **diff_info,
                }
            )
    return rows


def summarize_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate diff-feature diagnostic rows by diagnostic alpha."""
    summary: list[dict[str, Any]] = []
    for alpha in sorted({float(row["diagnostic_alpha"]) for row in rows}):
        group = [row for row in rows if float(row["diagnostic_alpha"]) == alpha]
        errors = np.asarray([float(row["diff_rotation_error_deg"]) for row in group], dtype=np.float64)
        scores = np.asarray([float(row["diff_best_score"]) for row in group], dtype=np.float64)
        margins = np.asarray([float(row["diff_corr_margin"]) for row in group], dtype=np.float64)
        summary.append(
            {
                "diagnostic_alpha": alpha,
                "num_images": len(group),
                "mean_diff_rotation_error_deg": float(np.nanmean(errors)),
                "median_diff_rotation_error_deg": float(np.nanmedian(errors)),
                "max_diff_rotation_error_deg": float(np.nanmax(errors)),
                "failure_rate_diff_rotation_error_gt_3deg": float(np.nanmean(errors > 3.0)),
                "mean_diff_best_score": float(np.nanmean(scores)),
                "mean_diff_corr_margin": float(np.nanmean(margins)),
            }
        )
    return summary


def import_matplotlib():
    """Import matplotlib with an Agg backend, returning None if unavailable."""
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def save_placeholder_plot(path: Path, title: str) -> None:
    """Save a simple placeholder image when matplotlib is unavailable."""
    canvas = np.ones((320, 480, 3), dtype=np.uint8) * 255
    Image.fromarray(canvas).save(path)


def plot_line(path: Path, rows: list[dict[str, Any]], x_name: str, y_name: str, title: str) -> None:
    """Save a simple line/scatter plot."""
    plt = import_matplotlib()
    if plt is None:
        save_placeholder_plot(path, title)
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    alphas = sorted({float(row["alpha"]) for row in rows})
    for alpha in alphas:
        group = [row for row in rows if float(row["alpha"]) == alpha]
        xs = [float(row[x_name]) for row in group]
        ys = [float(row[y_name]) for row in group]
        ax.scatter(xs, ys, s=18, label=f"alpha={alpha:g}")
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_box_by_alpha(path: Path, rows: list[dict[str, Any]]) -> None:
    """Save box plot of rotation error grouped by alpha."""
    plt = import_matplotlib()
    if plt is None:
        save_placeholder_plot(path, "rotation_error_box_by_alpha")
        return
    alphas = sorted({float(row["alpha"]) for row in rows})
    data = [[float(row["rotation_error_deg"]) for row in rows if float(row["alpha"]) == a] for a in alphas]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, labels=[f"{a:g}" for a in alphas])
    ax.set_title("rotation_error_box_by_alpha")
    ax.set_xlabel("alpha")
    ax.set_ylabel("rotation_error_deg")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary_metric(path: Path, summary: list[dict[str, Any]], names: list[str], title: str) -> None:
    """Save line plot for one or more summary metrics."""
    plt = import_matplotlib()
    if plt is None:
        save_placeholder_plot(path, title)
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [float(row["alpha"]) for row in summary]
    for name in names:
        ax.plot(xs, [float(row[name]) for row in summary], marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel("alpha")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_example_grid(path: Path, artifacts: dict[str, Any]) -> None:
    """Save the eight-panel example visualization."""
    plt = import_matplotlib()
    if plt is None:
        save_placeholder_plot(path, "example_grid")
        return

    panels = [
        ("original", artifacts["x"]),
        ("anchored", artifacts["x_anchor"]),
        ("rotated attack", artifacts["x_att"]),
        ("corrected", artifacts["x_corr"]),
        ("anchor removed", artifacts["x_clean"]),
        ("negative-anchor", artifacts["x_minus"]),
        ("anchor diff x8", np.clip(0.5 + 8.0 * (artifacts["x_anchor"] - artifacts["x"]), 0.0, 1.0)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for ax, (title, img) in zip(axes.flat[:7], panels):
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    ax = axes.flat[7]
    curve = np.asarray(artifacts["score_curve"], dtype=np.float32)
    ax.plot(curve)
    ax.set_title(
        "corr curve\n"
        f"alpha={artifacts['alpha']}, "
        f"rotation_gt={artifacts['rotation_gt_deg']}, "
        f"rotation_hat={artifacts['rotation_hat_deg']:.2f}",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_plots(outdir: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]], example: dict[str, Any]) -> None:
    """Save required PNG outputs."""
    save_example_grid(outdir / "example_grid.png", example)
    plot_line(
        outdir / "rotation_error_vs_rotation.png",
        rows,
        "rotation_gt_deg",
        "rotation_error_deg",
        "rotation_error_vs_rotation",
    )
    plot_line(outdir / "angle_error_vs_theta.png", rows, "rotation_gt_deg", "rotation_error_deg", "rotation_error_vs_rotation")
    plot_box_by_alpha(outdir / "rotation_error_box_by_alpha.png", rows)
    plot_box_by_alpha(outdir / "angle_error_box_by_alpha.png", rows)
    plot_summary_metric(
        outdir / "failure_rate_vs_alpha.png",
        summary,
        ["failure_rate_rotation_error_gt_1deg", "failure_rate_rotation_error_gt_3deg"],
        "failure_rate_vs_alpha",
    )
    plot_summary_metric(
        outdir / "quality_vs_alpha.png",
        summary,
        ["mean_psnr_anchor", "mean_psnr_clean", "mean_ssim_anchor", "mean_ssim_clean"],
        "quality_vs_alpha",
    )
    plot_summary_metric(outdir / "runtime_vs_alpha.png", summary, ["mean_runtime_ms"], "runtime_vs_alpha")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image")
    parser.add_argument("--image-dir")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--no-resize", action="store_true")
    parser.add_argument("--alphas", type=parse_float_list, default=parse_float_list("0.005,0.01,0.02,0.03,0.05,0.1,0.2"))
    parser.add_argument("--diagnostic-alphas", type=parse_float_list, default=parse_float_list("0.05,0.1,0.2"))
    parser.add_argument(
        "--angles",
        type=parse_float_list,
        default=parse_float_list("5,10,15,30,45,60,75,90,120,150,180"),
    )
    parser.add_argument("--key", type=int, default=0)
    parser.add_argument("--num-r", type=int, default=64)
    parser.add_argument("--num-angles", type=int, default=360)
    parser.add_argument("--no-refine-peak", action="store_true")
    parser.add_argument("--angular-bin-mode", choices=["nearest", "floor"], default="nearest")
    parser.add_argument("--attack-rotation-backend", choices=["torchvision", "scipy"], default="torchvision")
    parser.add_argument("--attack-rotation-interpolation", choices=["nearest", "bilinear"], default="nearest")
    parser.add_argument("--attack-rotation-fill", type=float, default=0.0)
    parser.add_argument("--correction-rotation-backend", choices=["torchvision", "scipy"], default="torchvision")
    parser.add_argument("--correction-rotation-interpolation", choices=["nearest", "bilinear"], default="bilinear")
    parser.add_argument("--correction-rotation-fill", type=float, default=0.0)
    parser.add_argument("--example-alpha", type=float, default=0.01)
    parser.add_argument("--example-theta", type=float, default=45.0)
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--angle-sign",
        choices=["auto", "raw", "neg"],
        default="raw",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--method", choices=["two_pair", "multi_ringpair"], default="two_pair")
    parser.add_argument("--num-ring-pairs", type=int, default=12)
    parser.add_argument("--vae-footprint", action="store_true")
    parser.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--vae-subfolder", default=None)
    parser.add_argument("--vae-device", default="auto")
    parser.add_argument("--vae-local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.image and not args.image_dir:
        parser.error("one of --image or --image-dir is required")
    if args.size <= 0:
        parser.error("--size must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the basic RotBind evaluation."""
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.angle_sign != "raw":
        print(f"[Deprecated --angle-sign]\nignored value = {args.angle_sign}; using raw rotation convention")

    vae_encoder = None
    if args.vae_footprint:
        vae_encoder = DiffusersVaeEncoder.from_pretrained(
            args.vae_model,
            device=args.vae_device,
            local_files_only=bool(args.vae_local_files_only),
            subfolder=args.vae_subfolder,
        )

    image_paths = discover_images(args)
    rows: list[dict[str, Any]] = []
    example_artifacts: dict[str, Any] | None = None

    for image_index, image_path in enumerate(image_paths):
        img = load_rgb_image(image_path, size=int(args.size), no_resize=bool(args.no_resize))
        H, W = img.shape[:2]
        modulation_grid, metadata = make_ring_pair_mask(
            H,
            W,
            num_angles=int(args.num_angles),
            key=int(args.key),
            method=args.method,
            num_ring_pairs=int(args.num_ring_pairs),
            angular_bin_mode=args.angular_bin_mode,
        )
        image_id = image_path.stem if image_path.stem else f"image_{image_index:04d}"
        for alpha in args.alphas:
            for theta_gt in args.angles:
                row, artifacts = evaluate_one(
                    img,
                    image_id,
                    image_path,
                    float(alpha),
                    float(theta_gt),
                    args,
                    modulation_grid,
                    metadata,
                    vae_encoder=vae_encoder,
                )
                rows.append(row)
                if (
                    example_artifacts is None
                    or (
                        abs(float(alpha) - float(args.example_alpha)) < 1e-12
                        and abs(float(theta_gt) - float(args.example_theta)) < 1e-12
                    )
                ):
                    example_artifacts = artifacts

    if not rows:
        raise RuntimeError("no evaluation rows produced")
    summary = summarize(rows, method=args.method)
    diagnostic_rows = evaluate_diagnostic_alphas(image_paths, args)
    diagnostic_summary = summarize_diagnostics(diagnostic_rows)
    write_csv(outdir / "rotbind_results.csv", RESULT_FIELDS, rows)
    write_csv(outdir / "summary.csv", SUMMARY_FIELDS, summary)
    write_csv(outdir / "diagnostic_summary.csv", DIAGNOSTIC_SUMMARY_FIELDS, diagnostic_summary)
    save_plots(outdir, rows, summary, example_artifacts or {})

    overall_mean = float(np.nanmean([float(row["rotation_error_deg"]) for row in rows]))
    overall_fail3 = float(np.nanmean([float(row["rotation_error_deg"]) > 3.0 for row in rows]))
    mean_psnr_anchor = float(np.nanmean([float(row["psnr_anchor"]) for row in rows]))
    mean_ssim_anchor = float(np.nanmean([float(row["ssim_anchor"]) for row in rows]))
    mean_psnr_clean = float(np.nanmean([float(row["psnr_clean"]) for row in rows]))
    mean_ssim_clean = float(np.nanmean([float(row["ssim_clean"]) for row in rows]))
    print("[Evaluation summary]")
    print(f"num_samples = {len(rows)}")
    print(f"mean rotation error = {overall_mean:.6f}")
    print(f"failure rate rotation error > 3deg = {overall_fail3:.6f}")
    print(f"mean PSNR anchor / clean = {mean_psnr_anchor:.6f} / {mean_psnr_clean:.6f}")
    print(f"mean SSIM anchor / clean = {mean_ssim_anchor:.6f} / {mean_ssim_clean:.6f}")
    print(f"results = {outdir / 'rotbind_results.csv'}")
    print(f"summary = {outdir / 'summary.csv'}")
    print(f"diagnostic_summary = {outdir / 'diagnostic_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
