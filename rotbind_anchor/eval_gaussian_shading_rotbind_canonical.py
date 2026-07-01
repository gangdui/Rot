"""Canonical RotBind anchor-removal necessity experiment for Gaussian Shading.

This script does not run a rotation attack. It compares Gaussian Shading
watermarked images before RotBind anchoring, after anchoring, and after explicit
RotBind anchor removal.
"""

from __future__ import annotations

import argparse
import csv
import json
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

from rotbind_anchor.eval_rotbind_anchor import (  # noqa: E402
    DiffusersVaeEncoder,
    cosine_similarity,
    load_rgb_image,
    psnr,
    simple_ssim,
    write_csv,
)
from rotbind_anchor.gaussian_shading_adapter import (  # noqa: E402
    detect_gaussian_shading,
    invert_to_zT,
    load_gaussian_shading_pipeline,
)
from rotbind_anchor.rotbind_anchor import (  # noqa: E402
    embed_rotbind_anchor_rgb,
    make_ring_pair_mask,
    remove_rotbind_anchor_rgb,
    rgb_to_ycbcr,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

RESULT_FIELDS = [
    "image_id",
    "image_path",
    "alpha",
    "method",
    "key",
    "height",
    "width",
    "gs_state_path",
    "gs_seed",
    "gs_prompt",
    "pixel_mse_anchor",
    "pixel_mse_clean",
    "psnr_anchor",
    "ssim_anchor",
    "psnr_clean",
    "ssim_clean",
    "vae_mse_anchor",
    "vae_mse_clean",
    "vae_raw_mse_anchor",
    "vae_raw_mse_clean",
    "vae_rel_mse_anchor",
    "vae_rel_mse_clean",
    "vae_cos_anchor",
    "vae_cos_clean",
    "zt_mse_anchor",
    "zt_mse_clean",
    "zt_rel_mse_anchor",
    "zt_rel_mse_clean",
    "zt_cos_anchor",
    "zt_cos_clean",
    "amp_vae_anchor",
    "amp_vae_clean",
    "amp_zt_anchor",
    "amp_zt_clean",
    "amp_total_anchor",
    "amp_total_clean",
    "detector_score_base",
    "detector_score_anchor",
    "detector_score_clean",
    "detector_score_delta_anchor",
    "detector_score_delta_clean",
    "detector_success_base",
    "detector_success_anchor",
    "detector_success_clean",
    "detector_threshold",
    "score_higher_is_better",
    "bit_accuracy_base",
    "bit_accuracy_anchor",
    "bit_accuracy_clean",
    "identification_accuracy_base",
    "identification_accuracy_anchor",
    "identification_accuracy_clean",
    "runtime_ms",
]

IDENTIFIER_FIELDS = {"image_id", "image_path", "method", "score_higher_is_better", "gs_state_path", "gs_prompt"}
SUCCESS_FIELDS = ["detector_success_base", "detector_success_anchor", "detector_success_clean"]
NUMERIC_SUMMARY_FIELDS = [
    field
    for field in RESULT_FIELDS
    if field not in IDENTIFIER_FIELDS and field not in SUCCESS_FIELDS
]
SUMMARY_FIELDS = [
    "num_images",
    "success_rate_base",
    "success_rate_anchor",
    "success_rate_clean",
    *[
        stat
        for field in NUMERIC_SUMMARY_FIELDS
        for stat in (f"mean_{field}", f"median_{field}")
    ],
]


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Return mean squared error between two arrays."""
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    return float(np.mean((av - bv) ** 2))


def rel_mse(variant: np.ndarray, base: np.ndarray, eps: float = 1e-12) -> float:
    """Return MSE normalized by base energy."""
    return mse(variant, base) / (float(np.mean(np.asarray(base, dtype=np.float32) ** 2)) + eps)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between flattened arrays."""
    return cosine_similarity(np.asarray(a).reshape(-1), np.asarray(b).reshape(-1))


def amplification(numerator: float, denominator: float, eps: float = 1e-12) -> float:
    """Return a finite amplification ratio."""
    return float(numerator) / (float(denominator) + eps)


def compute_pair_metrics(prefix: str, base: np.ndarray, variant: np.ndarray) -> dict[str, float]:
    """Return MSE, relative MSE, and cosine for a base/variant pair."""
    return {
        f"{prefix}_mse": mse(variant, base),
        f"{prefix}_rel_mse": rel_mse(variant, base),
        f"{prefix}_cos": cosine(variant, base),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image")
    parser.add_argument("--image-dir")
    parser.add_argument("--outdir", default="experiments/current/gaussian_shading_canonical_rotbind_alpha015")
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--method", choices=["two_pair", "multi_ringpair"], default="two_pair")
    parser.add_argument("--key", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--no-resize", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--num-angles", type=int, default=180)
    parser.add_argument("--num-r", type=int, default=64)
    parser.add_argument("--angular-bin-mode", choices=["nearest", "floor"], default="nearest")
    parser.add_argument("--num-ring-pairs", type=int, default=12)
    parser.add_argument("--vae-model", default=None)
    parser.add_argument("--vae-subfolder", default=None)
    parser.add_argument("--vae-local-files-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gs-adapter-module", default=None)
    parser.add_argument("--gs-config", default=None)
    parser.add_argument("--gs-key", default=None)
    parser.add_argument("--gs-model-path", default=None)
    parser.add_argument("--gs-watermark-state", default=None)
    parser.add_argument("--gs-metadata", default=None)
    parser.add_argument("--gs-code-root", default="Gaussian-Shading-master")
    parser.add_argument("--gs-num-inversion-steps", type=int, default=None)
    parser.add_argument("--gs-prompt", default="")
    parser.add_argument("--gs-threshold", type=float, default=None)
    parser.add_argument("--use-fake-gs-pipeline", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.image and not args.image_dir:
        parser.error("one of --image or --image-dir is required")
    if args.size <= 0:
        parser.error("--size must be positive")
    return args


def discover_images(args: argparse.Namespace) -> list[Path]:
    """Return input images from --image and/or --image-dir."""
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
    return paths


def load_gs_metadata(metadata_path: str | Path) -> list[dict[str, Any]]:
    """Load Gaussian Shading JSONL metadata rows."""
    path = Path(metadata_path)
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"metadata row {line_no} is not an object: {path}")
            rows.append(row)
    return rows


def build_state_lookup(metadata_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build lookup keys for per-image watermark state metadata."""
    lookup: dict[str, dict[str, Any]] = {}
    for row in metadata_rows:
        image_path = row.get("image_path")
        if image_path:
            path = Path(str(image_path))
            lookup[str(path.resolve())] = row
            lookup[path.name] = row
        image_id = row.get("image_id")
        if image_id:
            lookup[str(image_id)] = row
    return lookup


def resolve_state_for_image(image_path: Path, state_lookup: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Return the metadata row matching an image path/name/id."""
    keys = [str(image_path.resolve()), image_path.name, image_path.stem]
    for key in keys:
        if key in state_lookup:
            return state_lookup[key]
    return None


def state_path_from_metadata_row(row: dict[str, Any], image_path: Path) -> Path:
    """Return an existing watermark state path for a metadata row."""
    raw_state = row.get("state_path")
    if not raw_state:
        raise ValueError(f"No state_path found in metadata for image: {image_path}")
    state_path = Path(str(raw_state))
    if not state_path.exists():
        raise FileNotFoundError(f"Gaussian Shading watermark state not found for image {image_path}: {state_path}")
    return state_path


def get_vae_encoder(args: argparse.Namespace, gs_pipeline: Any) -> Any:
    """Return the VAE encoder used for scaled latent footprint metrics."""
    if hasattr(gs_pipeline, "vae_encoder"):
        return gs_pipeline.vae_encoder
    if args.vae_model:
        return DiffusersVaeEncoder.from_pretrained(
            args.vae_model,
            device=args.device,
            local_files_only=bool(args.vae_local_files_only),
            subfolder=args.vae_subfolder,
        )
    raise ValueError(
        "VAE metrics require a Gaussian Shading pipeline with vae_encoder or "
        "an explicit --vae-model."
    )


def compute_vae_metrics(vae_encoder: Any, x_base: np.ndarray, x_anchor: np.ndarray, x_clean: np.ndarray) -> dict[str, float]:
    """Compute scaled and raw VAE latent footprint metrics."""
    z_base, z_anchor, z_clean = vae_encoder.encode_images([x_base, x_anchor, x_clean])
    if hasattr(vae_encoder, "encode_images_raw"):
        raw_base, raw_anchor, raw_clean = vae_encoder.encode_images_raw([x_base, x_anchor, x_clean])
    else:
        raw_base, raw_anchor, raw_clean = z_base, z_anchor, z_clean
    return {
        "vae_mse_anchor": mse(z_anchor, z_base),
        "vae_mse_clean": mse(z_clean, z_base),
        "vae_raw_mse_anchor": mse(raw_anchor, raw_base),
        "vae_raw_mse_clean": mse(raw_clean, raw_base),
        "vae_rel_mse_anchor": rel_mse(z_anchor, z_base),
        "vae_rel_mse_clean": rel_mse(z_clean, z_base),
        "vae_cos_anchor": cosine(z_anchor, z_base),
        "vae_cos_clean": cosine(z_clean, z_base),
    }


def evaluate_one(
    img: np.ndarray,
    image_id: str,
    image_path: Path,
    args: argparse.Namespace,
    gs_pipeline: Any,
    vae_encoder: Any,
    metadata_row: dict[str, Any] | None = None,
    state_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one Gaussian Shading watermarked image."""
    start = time.perf_counter()
    H, W = img.shape[:2]
    modulation_grid, _ = make_ring_pair_mask(
        H,
        W,
        num_angles=int(args.num_angles),
        key=int(args.key),
        method=args.method,
        num_ring_pairs=int(args.num_ring_pairs),
        angular_bin_mode=args.angular_bin_mode,
    )
    x_base = np.clip(img, 0.0, 1.0).astype(np.float32)
    x_anchor = embed_rotbind_anchor_rgb(x_base, modulation_grid, float(args.alpha))
    x_clean = remove_rotbind_anchor_rgb(x_anchor, modulation_grid, float(args.alpha))

    pixel_mse_anchor = mse(x_anchor, x_base)
    pixel_mse_clean = mse(x_clean, x_base)
    vae_metrics = compute_vae_metrics(vae_encoder, x_base, x_anchor, x_clean)

    zt_base = invert_to_zT(gs_pipeline, x_base)
    zt_anchor = invert_to_zT(gs_pipeline, x_anchor)
    zt_clean = invert_to_zT(gs_pipeline, x_clean)
    zt_metrics = {
        "zt_mse_anchor": mse(zt_anchor, zt_base),
        "zt_mse_clean": mse(zt_clean, zt_base),
        "zt_rel_mse_anchor": rel_mse(zt_anchor, zt_base),
        "zt_rel_mse_clean": rel_mse(zt_clean, zt_base),
        "zt_cos_anchor": cosine(zt_anchor, zt_base),
        "zt_cos_clean": cosine(zt_clean, zt_base),
    }

    det_base = detect_gaussian_shading(gs_pipeline, zT=zt_base, image=x_base)
    det_anchor = detect_gaussian_shading(gs_pipeline, zT=zt_anchor, image=x_anchor)
    det_clean = detect_gaussian_shading(gs_pipeline, zT=zt_clean, image=x_clean)

    row: dict[str, Any] = {
        "image_id": image_id,
        "image_path": str(image_path),
        "alpha": float(args.alpha),
        "method": args.method,
        "key": int(args.key),
        "height": int(H),
        "width": int(W),
        "gs_state_path": str(state_path) if state_path is not None else str(args.gs_watermark_state or ""),
        "gs_seed": float(metadata_row["seed"]) if metadata_row is not None and metadata_row.get("seed") is not None else float("nan"),
        "gs_prompt": str(metadata_row.get("prompt", "")) if metadata_row is not None else str(args.gs_prompt or ""),
        "pixel_mse_anchor": pixel_mse_anchor,
        "pixel_mse_clean": pixel_mse_clean,
        "psnr_anchor": psnr(x_base, x_anchor),
        "ssim_anchor": simple_ssim(x_base, x_anchor),
        "psnr_clean": psnr(x_base, x_clean),
        "ssim_clean": simple_ssim(x_base, x_clean),
        **vae_metrics,
        **zt_metrics,
        "amp_vae_anchor": amplification(vae_metrics["vae_mse_anchor"], pixel_mse_anchor),
        "amp_vae_clean": amplification(vae_metrics["vae_mse_clean"], pixel_mse_clean),
        "amp_zt_anchor": amplification(zt_metrics["zt_mse_anchor"], vae_metrics["vae_mse_anchor"]),
        "amp_zt_clean": amplification(zt_metrics["zt_mse_clean"], vae_metrics["vae_mse_clean"]),
        "amp_total_anchor": amplification(zt_metrics["zt_mse_anchor"], pixel_mse_anchor),
        "amp_total_clean": amplification(zt_metrics["zt_mse_clean"], pixel_mse_clean),
        "detector_score_base": det_base["detector_score"],
        "detector_score_anchor": det_anchor["detector_score"],
        "detector_score_clean": det_clean["detector_score"],
        "detector_score_delta_anchor": det_anchor["detector_score"] - det_base["detector_score"],
        "detector_score_delta_clean": det_clean["detector_score"] - det_base["detector_score"],
        "detector_success_base": bool(det_base["detector_success"]),
        "detector_success_anchor": bool(det_anchor["detector_success"]),
        "detector_success_clean": bool(det_clean["detector_success"]),
        "detector_threshold": det_base["detector_threshold"],
        "score_higher_is_better": bool(det_base["score_higher_is_better"]),
        "bit_accuracy_base": det_base["bit_accuracy"],
        "bit_accuracy_anchor": det_anchor["bit_accuracy"],
        "bit_accuracy_clean": det_clean["bit_accuracy"],
        "identification_accuracy_base": det_base["identification_accuracy"],
        "identification_accuracy_anchor": det_anchor["identification_accuracy"],
        "identification_accuracy_clean": det_clean["identification_accuracy"],
        "runtime_ms": (time.perf_counter() - start) * 1000.0,
    }
    if not bool(det_base["detector_success"]):
        print(
            "[Warning] Gaussian Shading baseline detector failed for "
            f"{image_path}: score={det_base['detector_score']}, threshold={det_base['detector_threshold']}"
        )
    artifacts = {
        "x_base": x_base,
        "x_anchor": x_anchor,
        "x_clean": x_clean,
        "zt_mse_anchor": zt_metrics["zt_mse_anchor"],
        "zt_mse_clean": zt_metrics["zt_mse_clean"],
        "zt_shape": tuple(np.asarray(zt_base).shape),
        "zt_dtype": str(np.asarray(zt_base).dtype),
        "detector_scores": [
            det_base["detector_score"],
            det_anchor["detector_score"],
            det_clean["detector_score"],
        ],
    }
    return row, artifacts


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate result rows into one summary row."""
    if not rows:
        return []

    def values(field: str) -> np.ndarray:
        return np.asarray([float(row.get(field, float("nan"))) for row in rows], dtype=np.float64)

    summary: dict[str, Any] = {
        "num_images": len(rows),
        "success_rate_base": float(np.mean(values("detector_success_base"))),
        "success_rate_anchor": float(np.mean(values("detector_success_anchor"))),
        "success_rate_clean": float(np.mean(values("detector_success_clean"))),
    }
    for field in NUMERIC_SUMMARY_FIELDS:
        vals = values(field)
        if np.all(np.isnan(vals)):
            summary[f"mean_{field}"] = float("nan")
            summary[f"median_{field}"] = float("nan")
        else:
            summary[f"mean_{field}"] = float(np.nanmean(vals))
            summary[f"median_{field}"] = float(np.nanmedian(vals))
    return [summary]


def save_readme(outdir: Path, args: argparse.Namespace, score_higher_is_better: bool) -> None:
    """Write experiment README."""
    text = f"""# Gaussian Shading Canonical RotBind Necessity

Gaussian Shading canonical RotBind anchor-removal necessity experiment.

- no rotation attack
- input images are Gaussian Shading watermarked images
- evaluates whether RotBind anchor removal is necessary before Gaussian Shading inversion/detection
- VAE MSE uses Stable Diffusion scaled latents
- zT metrics compare Gaussian Shading inversion outputs
- detector score direction is recorded by `score_higher_is_better`

Configuration:

```text
alpha = {float(args.alpha):g}
method = {args.method}
key = {int(args.key)}
score_higher_is_better = {bool(score_higher_is_better)}
```
"""
    (outdir / "README.md").write_text(text)


def save_example_grid(path: Path, artifacts: dict[str, Any]) -> None:
    """Save a minimal diagnostic grid."""
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        Image.fromarray(np.ones((320, 480, 3), dtype=np.uint8) * 255).save(path)
        return

    x_base = artifacts["x_base"]
    x_anchor = artifacts["x_anchor"]
    x_clean = artifacts["x_clean"]
    y_base = rgb_to_ycbcr(x_base)[..., 0]
    y_anchor = rgb_to_ycbcr(x_anchor)[..., 0]
    log_fft_diff = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(y_anchor)))) - np.log1p(
        np.abs(np.fft.fftshift(np.fft.fft2(y_base)))
    )

    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    panels = [
        ("x_base", x_base),
        ("x_anchor", x_anchor),
        ("x_clean", x_clean),
        ("abs(anchor-base) x50", np.clip(np.abs(x_anchor - x_base) * 50.0, 0.0, 1.0)),
        ("abs(clean-base) x50", np.clip(np.abs(x_clean - x_base) * 50.0, 0.0, 1.0)),
    ]
    for ax, (title, img) in zip(axes.flat[:5], panels):
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    ax = axes.flat[5]
    im = ax.imshow(log_fft_diff, cmap="coolwarm")
    ax.set_title("log FFT diff Y", fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes.flat[6]
    ax.bar(["base", "anchor", "clean"], artifacts["detector_scores"])
    ax.set_title("detector score", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax = axes.flat[7]
    ax.bar(["anchor", "clean"], [artifacts["zt_mse_anchor"], artifacts["zt_mse_clean"]])
    ax.set_title("zT MSE", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Run the canonical Gaussian Shading/RotBind experiment."""
    args = parse_args(argv)
    np.random.seed(int(args.seed))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image_paths = discover_images(args)
    metadata_rows: list[dict[str, Any]] | None = None
    state_lookup: dict[str, dict[str, Any]] = {}
    if args.gs_metadata is not None:
        metadata_rows = load_gs_metadata(args.gs_metadata)
        state_lookup = build_state_lookup(metadata_rows)
        first_row = resolve_state_for_image(image_paths[0], state_lookup)
        if first_row is None:
            raise ValueError(f"No state_path found in metadata for image: {image_paths[0]}")
        first_state = state_path_from_metadata_row(first_row, image_paths[0])
        if not args.gs_watermark_state:
            args.gs_watermark_state = str(first_state)

    gs_pipeline = load_gaussian_shading_pipeline(args)
    vae_encoder = get_vae_encoder(args, gs_pipeline)

    rows: list[dict[str, Any]] = []
    example: dict[str, Any] | None = None
    for path in image_paths:
        metadata_row = None
        state_path = None
        if args.gs_metadata is not None:
            metadata_row = resolve_state_for_image(path, state_lookup)
            if metadata_row is None:
                raise ValueError(f"No state_path found in metadata for image: {path}")
            state_path = state_path_from_metadata_row(metadata_row, path)
            if not hasattr(gs_pipeline, "set_watermark_state"):
                raise ValueError("Pipeline does not support per-image watermark state switching")
            gs_pipeline.set_watermark_state(state_path)
        elif args.gs_watermark_state:
            state_path = Path(str(args.gs_watermark_state))
        img = load_rgb_image(path, size=int(args.size), no_resize=bool(args.no_resize))
        row, artifacts = evaluate_one(
            img,
            path.stem,
            path,
            args,
            gs_pipeline,
            vae_encoder,
            metadata_row=metadata_row,
            state_path=state_path,
        )
        rows.append(row)
        if example is None:
            example = artifacts

    summary = summarize_rows(rows)
    write_csv(outdir / "rotbind_gs_canonical_results.csv", RESULT_FIELDS, rows)
    write_csv(outdir / "summary.csv", SUMMARY_FIELDS, summary)
    if example is not None:
        save_example_grid(outdir / "example_grid.png", example)
    score_higher = bool(rows[0].get("score_higher_is_better", True)) if rows else True
    save_readme(outdir, args, score_higher)

    print("[Gaussian Shading canonical RotBind summary]")
    print(f"num_images = {len(rows)}")
    if example is not None:
        print(f"zT shape / dtype = {example.get('zt_shape')} / {example.get('zt_dtype')}")
    print(f"results = {outdir / 'rotbind_gs_canonical_results.csv'}")
    print(f"summary = {outdir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
