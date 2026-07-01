"""Paper-aligned Gaussian Shading + RotBind image quality evaluation.

This script separates two metric families:

1. RotBind incremental paired quality: compare a Gaussian Shading baseline
   watermarked image against its RotBind-anchored and anchor-removed variants.
2. Generation/distribution quality: CLIP score and optional FID-style metrics
   for the baseline, anchor, and clean image sets.

It does not run Gaussian Shading inversion/detection and does not modify the
RotBind embedding/removal algorithms.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rotbind_anchor.eval_rotbind_anchor import (  # noqa: E402
    load_rgb_image,
    psnr,
    simple_ssim,
    write_csv,
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
    "prompt",
    "seed",
    "alpha",
    "method",
    "key",
    "height",
    "width",
    "pixel_mse_anchor",
    "pixel_mse_clean",
    "psnr_anchor",
    "psnr_clean",
    "ssim_anchor",
    "ssim_clean",
    "lpips_anchor",
    "lpips_clean",
    "clip_score_baseline",
    "clip_score_anchor",
    "clip_score_clean",
    "clip_score_delta_anchor",
    "clip_score_delta_clean",
]

SUMMARY_METRIC_FIELDS = [
    "pixel_mse_anchor",
    "pixel_mse_clean",
    "psnr_anchor",
    "psnr_clean",
    "ssim_anchor",
    "ssim_clean",
    "lpips_anchor",
    "lpips_clean",
    "clip_score_baseline",
    "clip_score_anchor",
    "clip_score_clean",
    "clip_score_delta_anchor",
    "clip_score_delta_clean",
]

FID_FIELDS = [
    "fid_baseline_vs_reference",
    "fid_anchor_vs_reference",
    "fid_clean_vs_reference",
    "fid_anchor_vs_baseline",
    "fid_clean_vs_baseline",
]

SUMMARY_FIELDS = [
    "num_images",
    *[stat for field in SUMMARY_METRIC_FIELDS for stat in (f"mean_{field}", f"median_{field}")],
    "finite_mean_psnr_anchor",
    "finite_mean_psnr_clean",
    *FID_FIELDS,
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--outdir", default="experiments/current/gaussian_shading_quality_rotbind_alpha015_real10")
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--method", choices=["two_pair", "multi_ringpair"], default="two_pair")
    parser.add_argument("--key", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--no-resize", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--num-angles", type=int, default=180)
    parser.add_argument("--num-ring-pairs", type=int, default=12)
    parser.add_argument("--angular-bin-mode", choices=["nearest", "floor"], default="nearest")
    parser.add_argument("--clip-model", default="ViT-g-14")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--open-clip-code-root", default="Gaussian-Shading-master")
    parser.add_argument("--allow-clip-download", action="store_true")
    parser.add_argument("--clip-scale-100", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reference-image-dir")
    parser.add_argument("--compute-fid", action="store_true")
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument("--save-variant-images", action="store_true")
    parser.add_argument("--use-fake-clip", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.size <= 0:
        parser.error("--size must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def discover_images(image_dir: str | Path, max_images: int | None = None) -> list[Path]:
    """Return image paths from an input directory."""
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"image directory not found: {root}")
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if max_images is not None:
        paths = paths[: int(max_images)]
    if not paths:
        raise ValueError(f"no input images found in {root}")
    return paths


def load_metadata_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load Gaussian Shading JSONL metadata rows."""
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"metadata row {line_no} is not an object: {path}")
            rows.append(row)
    return rows


def build_metadata_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build lookup keys for image id, filename, and absolute image path."""
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        image_id = row.get("image_id")
        if image_id is not None:
            lookup[str(image_id)] = row
        image_path = row.get("image_path")
        if image_path:
            path = Path(str(image_path))
            lookup[path.name] = row
            lookup[path.stem] = row
            lookup[str(path.resolve())] = row
    return lookup


def metadata_for_image(path: Path, lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return the metadata row for an image path."""
    for key in (str(path.resolve()), path.name, path.stem):
        if key in lookup:
            return lookup[key]
    raise ValueError(f"No metadata row found for image: {path}")


def pixel_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Return image-space MSE."""
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    return float(np.mean((av - bv) ** 2))


def save_rgb(path: Path, img: np.ndarray) -> None:
    """Save a float RGB image in [0, 1]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(img, 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


class FakeClipScorer:
    """Small deterministic scorer used only for tests."""

    def __init__(self, scale_100: bool = False) -> None:
        self.scale = 100.0 if scale_100 else 1.0

    def score(self, images: list[np.ndarray], prompts: list[str]) -> list[float]:
        """Return deterministic prompt-image pseudo cosine scores."""
        scores: list[float] = []
        for img, prompt in zip(images, prompts):
            digest = hashlib.sha1(prompt.encode("utf-8")).digest()
            prompt_val = int.from_bytes(digest[:4], "big") / float(2**32 - 1)
            img_val = float(np.mean(np.asarray(img, dtype=np.float32)))
            score = 0.5 * prompt_val + 0.5 * img_val
            scores.append(float(score * self.scale))
        return scores


class OpenClipScorer:
    """open_clip image-text similarity scorer."""

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: str = "auto",
        scale_100: bool = False,
        allow_download: bool = False,
        open_clip_code_root: str | Path = "Gaussian-Shading-master",
    ) -> None:
        try:
            import torch
            open_clip = import_open_clip(open_clip_code_root)
        except Exception as exc:
            raise RuntimeError(
                "CLIP Score requires open_clip. Install open_clip_torch, keep "
                "Gaussian-Shading-master/open_clip available, or use --use-fake-clip for tests. "
                f"Original error: {exc}"
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.open_clip = open_clip
        self.device = device
        self.scale = 100.0 if scale_100 else 1.0
        resolved_pretrained = resolve_clip_pretrained(
            open_clip,
            model_name,
            pretrained,
            allow_download=allow_download,
        )
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=resolved_pretrained,
                device=device,
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to load open_clip model/pretrained weights. To avoid a stalled network "
                "download, specify a locally available --clip-model/--clip-pretrained or install "
                f"the requested weights first. Original error: {exc}"
            ) from exc
        self.model = model.eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def score(self, images: list[np.ndarray], prompts: list[str]) -> list[float]:
        """Return normalized image-text dot products."""
        torch = self.torch
        pil_images = [
            Image.fromarray((np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
            for img in images
        ]
        image_tensor = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        text_tensor = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            scores = (image_features * text_features).sum(dim=-1) * self.scale
        return [float(v) for v in scores.detach().float().cpu().numpy()]


def build_clip_scorer(args: argparse.Namespace) -> Any:
    """Return a CLIP scorer."""
    if bool(args.use_fake_clip):
        return FakeClipScorer(scale_100=bool(args.clip_scale_100))
    import_open_clip(args.open_clip_code_root)
    return OpenClipScorer(
        args.clip_model,
        args.clip_pretrained,
        device=args.device,
        scale_100=bool(args.clip_scale_100),
        allow_download=bool(args.allow_clip_download),
        open_clip_code_root=args.open_clip_code_root,
    )


def import_open_clip(code_root: str | Path = "Gaussian-Shading-master") -> Any:
    """Import open_clip, falling back to Gaussian Shading's vendored package."""
    try:
        import open_clip

        return open_clip
    except ModuleNotFoundError as first_error:
        root = Path(code_root)
        if not root.exists():
            raise first_error
        root_str = str(root.resolve())
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        try:
            import open_clip

            return open_clip
        except ModuleNotFoundError as fallback_error:
            raise fallback_error from first_error


def resolve_clip_pretrained(
    open_clip: Any,
    model_name: str,
    pretrained: str,
    allow_download: bool = False,
) -> str:
    """Resolve an open_clip pretrained tag to a local checkpoint path.

    By default this function refuses remote downloads. This keeps quality
    evaluation from hanging on slow or blocked network access.
    """
    candidate = Path(str(pretrained)).expanduser()
    if candidate.exists():
        return str(candidate)

    cfg = open_clip.pretrained.get_pretrained_cfg(model_name, pretrained)
    if not cfg:
        return pretrained

    cached = find_cached_open_clip_checkpoint(cfg)
    if cached is not None:
        return str(cached)
    if allow_download:
        return download_open_clip_checkpoint(open_clip, cfg)

    raise RuntimeError(
        "CLIP pretrained weights are not available in the local cache: "
        f"model={model_name}, pretrained={pretrained}. "
        "Install/cache the weights first, pass --clip-pretrained as a local "
        "checkpoint path, choose a locally available open_clip configuration, "
        "or explicitly add --allow-clip-download."
    )


def find_cached_open_clip_checkpoint(cfg: dict[str, Any]) -> Path | None:
    """Return a cached open_clip checkpoint path when one is already present."""
    url = str(cfg.get("url", "") or "")
    if url:
        cache_path = Path("~/.cache/clip").expanduser() / Path(urllib.parse.urlparse(url).path).name
        if cache_path.exists():
            return cache_path
    hf_hub = str(cfg.get("hf_hub", "") or "")
    if hf_hub:
        try:
            from huggingface_hub import hf_hub_download

            model_id, filename = os.path.split(hf_hub)
            if not filename:
                filename = "open_clip_pytorch_model.bin"
            return Path(hf_hub_download(model_id, filename=filename, local_files_only=True))
        except Exception:
            return None
    return None


def download_open_clip_checkpoint(open_clip: Any, cfg: dict[str, Any]) -> str:
    """Download an open_clip checkpoint after explicit user opt-in."""
    try:
        url = str(cfg.get("url", "") or "")
        if url:
            cache_dir = Path("~/.cache/clip").expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            target = cache_dir / Path(urllib.parse.urlparse(url).path).name
            urllib.request.urlretrieve(url, target)
            return str(target)
        hf_hub = str(cfg.get("hf_hub", "") or "")
        if hf_hub:
            from huggingface_hub import hf_hub_download

            model_id, filename = os.path.split(hf_hub)
            if not filename:
                filename = "open_clip_pytorch_model.bin"
            return str(hf_hub_download(model_id, filename=filename))
        return open_clip.pretrained.download_pretrained(cfg)
    except Exception as exc:
        raise RuntimeError(
            "CLIP weight download failed. This is usually a network, SSL, Hugging Face "
            "mirror, or Xet/CAS download issue rather than a RotBind error. "
            "Recommended fixes: pre-download the open_clip checkpoint and pass "
            "`--clip-pretrained /path/to/open_clip_pytorch_model.bin`; or retry with "
            "`HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`; or use a locally "
            "cached open_clip model/pretrained pair. Original error: "
            f"{exc}"
        ) from exc


def compute_lpips_pairs(
    x_base: np.ndarray,
    x_anchor: np.ndarray,
    x_clean: np.ndarray,
    enabled: bool,
) -> tuple[float, float]:
    """Compute LPIPS for paired variants, or nan when disabled/unavailable."""
    if not enabled:
        return float("nan"), float("nan")
    try:
        import torch
        import lpips
    except Exception as exc:
        print(f"[Warning] LPIPS requested but dependency is unavailable; writing nan. Original error: {exc}")
        return float("nan"), float("nan")

    loss_fn = lpips.LPIPS(net="alex")

    def to_tensor(img: np.ndarray) -> Any:
        arr = np.clip(img, 0.0, 1.0).astype(np.float32) * 2.0 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        lp_anchor = loss_fn(to_tensor(x_base), to_tensor(x_anchor))
        lp_clean = loss_fn(to_tensor(x_base), to_tensor(x_clean))
    return float(lp_anchor.item()), float(lp_clean.item())


def evaluate_one(
    path: Path,
    args: argparse.Namespace,
    metadata_row: dict[str, Any],
    clip_scorer: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one baseline Gaussian Shading image and its RotBind variants."""
    x_base = load_rgb_image(path, size=int(args.size), no_resize=bool(args.no_resize))
    H, W = x_base.shape[:2]
    modulation_grid, _ = make_ring_pair_mask(
        H,
        W,
        num_angles=int(args.num_angles),
        key=int(args.key),
        method=args.method,
        num_ring_pairs=int(args.num_ring_pairs),
        angular_bin_mode=args.angular_bin_mode,
    )
    x_anchor = embed_rotbind_anchor_rgb(x_base, modulation_grid, float(args.alpha))
    x_clean = remove_rotbind_anchor_rgb(x_anchor, modulation_grid, float(args.alpha))
    prompt = str(metadata_row.get("prompt", ""))
    image_id = str(metadata_row.get("image_id") or path.stem)
    seed = metadata_row.get("seed", "")
    clip_baseline, clip_anchor, clip_clean = clip_scorer.score(
        [x_base, x_anchor, x_clean],
        [prompt, prompt, prompt],
    )
    lpips_anchor, lpips_clean = compute_lpips_pairs(
        x_base,
        x_anchor,
        x_clean,
        enabled=bool(args.compute_lpips),
    )
    row: dict[str, Any] = {
        "image_id": image_id,
        "image_path": str(path),
        "prompt": prompt,
        "seed": seed,
        "alpha": float(args.alpha),
        "method": args.method,
        "key": int(args.key),
        "height": int(H),
        "width": int(W),
        "pixel_mse_anchor": pixel_mse(x_base, x_anchor),
        "pixel_mse_clean": pixel_mse(x_base, x_clean),
        "psnr_anchor": psnr(x_base, x_anchor),
        "psnr_clean": psnr(x_base, x_clean),
        "ssim_anchor": simple_ssim(x_base, x_anchor),
        "ssim_clean": simple_ssim(x_base, x_clean),
        "lpips_anchor": lpips_anchor,
        "lpips_clean": lpips_clean,
        "clip_score_baseline": clip_baseline,
        "clip_score_anchor": clip_anchor,
        "clip_score_clean": clip_clean,
        "clip_score_delta_anchor": clip_anchor - clip_baseline,
        "clip_score_delta_clean": clip_clean - clip_baseline,
    }
    artifacts = {
        "x_base": x_base,
        "x_anchor": x_anchor,
        "x_clean": x_clean,
        "row": row,
    }
    return row, artifacts


def finite_mean(values: np.ndarray) -> float:
    """Return mean over finite values only."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmean(finite))


def summarize_rows(rows: list[dict[str, Any]], fid_metrics: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Aggregate per-image quality rows into a one-row summary."""
    if not rows:
        return []
    summary: dict[str, Any] = {"num_images": len(rows)}
    for field in SUMMARY_METRIC_FIELDS:
        vals = np.asarray([float(row.get(field, float("nan"))) for row in rows], dtype=np.float64)
        if np.all(np.isnan(vals)):
            summary[f"mean_{field}"] = float("nan")
            summary[f"median_{field}"] = float("nan")
        else:
            summary[f"mean_{field}"] = float(np.nanmean(vals))
            summary[f"median_{field}"] = float(np.nanmedian(vals))
    summary["finite_mean_psnr_anchor"] = finite_mean(
        np.asarray([float(row.get("psnr_anchor", float("nan"))) for row in rows], dtype=np.float64)
    )
    summary["finite_mean_psnr_clean"] = finite_mean(
        np.asarray([float(row.get("psnr_clean", float("nan"))) for row in rows], dtype=np.float64)
    )
    fid_metrics = fid_metrics or {}
    for field in FID_FIELDS:
        summary[field] = float(fid_metrics.get(field, float("nan")))
    return [summary]


def compute_fid_metrics(args: argparse.Namespace, outdir: Path) -> dict[str, float]:
    """Compute optional clean-fid metrics for saved variant image sets."""
    metrics = {field: float("nan") for field in FID_FIELDS}
    if not bool(args.compute_fid):
        return metrics
    try:
        from cleanfid import fid
    except Exception as exc:
        print(f"[Warning] FID requested but clean-fid is unavailable; writing nan. Original error: {exc}")
        return metrics

    baseline_dir = outdir / "images" / "baseline"
    anchor_dir = outdir / "images" / "anchor"
    clean_dir = outdir / "images" / "clean"
    if args.reference_image_dir:
        reference_dir = Path(args.reference_image_dir)
        metrics["fid_baseline_vs_reference"] = float(fid.compute_fid(str(baseline_dir), str(reference_dir), mode="clean"))
        metrics["fid_anchor_vs_reference"] = float(fid.compute_fid(str(anchor_dir), str(reference_dir), mode="clean"))
        metrics["fid_clean_vs_reference"] = float(fid.compute_fid(str(clean_dir), str(reference_dir), mode="clean"))
    else:
        print("[Warning] --compute-fid was set without --reference-image-dir; reference FID fields remain nan.")
    metrics["fid_anchor_vs_baseline"] = float(fid.compute_fid(str(anchor_dir), str(baseline_dir), mode="clean"))
    metrics["fid_clean_vs_baseline"] = float(fid.compute_fid(str(clean_dir), str(baseline_dir), mode="clean"))
    return metrics


def save_example_grid(path: Path, artifacts: dict[str, Any]) -> None:
    """Save a compact diagnostic quality grid."""
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        Image.fromarray(np.ones((512, 768, 3), dtype=np.uint8) * 255).save(path)
        return

    x_base = artifacts["x_base"]
    x_anchor = artifacts["x_anchor"]
    x_clean = artifacts["x_clean"]
    row = artifacts["row"]
    y_base = rgb_to_ycbcr(x_base)[..., 0]
    y_anchor = rgb_to_ycbcr(x_anchor)[..., 0]
    log_fft_diff = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(y_anchor)))) - np.log1p(
        np.abs(np.fft.fftshift(np.fft.fft2(y_base)))
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    panels = [
        ("baseline", x_base),
        ("anchor", x_anchor),
        ("clean", x_clean),
        ("abs(anchor-baseline) x50", np.clip(np.abs(x_anchor - x_base) * 50.0, 0.0, 1.0)),
        ("abs(clean-baseline) x50", np.clip(np.abs(x_clean - x_base) * 50.0, 0.0, 1.0)),
    ]
    for ax, (title, img) in zip(axes.flat[:5], panels):
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    ax = axes.flat[5]
    im = ax.imshow(log_fft_diff, cmap="coolwarm")
    ax.set_title("log_fft_diff_Y", fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        "PSNR anchor/clean: "
        f"{float(row['psnr_anchor']):.2f}/{float(row['psnr_clean']):.2f} | "
        "SSIM anchor/clean: "
        f"{float(row['ssim_anchor']):.4f}/{float(row['ssim_clean']):.4f} | "
        "CLIP baseline/anchor/clean: "
        f"{float(row['clip_score_baseline']):.4f}/"
        f"{float(row['clip_score_anchor']):.4f}/"
        f"{float(row['clip_score_clean']):.4f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_readme(outdir: Path, args: argparse.Namespace) -> None:
    """Write output README explaining metric semantics."""
    reference = args.reference_image_dir or "not provided"
    scale = "raw cosine-like similarity" if not bool(args.clip_scale_100) else "cosine-like similarity multiplied by 100"
    text = f"""# Gaussian Shading + RotBind Quality Evaluation

This directory reports paper-aligned image quality metrics for Gaussian Shading
baseline images and RotBind variants.

Metric semantics:

- PSNR/SSIM/LPIPS are RotBind incremental paired quality metrics. They compare
  `x_w` against `x_anchor` and `x_clean`, so they are not a direct full-pipeline
  comparison with external latent watermark methods.
- FID/CLIP are closer to latent watermark paper generation quality metrics.
- If `reference-image-dir` is not provided, `fid_*_vs_reference` is `nan`.
- `fid_anchor_vs_baseline` and `fid_clean_vs_baseline` only measure additional
  RotBind distribution drift relative to the Gaussian Shading baseline. They
  should not be used for direct external-method comparison.
- LPIPS is lower-is-better, CLIP Score is higher-is-better, and FID is
  lower-is-better.
- PSNR can be `inf`; formal reporting should prefer median PSNR or finite mean
  PSNR when exact reconstruction occurs.

Configuration:

```text
alpha = {float(args.alpha):g}
method = {args.method}
key = {int(args.key)}
clip_model = {args.clip_model}
clip_pretrained = {args.clip_pretrained}
clip_score_scale = {scale}
reference_image_dir = {reference}
compute_lpips = {bool(args.compute_lpips)}
compute_fid = {bool(args.compute_fid)}
```
"""
    (outdir / "README.md").write_text(text)


def ensure_variant_dirs(outdir: Path) -> dict[str, Path]:
    """Create and return variant image directories."""
    dirs = {
        "baseline": outdir / "images" / "baseline",
        "anchor": outdir / "images" / "anchor",
        "clean": outdir / "images" / "clean",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def main(argv: list[str] | None = None) -> int:
    """Run the quality evaluation."""
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    image_paths = discover_images(args.image_dir, max_images=args.max_images)
    metadata_lookup = build_metadata_lookup(load_metadata_jsonl(args.metadata))
    clip_scorer = build_clip_scorer(args)
    variant_dirs = ensure_variant_dirs(outdir)

    rows: list[dict[str, Any]] = []
    example: dict[str, Any] | None = None
    save_variants = bool(args.save_variant_images) or bool(args.compute_fid)
    for path in image_paths:
        metadata_row = metadata_for_image(path, metadata_lookup)
        row, artifacts = evaluate_one(path, args, metadata_row, clip_scorer)
        rows.append(row)
        if save_variants:
            image_id = str(row["image_id"])
            save_rgb(variant_dirs["baseline"] / f"{image_id}.png", artifacts["x_base"])
            save_rgb(variant_dirs["anchor"] / f"{image_id}.png", artifacts["x_anchor"])
            save_rgb(variant_dirs["clean"] / f"{image_id}.png", artifacts["x_clean"])
        if example is None:
            example = artifacts

    fid_metrics = compute_fid_metrics(args, outdir)
    summary = summarize_rows(rows, fid_metrics=fid_metrics)
    write_csv(outdir / "quality_results.csv", RESULT_FIELDS, rows)
    write_csv(outdir / "quality_summary.csv", SUMMARY_FIELDS, summary)
    if example is not None:
        save_example_grid(outdir / "example_quality_grid.png", example)
    save_readme(outdir, args)

    print("[Gaussian Shading + RotBind quality summary]")
    print(f"num_images = {len(rows)}")
    print(f"results = {outdir / 'quality_results.csv'}")
    print(f"summary = {outdir / 'quality_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
