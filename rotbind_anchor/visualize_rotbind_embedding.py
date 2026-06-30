"""Visualize RotBind anchor embedding diagnostics for papers and talks."""

from __future__ import annotations

import argparse
import os
import sys
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
    make_ring_pair_mask,
    remove_rotbind_anchor_rgb,
    rgb_to_ycbcr,
    rotate_image_keep_size,
    shift_to_attack_angle,
    wrap_angle_signed,
)


def load_rgb(path: Path, size: int) -> np.ndarray:
    """Load and resize an RGB image to float32 [0, 1]."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((int(size), int(size)), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def save_rgb(path: Path, img: np.ndarray) -> None:
    """Save a float RGB or grayscale image."""
    arr = np.clip(img, 0.0, 1.0)
    if arr.ndim == 2:
        Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


def normalize01(x: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Normalize an array into [0, 1] for display."""
    arr = np.asarray(x, dtype=np.float32)
    if lo is None:
        lo = float(np.nanpercentile(arr, 1.0))
    if hi is None:
        hi = float(np.nanpercentile(arr, 99.0))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def normalize_z(x: np.ndarray) -> np.ndarray:
    """Normalize a vector to zero mean and unit standard deviation."""
    arr = np.asarray(x, dtype=np.float32)
    return ((arr - arr.mean()) / (arr.std() + 1e-8)).astype(np.float32)


def fft_log_mag_y(y: np.ndarray) -> np.ndarray:
    """Return fftshifted log magnitude for a luminance channel."""
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(y)))).astype(np.float32)


def matplotlib_pyplot():
    """Import matplotlib using a writable cache directory."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    return plt, LinearSegmentedColormap


def rotbind_cmap():
    """Return a blue-white-red colormap for negative/zero/positive masks."""
    _, cmap_cls = matplotlib_pyplot()
    return cmap_cls.from_list("rotbind_bwr", ["#1f5fd0", "#ffffff", "#d92727"], N=256)


def ring_masks_from_metadata(shape: tuple[int, int], metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Build unshifted positive and negative ring masks using fftfreq coordinates."""
    h, w = shape
    fy = np.fft.fftfreq(h).astype(np.float32)
    fx = np.fft.fftfreq(w).astype(np.float32)
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    radius = np.sqrt(fx_grid * fx_grid + fy_grid * fy_grid).astype(np.float32)
    pos = np.zeros((h, w), dtype=bool)
    neg = np.zeros((h, w), dtype=bool)
    for r0, r1 in metadata["pos_bands"]:
        pos |= (radius >= float(r0)) & (radius <= float(r1))
    for r0, r1 in metadata["neg_bands"]:
        neg |= (radius >= float(r0)) & (radius <= float(r1))
    return pos, neg


def ring_bands_rgb(shape: tuple[int, int], metadata: dict[str, Any]) -> np.ndarray:
    """Create a shifted RGB mask with positive rings red and negative rings blue."""
    pos, neg = ring_masks_from_metadata(shape, metadata)
    pos = np.fft.fftshift(pos)
    neg = np.fft.fftshift(neg)
    rgb = np.ones(shape + (3,), dtype=np.float32)
    rgb[pos] = np.array([1.0, 0.15, 0.15], dtype=np.float32)
    rgb[neg] = np.array([0.15, 0.35, 1.0], dtype=np.float32)
    return rgb


def add_ring_overlays(ax: Any, shape: tuple[int, int], metadata: dict[str, Any]) -> None:
    """Draw R+ and R- inner/outer boundaries on shifted frequency plots."""
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D

    h, w = shape
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    max_ref = min(h, w)
    for r0, r1 in metadata["pos_bands"]:
        for r in (r0, r1):
            ax.add_patch(
                patches.Circle(center, float(r) * max_ref, fill=False, edgecolor="red", linewidth=1.2)
            )
    for r0, r1 in metadata["neg_bands"]:
        for r in (r0, r1):
            ax.add_patch(
                patches.Circle(
                    center,
                    float(r) * max_ref,
                    fill=False,
                    edgecolor="blue",
                    linestyle="--",
                    linewidth=1.2,
                )
            )
    handles = [
        Line2D([0], [0], color="red", lw=1.5, label="R+ positive rings"),
        Line2D([0], [0], color="blue", lw=1.5, linestyle="--", label="R- negative rings"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.85)


def compute_signature_and_corr(
    img: np.ndarray,
    metadata: dict[str, Any],
    rotation_gt_deg: float | None = None,
) -> dict[str, Any]:
    """Compute ring-difference signature and circular correlation."""
    all_bands = metadata["pos_bands"] + metadata["neg_bands"]
    rmin = min(float(b[0]) for b in all_bands)
    rmax = max(float(b[1]) for b in all_bands)
    polar, info = fft_polar_log_magnitude(
        img,
        rmin,
        rmax,
        num_r=64,
        num_angles=int(metadata["num_angles"]),
        normalize="per_radius",
    )
    signature, _ = extract_metadata_ring_difference_signature(polar, info["r_values"], metadata)
    angle_period = 180.0 if metadata.get("pi_periodic", True) else 360.0
    corr_shift_deg, score, corr, corr_info = circular_correlation_shift(
        signature,
        metadata["angular_code"],
        angle_period=angle_period,
    )
    rotation_hat_deg = shift_to_attack_angle(corr_shift_deg, angle_period)
    rotation_error_deg = (
        circular_angle_error(rotation_hat_deg, rotation_gt_deg, period=angle_period)
        if rotation_gt_deg is not None
        else float("nan")
    )
    return {
        "signature": signature,
        "angle_period": angle_period,
        "corr_shift_deg": corr_shift_deg,
        "corr_shift_display_deg": wrap_angle_signed(corr_shift_deg, period=angle_period),
        "rotation_hat_deg": rotation_hat_deg,
        "rotation_hat_display_deg": wrap_angle_signed(rotation_hat_deg, period=angle_period),
        "rotation_gt_deg": float(rotation_gt_deg) if rotation_gt_deg is not None else float("nan"),
        "rotation_error_deg": rotation_error_deg,
        "score": score,
        "corr": corr,
        "corr_info": corr_info,
        # Deprecated compatibility aliases.
        "theta_shift": corr_shift_deg,
        "theta_attack_hat": rotation_hat_deg,
    }


def build_diagnostics(img: np.ndarray, alpha: float, method: str, key: int, theta: float) -> dict[str, Any]:
    """Build all RotBind visualization diagnostics from actual module functions."""
    h, w = img.shape[:2]
    mask, metadata = make_ring_pair_mask(h, w, num_angles=360, key=key, method=method)
    anchored = embed_rotbind_anchor_rgb(img, mask, alpha)

    y_orig = rgb_to_ycbcr(img)[..., 0]
    y_anchor = rgb_to_ycbcr(anchored)[..., 0]
    log_orig = fft_log_mag_y(y_orig)
    log_anchor = fft_log_mag_y(y_anchor)
    canonical_corr = compute_signature_and_corr(anchored, metadata)

    attack: dict[str, Any] | None = None
    if abs(float(theta)) > 1e-12:
        x_att = rotate_image_keep_size(anchored, float(theta))
        rotation_hat_deg, score, corr, info = detect_rotbind_angle(
            np.clip(x_att, 0.0, 1.0).astype(np.float32),
            metadata,
            num_r=64,
            resolve_ambiguity=False,
        )
        angle_period = 180.0 if metadata.get("pi_periodic", True) else 360.0
        corr_shift_deg = float(info.get("corr_shift_deg", float("nan")))
        rotation_error_deg = circular_angle_error(rotation_hat_deg, theta, period=angle_period)
        x_corr = rotate_image_keep_size(x_att, -rotation_hat_deg)
        x_corr = np.clip(x_corr, 0.0, 1.0).astype(np.float32)
        x_clean = remove_rotbind_anchor_rgb(x_corr, mask, alpha)
        attack = {
            "x_att": np.clip(x_att, 0.0, 1.0).astype(np.float32),
            "x_corr": x_corr,
            "x_clean": x_clean,
            "corr_shift_deg": corr_shift_deg,
            "corr_shift_display_deg": wrap_angle_signed(corr_shift_deg, period=angle_period),
            "rotation_gt_deg": float(theta),
            "rotation_hat_deg": rotation_hat_deg,
            "rotation_hat_display_deg": wrap_angle_signed(rotation_hat_deg, period=angle_period),
            "rotation_error_deg": rotation_error_deg,
            "score": score,
            "corr": corr,
            "corr_info": info,
            "angle_period": angle_period,
            # Deprecated compatibility aliases.
            "theta_shift": corr_shift_deg,
            "theta_attack_hat": rotation_hat_deg,
            "theta_hat": rotation_hat_deg,
            "angle_error": rotation_error_deg,
        }

    return {
        "metadata": metadata,
        "mask": mask,
        "mask_shifted": np.fft.fftshift(mask),
        "gain": 1.0 + float(alpha) * mask,
        "gain_shifted": np.fft.fftshift(1.0 + float(alpha) * mask),
        "anchored": anchored,
        "y_orig": y_orig,
        "y_anchor": y_anchor,
        "log_orig": log_orig,
        "log_anchor": log_anchor,
        "log_diff": log_anchor - log_orig,
        "ring_bands": ring_bands_rgb((h, w), metadata),
        **canonical_corr,
        "attack": attack,
    }


def save_panel_figure(path: Path, draw_fn: Any, figsize: tuple[float, float] = (6, 5)) -> None:
    """Save a single matplotlib panel."""
    plt, _ = matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(fig, ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_image(ax: Any, img: np.ndarray, title: str) -> None:
    """Plot RGB image panel."""
    ax.imshow(np.clip(img, 0.0, 1.0))
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_gray(ax: Any, img: np.ndarray, title: str) -> None:
    """Plot grayscale image panel."""
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_log_diff(ax: Any, diagnostics: dict[str, Any], title: str) -> None:
    """Plot log FFT difference panel."""
    cmap = rotbind_cmap()
    vmax = float(np.nanpercentile(np.abs(diagnostics["log_diff"]), 99.0))
    ax.imshow(diagnostics["log_diff"], cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_mask_shifted(ax: Any, diagnostics: dict[str, Any], title: str) -> None:
    """Plot shifted modulation mask with ring overlays."""
    cmap = rotbind_cmap()
    ax.imshow(diagnostics["mask_shifted"], cmap=cmap, vmin=-1.0, vmax=1.0)
    add_ring_overlays(ax, diagnostics["mask_shifted"].shape, diagnostics["metadata"])
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_gain_shifted(fig: Any, ax: Any, diagnostics: dict[str, Any], alpha: float, title: str) -> None:
    """Plot shifted gain map with fixed alpha range."""
    im = ax.imshow(diagnostics["gain_shifted"], cmap="viridis", vmin=1.0 - alpha, vmax=1.0 + alpha)
    add_ring_overlays(ax, diagnostics["gain_shifted"].shape, diagnostics["metadata"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_ring_bands(ax: Any, diagnostics: dict[str, Any]) -> None:
    """Plot positive/negative ring bands."""
    ax.imshow(diagnostics["ring_bands"])
    add_ring_overlays(ax, diagnostics["ring_bands"].shape[:2], diagnostics["metadata"])
    ax.set_title("positive / negative ring bands", fontsize=10)
    ax.axis("off")


def plot_signature_vs_key(ax: Any, diagnostics: dict[str, Any]) -> None:
    """Plot observed ring-difference signature and key code together."""
    d_norm = normalize_z(diagnostics["signature"])
    p_norm = normalize_z(diagnostics["metadata"]["angular_code"])
    xs = np.linspace(0.0, 360.0, len(d_norm), endpoint=False)
    ax.plot(xs, d_norm, label="D(phi): observed ring-difference signature", linewidth=1.4)
    ax.plot(xs, p_norm, label="P_K(phi): key angular code", linewidth=1.0, alpha=0.75)
    ax.set_title("ring-difference signature vs key code", fontsize=10)
    ax.set_xlabel("angle phi (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")


def plot_correlation_curve(ax: Any, corr_info: dict[str, Any], title_prefix: str) -> None:
    """Plot circular correlation curve with image-rotation semantics."""
    corr = np.asarray(corr_info["corr"], dtype=np.float32)
    angle_period = float(corr_info["angle_period"])
    xs = np.arange(len(corr), dtype=np.float32) / len(corr) * 360.0
    corr_shift_deg = float(corr_info["corr_shift_deg"])
    rotation_hat_deg = float(corr_info["rotation_hat_deg"])
    rotation_gt_deg = float(corr_info.get("rotation_gt_deg", np.nan))
    rotation_error_deg = float(corr_info.get("rotation_error_deg", np.nan))
    best_idx = int(corr_info["corr_info"].get("angle_bin", int(np.argmax(corr))))
    corr_shift_full_deg = float(
        corr_info["corr_info"].get("corr_shift_full_deg", corr_info["corr_info"].get("theta_full", best_idx / len(corr) * 360.0))
    )
    margin = float(corr_info["corr_info"].get("corr_margin", np.nan))
    ax.plot(xs, corr, color="black", linewidth=1.2)
    ax.axvline(corr_shift_full_deg, color="red", linewidth=1.5, label=f"corr shift = {corr_shift_deg:.1f} deg")
    ax.scatter([corr_shift_full_deg], [corr[best_idx]], color="red", s=24)
    gt_line = f"GT rotation: {rotation_gt_deg:.1f} deg\n" if np.isfinite(rotation_gt_deg) else ""
    err_line = f"Error: {rotation_error_deg:.1f} deg\n" if np.isfinite(rotation_error_deg) else ""
    ax.text(
        0.02,
        0.96,
        f"{gt_line}"
        f"Estimated rotation: {rotation_hat_deg:.1f} deg\n"
        f"{err_line}"
        f"Debug corr shift: {corr_shift_deg:.1f} deg\n"
        f"best peak = {corr_shift_full_deg:.1f} deg\n"
        f"corr_margin = {margin:.3g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.85},
    )
    suffix = f"period = {angle_period:.0f} deg"
    if angle_period == 180.0:
        suffix += "; 0/180 ambiguity"
    ax.set_title(f"{title_prefix}\n{suffix}", fontsize=10)
    ax.set_xlabel("correlation shift (deg), not image rotation")
    ax.set_ylabel("correlation")
    ax.set_xlim(0.0, 360.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")


def save_single_panels(outdir: Path, img: np.ndarray, diagnostics: dict[str, Any], alpha: float) -> None:
    """Save paper-friendly split panel images."""
    panels = outdir / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    anchored = diagnostics["anchored"]
    diff_raw = np.clip(anchored - img + 0.5, 0.0, 1.0)
    diff_x50 = np.clip((anchored - img) * 50.0 + 0.5, 0.0, 1.0)

    save_rgb(panels / "original_rgb.png", img)
    save_rgb(panels / "anchored_rgb.png", anchored)
    save_rgb(panels / "diff_rgb_raw.png", diff_raw)
    save_rgb(panels / "diff_rgb_x50.png", diff_x50)
    if diagnostics["attack"] is not None:
        save_rgb(panels / "attacked_rgb.png", diagnostics["attack"]["x_att"])
        save_rgb(panels / "corrected_rgb.png", diagnostics["attack"]["x_corr"])
        save_rgb(panels / "cleaned_rgb.png", diagnostics["attack"]["x_clean"])

    save_panel_figure(panels / "log_fft_diff.png", lambda fig, ax: plot_log_diff(ax, diagnostics, "log_fft_diff"))
    save_panel_figure(
        panels / "modulation_mask_shifted.png",
        lambda fig, ax: plot_mask_shifted(ax, diagnostics, "modulation mask M_K (fftshifted)"),
    )
    save_panel_figure(
        panels / "gain_map_shifted.png",
        lambda fig, ax: plot_gain_shifted(fig, ax, diagnostics, alpha, "gain map 1 + alpha M_K (fftshifted)"),
    )
    save_panel_figure(panels / "ring_bands_shifted.png", lambda fig, ax: plot_ring_bands(ax, diagnostics))
    save_panel_figure(panels / "signature_vs_key.png", lambda fig, ax: plot_signature_vs_key(ax, diagnostics), (7, 4))
    corr_source = diagnostics["attack"] if diagnostics["attack"] is not None else diagnostics
    corr_title = "correlation curve after attack" if diagnostics["attack"] is not None else "circular correlation curve C(k)"
    save_panel_figure(
        panels / "correlation_curve.png",
        lambda fig, ax: plot_correlation_curve(ax, corr_source, corr_title),
        (7, 4),
    )
    if diagnostics["attack"] is not None:
        save_panel_figure(
            panels / "correlation_curve_canonical.png",
            lambda fig, ax: plot_correlation_curve(ax, diagnostics, "canonical circular correlation curve C(k)"),
            (7, 4),
        )


def save_grid(outdir: Path, img: np.ndarray, diagnostics: dict[str, Any], alpha: float, method: str, key: int, theta: float) -> None:
    """Save the diagnostic grid."""
    plt, _ = matplotlib_pyplot()
    anchored = diagnostics["anchored"]
    diff_x50 = np.clip((anchored - img) * 50.0 + 0.5, 0.0, 1.0)
    title_suffix = f"alpha={alpha:g}, method={method}, key={key}"
    attack = diagnostics["attack"]

    base_panels = [
        ("original_rgb", "image", img),
        ("original_Y", "gray", diagnostics["y_orig"]),
        ("anchored_rgb", "image", anchored),
        ("anchored_Y", "gray", diagnostics["y_anchor"]),
        ("diff_rgb_raw", "image", np.clip(anchored - img + 0.5, 0.0, 1.0)),
        ("diff_rgb_x50", "image", diff_x50),
        ("log_fft_Y_original", "magma", normalize01(diagnostics["log_orig"])),
        ("log_fft_Y_anchor", "magma", normalize01(diagnostics["log_anchor"])),
        ("log_fft_diff", "logdiff", None),
        ("modulation mask M_K (fftshifted)", "mask", None),
        ("gain map 1 + alpha M_K (fftshifted)", "gain", None),
        ("positive / negative ring bands", "rings", None),
        ("ring-difference signature vs key code", "signature", None),
        ("circular correlation curve C(k)", "corr", diagnostics),
    ]
    if attack is not None:
        base_panels.extend(
            [
                ("attacked_rgb", "image", attack["x_att"]),
                (
                    "corrected_rgb\n"
                    f"GT rotation = {theta:.1f} deg\n"
                    f"Estimated rotation = {attack['rotation_hat_deg']:.1f} deg\n"
                    f"Error = {attack['rotation_error_deg']:.1f} deg",
                    "image",
                    attack["x_corr"],
                ),
                ("cleaned_rgb", "image", attack["x_clean"]),
                ("correlation curve after attack", "corr", attack),
            ]
        )

    cols = 4
    rows = int(np.ceil(len(base_panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.2 * rows))
    axes_flat = np.asarray(axes).reshape(-1)
    for ax, (name, kind, data) in zip(axes_flat, base_panels):
        if kind == "image":
            plot_image(ax, data, f"{name}\n{title_suffix}")
        elif kind == "gray":
            plot_gray(ax, data, f"{name}\n{title_suffix}")
        elif kind == "magma":
            ax.imshow(data, cmap="magma", vmin=0.0, vmax=1.0)
            ax.set_title(f"{name}\n{title_suffix}", fontsize=10)
            ax.axis("off")
        elif kind == "logdiff":
            plot_log_diff(ax, diagnostics, f"{name}\n{title_suffix}")
        elif kind == "mask":
            plot_mask_shifted(ax, diagnostics, f"{name}\n{title_suffix}")
        elif kind == "gain":
            plot_gain_shifted(fig, ax, diagnostics, alpha, f"{name}\n{title_suffix}")
        elif kind == "rings":
            plot_ring_bands(ax, diagnostics)
        elif kind == "signature":
            plot_signature_vs_key(ax, diagnostics)
        elif kind == "corr":
            plot_correlation_curve(ax, data, f"{name}\n{title_suffix}")
    for ax in axes_flat[len(base_panels) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(outdir / "diagnostic_grid.png", dpi=150)
    fig.savefig(outdir / "rotbind_diagnostic_grid.png", dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--method", choices=["two_pair", "multi_ringpair"], default="two_pair")
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--angle", type=float, default=None, help="Deprecated alias for --theta")
    parser.add_argument("--key", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run RotBind embedding visualization."""
    args = parse_args(argv)
    theta = float(args.theta if args.angle is None else args.angle)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    img = load_rgb(Path(args.image), args.size)
    diagnostics = build_diagnostics(img, args.alpha, args.method, args.key, theta)
    anchored = diagnostics["anchored"]

    diff_raw = np.clip(anchored - img + 0.5, 0.0, 1.0)
    diff_x50 = np.clip((anchored - img) * 50.0 + 0.5, 0.0, 1.0)
    save_rgb(outdir / "diff_rgb_raw.png", diff_raw)
    save_rgb(outdir / "diff_rgb_x50.png", diff_x50)
    save_single_panels(outdir, img, diagnostics, args.alpha)
    save_grid(outdir, img, diagnostics, args.alpha, args.method, args.key, theta)

    attack = diagnostics["attack"]
    print(f"wrote {outdir / 'diagnostic_grid.png'}")
    if attack is None:
        print(
            f"corr_shift_deg={diagnostics['corr_shift_deg']:.6f}, "
            f"rotation_hat_deg={diagnostics['rotation_hat_deg']:.6f}, score={diagnostics['score']:.6f}"
        )
    else:
        print(
            f"rotation_gt_deg={theta:.6f}, corr_shift_deg={attack['corr_shift_deg']:.6f}, "
            f"rotation_hat_deg={attack['rotation_hat_deg']:.6f}, "
            f"rotation_error_deg={attack['rotation_error_deg']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
