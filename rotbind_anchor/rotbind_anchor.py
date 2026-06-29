"""Core RotBind pixel-frequency rotation synchronization anchor.

This module implements only the standalone anchor mechanics: angular code
generation, Fourier-domain embedding, polar feature extraction, angle
detection, inverse anchor removal, and negative-anchor construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


DEFAULT_POS_BANDS = [(0.14, 0.18), (0.24, 0.28)]
DEFAULT_NEG_BANDS = [(0.19, 0.23), (0.29, 0.33)]


def _as_float32_array(x: np.ndarray, name: str) -> np.ndarray:
    """Return an ndarray view/copy as float32 and reject non-arrays."""
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    return x.astype(np.float32, copy=False)


def _check_rgb_image(img_rgb: np.ndarray, name: str = "img_rgb") -> np.ndarray:
    """Validate a float RGB image in [0, 1]."""
    img = _as_float32_array(img_rgb, name)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{name} must have shape [H, W, 3]")
    if not np.all(np.isfinite(img)):
        raise ValueError(f"{name} must contain only finite values")
    if float(img.min()) < -1e-6 or float(img.max()) > 1.0 + 1e-6:
        raise ValueError(f"{name} must be in range [0, 1]")
    return img


def _check_2d(x: np.ndarray, name: str) -> np.ndarray:
    """Validate a two-dimensional array."""
    arr = _as_float32_array(x, name)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [H, W]")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_bands(bands: list[tuple[float, float]], name: str) -> list[tuple[float, float]]:
    """Validate normalized frequency bands."""
    if not bands:
        raise ValueError(f"{name} must contain at least one band")
    out: list[tuple[float, float]] = []
    for band in bands:
        if len(band) != 2:
            raise ValueError(f"each {name} entry must be (rmin, rmax)")
        rmin, rmax = float(band[0]), float(band[1])
        if not (0.0 <= rmin < rmax <= 0.5):
            raise ValueError(f"{name} bands must satisfy 0 <= rmin < rmax <= 0.5")
        out.append((rmin, rmax))
    return out


def rgb_to_ycbcr(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB float image in [0, 1] to full-range YCbCr."""
    img = _check_rgb_image(img_rgb)
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return np.stack([y, cb, cr], axis=-1).astype(np.float32)


def ycbcr_to_rgb(img_ycbcr: np.ndarray) -> np.ndarray:
    """Convert full-range YCbCr float image to RGB."""
    img = _as_float32_array(img_ycbcr, "img_ycbcr")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img_ycbcr must have shape [H, W, 3]")
    y = img[..., 0]
    cb = img[..., 1] - 0.5
    cr = img[..., 2] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def normalize_zero_mean_unit_std(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize an array to zero mean and unit standard deviation."""
    arr = _as_float32_array(x, "x")
    mean = np.mean(arr, dtype=np.float64)
    std = np.std(arr, dtype=np.float64)
    return ((arr - mean) / max(float(std), eps)).astype(np.float32)


def circular_angle_error(theta_hat: float, theta_gt: float, period: float = 360.0) -> float:
    """Return the minimal absolute circular error between two angles."""
    if period <= 0:
        raise ValueError("period must be positive")
    err = (float(theta_hat) - float(theta_gt) + period / 2.0) % period - period / 2.0
    return float(abs(err))


def rotate_image_keep_size(img: np.ndarray, angle: float, mode: str = "reflect") -> np.ndarray:
    """Rotate a 2D or RGB image with scipy.ndimage.rotate while preserving shape."""
    arr = _as_float32_array(img, "img")
    if arr.ndim not in (2, 3):
        raise ValueError("img must be a 2D array or an RGB image")
    if arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError("3D img must have shape [H, W, 3]")
    rotated = ndimage.rotate(
        arr,
        float(angle),
        axes=(0, 1),
        reshape=False,
        order=1,
        mode=mode,
        prefilter=False,
    )
    return rotated.astype(np.float32)


def make_angular_code(
    num_angles: int,
    key: int = 0,
    mode: str = "rademacher",
    pi_periodic: bool = True,
) -> np.ndarray:
    """Create a key-controlled angular code."""
    if num_angles <= 0:
        raise ValueError("num_angles must be positive")
    if mode != "rademacher":
        raise ValueError(f"unsupported angular code mode: {mode}")
    rng = np.random.default_rng(int(key))
    if pi_periodic:
        if num_angles % 2 != 0:
            raise ValueError("num_angles must be even when pi_periodic=True")
        half = num_angles // 2
        code = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=half)
        code = np.concatenate([code, code], axis=0)
    else:
        code = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=num_angles)
    return normalize_zero_mean_unit_std(code)


def soft_bandpass_radius(
    radius: np.ndarray,
    rmin: float,
    rmax: float,
    transition: float = 0.01,
) -> np.ndarray:
    """Build a raised-cosine soft bandpass window over normalized radius."""
    r = _as_float32_array(radius, "radius")
    rmin = float(rmin)
    rmax = float(rmax)
    transition = float(transition)
    if not (0.0 <= rmin < rmax <= 0.5):
        raise ValueError("expected 0 <= rmin < rmax <= 0.5")
    if transition <= 0:
        return ((r >= rmin) & (r <= rmax)).astype(np.float32)

    window = np.zeros_like(r, dtype=np.float32)
    inside = (r >= rmin) & (r <= rmax)
    window[inside] = 1.0

    lo = (r > rmin - transition) & (r < rmin)
    if np.any(lo):
        t = (r[lo] - (rmin - transition)) / transition
        window[lo] = 0.5 - 0.5 * np.cos(np.pi * t)

    hi = (r > rmax) & (r < rmax + transition)
    if np.any(hi):
        t = (r[hi] - rmax) / transition
        window[hi] = 0.5 + 0.5 * np.cos(np.pi * t)

    return np.clip(window, 0.0, 1.0).astype(np.float32)


def make_ring_pair_mask(
    H: int,
    W: int,
    pos_bands: list[tuple[float, float]] | None = None,
    neg_bands: list[tuple[float, float]] | None = None,
    num_angles: int = 360,
    key: int = 0,
    mode: str = "rademacher",
    pi_periodic: bool = True,
    transition: float = 0.01,
    method: str = "two_pair",
    num_ring_pairs: int = 12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct an unshifted FFT-grid differential ring-pair modulation mask."""
    H = int(H)
    W = int(W)
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive")
    if method not in {"two_pair", "multi_ringpair"}:
        raise ValueError("method must be 'two_pair' or 'multi_ringpair'")
    angular_code = make_angular_code(num_angles, key=key, mode=mode, pi_periodic=pi_periodic)

    ring_pairs: list[dict[str, Any]] = []
    if method == "two_pair":
        pos_bands = _validate_bands(pos_bands or DEFAULT_POS_BANDS, "pos_bands")
        neg_bands = _validate_bands(neg_bands or DEFAULT_NEG_BANDS, "neg_bands")
        for pos_band, neg_band in zip(pos_bands, neg_bands):
            ring_pairs.append({"pos_band": pos_band, "neg_band": neg_band, "sign": 1})
    else:
        num_ring_pairs = int(num_ring_pairs)
        if not 8 <= num_ring_pairs <= 16:
            raise ValueError("num_ring_pairs must be in [8, 16] for method='multi_ringpair'")
        pair_rng = np.random.default_rng(int(key) + 7919)
        pair_signs = pair_rng.choice(np.array([-1, 1], dtype=np.int32), size=num_ring_pairs)
        r0, r1 = 0.12, 0.36
        span = (r1 - r0) / num_ring_pairs
        pos_bands = []
        neg_bands = []
        for idx in range(num_ring_pairs):
            start = r0 + idx * span
            pos_band = (start + 0.12 * span, start + 0.38 * span)
            neg_band = (start + 0.62 * span, start + 0.88 * span)
            sign = int(pair_signs[idx])
            pos_bands.append(pos_band)
            neg_bands.append(neg_band)
            ring_pairs.append({"pos_band": pos_band, "neg_band": neg_band, "sign": sign})

    fy = np.fft.fftfreq(H).astype(np.float32)
    fx = np.fft.fftfreq(W).astype(np.float32)
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    radius = np.sqrt(fx_grid * fx_grid + fy_grid * fy_grid).astype(np.float32)
    angle = np.mod(np.arctan2(fy_grid, fx_grid), 2.0 * np.pi)
    angle_idx = np.floor(angle / (2.0 * np.pi) * num_angles).astype(np.int64)
    angle_idx = np.clip(angle_idx, 0, num_angles - 1)
    code_grid = angular_code[angle_idx]

    pair_window = np.zeros((H, W), dtype=np.float32)
    for pair in ring_pairs:
        pos0, pos1 = pair["pos_band"]
        neg0, neg1 = pair["neg_band"]
        pos_window = soft_bandpass_radius(radius, pos0, pos1, transition=transition)
        neg_window = soft_bandpass_radius(radius, neg0, neg1, transition=transition)
        pair_window += float(pair["sign"]) * (pos_window - neg_window)

    modulation_grid = pair_window * code_grid

    if pi_periodic:
        y_neg = (-np.arange(H)) % H
        x_neg = (-np.arange(W)) % W
        modulation_grid = 0.5 * (modulation_grid + modulation_grid[np.ix_(y_neg, x_neg)])

    max_abs = float(np.max(np.abs(modulation_grid)))
    if max_abs > 0:
        modulation_grid = modulation_grid / max_abs

    metadata: dict[str, Any] = {
        "H": H,
        "W": W,
        "pos_bands": pos_bands,
        "neg_bands": neg_bands,
        "num_angles": int(num_angles),
        "key": int(key),
        "mode": mode,
        "pi_periodic": bool(pi_periodic),
        "transition": float(transition),
        "method": method,
        "num_ring_pairs": int(len(ring_pairs)),
        "ring_pairs": ring_pairs,
        "angular_code": angular_code.astype(np.float32),
    }
    return modulation_grid.astype(np.float32), metadata


def embed_rotbind_anchor_rgb(
    img_rgb: np.ndarray,
    modulation_grid: np.ndarray,
    alpha: float,
    gain_clip: tuple[float, float] = (0.1, 10.0),
) -> np.ndarray:
    """Embed the RotBind anchor into the Y-channel Fourier magnitude."""
    img = _check_rgb_image(img_rgb)
    mod = _check_2d(modulation_grid, "modulation_grid")
    if mod.shape != img.shape[:2]:
        raise ValueError("modulation_grid shape must match image height and width")
    ycbcr = rgb_to_ycbcr(img)
    y = ycbcr[..., 0]
    gain = np.clip(1.0 + float(alpha) * mod, gain_clip[0], gain_clip[1])
    f_y = np.fft.fft2(y)
    y_sync = np.fft.ifft2(f_y * gain).real
    ycbcr_sync = ycbcr.copy()
    ycbcr_sync[..., 0] = np.clip(y_sync, 0.0, 1.0).astype(np.float32)
    return np.clip(ycbcr_to_rgb(ycbcr_sync), 0.0, 1.0).astype(np.float32)


def fft_polar_log_magnitude(
    img_rgb_or_y: np.ndarray,
    rmin: float,
    rmax: float,
    num_r: int = 64,
    num_angles: int = 360,
    normalize: str = "per_radius",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample centered FFT log magnitude on a polar grid."""
    arr = _as_float32_array(img_rgb_or_y, "img_rgb_or_y")
    if arr.ndim == 3:
        y = rgb_to_ycbcr(arr)[..., 0]
    elif arr.ndim == 2:
        y = arr
    else:
        raise ValueError("img_rgb_or_y must be RGB [H, W, 3] or 2D [H, W]")
    if not (0.0 <= float(rmin) < float(rmax) <= 0.5):
        raise ValueError("expected 0 <= rmin < rmax <= 0.5")
    if num_r <= 0 or num_angles <= 0:
        raise ValueError("num_r and num_angles must be positive")

    H, W = y.shape
    f_y = np.fft.fftshift(np.fft.fft2(y))
    log_mag = np.log1p(np.abs(f_y)).astype(np.float32)

    r_values = np.linspace(float(rmin), float(rmax), int(num_r), dtype=np.float32)
    theta_values = np.linspace(0.0, 2.0 * np.pi, int(num_angles), endpoint=False, dtype=np.float32)
    rr, tt = np.meshgrid(r_values, theta_values, indexing="ij")
    fy = rr * np.sin(tt)
    fx = rr * np.cos(tt)
    rows = H // 2 + fy * H
    cols = W // 2 + fx * W
    polar = ndimage.map_coordinates(
        log_mag,
        [rows, cols],
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32)

    if normalize == "per_radius":
        mean = polar.mean(axis=1, keepdims=True, dtype=np.float64)
        std = polar.std(axis=1, keepdims=True, dtype=np.float64)
        polar = ((polar - mean) / np.maximum(std, 1e-8)).astype(np.float32)
    elif normalize == "global":
        polar = normalize_zero_mean_unit_std(polar)
    elif normalize == "none":
        polar = polar.astype(np.float32)
    else:
        raise ValueError("normalize must be 'per_radius', 'global', or 'none'")

    info: dict[str, Any] = {
        "H": H,
        "W": W,
        "r_values": r_values,
        "theta_values": theta_values,
        "normalize": normalize,
    }
    return polar.astype(np.float32), info


def band_indices_for_polar_grid(
    r_values: np.ndarray,
    bands: list[tuple[float, float]],
) -> list[int]:
    """Return polar radius indices that fall within any of the supplied bands."""
    r = _as_float32_array(r_values, "r_values")
    if r.ndim != 1:
        raise ValueError("r_values must be one-dimensional")
    bands = _validate_bands(bands, "bands")
    keep = np.zeros(r.shape, dtype=bool)
    for r0, r1 in bands:
        keep |= (r >= r0) & (r <= r1)
    indices = np.flatnonzero(keep).astype(int).tolist()
    if not indices:
        raise ValueError(f"no polar radius indices found for bands {bands}")
    return indices


def extract_ring_difference_signature(
    polar_feature: np.ndarray,
    pos_r_indices: list[int],
    neg_r_indices: list[int],
) -> np.ndarray:
    """Extract the normalized positive-minus-negative ring signature."""
    polar = _as_float32_array(polar_feature, "polar_feature")
    if polar.ndim != 2:
        raise ValueError("polar_feature must have shape [num_r, num_angles]")
    if not pos_r_indices or not neg_r_indices:
        raise ValueError("pos_r_indices and neg_r_indices must be non-empty")
    pos = polar[np.asarray(pos_r_indices, dtype=np.int64), :].mean(axis=0)
    neg = polar[np.asarray(neg_r_indices, dtype=np.int64), :].mean(axis=0)
    return normalize_zero_mean_unit_std(pos - neg)


def extract_metadata_ring_difference_signature(
    polar_feature: np.ndarray,
    r_values: np.ndarray,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract a metadata-aware signed ring-difference signature."""
    method = metadata.get("method", "two_pair")
    if method == "multi_ringpair":
        polar = _as_float32_array(polar_feature, "polar_feature")
        if polar.ndim != 2:
            raise ValueError("polar_feature must have shape [num_r, num_angles]")
        signed_sum = np.zeros(polar.shape[1], dtype=np.float32)
        pair_infos = []
        for pair in metadata.get("ring_pairs", []):
            pos_idx = band_indices_for_polar_grid(r_values, [tuple(pair["pos_band"])])
            neg_idx = band_indices_for_polar_grid(r_values, [tuple(pair["neg_band"])])
            pos = polar[np.asarray(pos_idx, dtype=np.int64), :].mean(axis=0)
            neg = polar[np.asarray(neg_idx, dtype=np.int64), :].mean(axis=0)
            signed_sum += float(pair["sign"]) * (pos - neg)
            pair_infos.append({"pos_r_indices": pos_idx, "neg_r_indices": neg_idx, "sign": int(pair["sign"])})
        if not pair_infos:
            raise ValueError("metadata['ring_pairs'] must be non-empty for method='multi_ringpair'")
        return normalize_zero_mean_unit_std(signed_sum), {"ring_pair_indices": pair_infos}

    pos_bands = _validate_bands(metadata["pos_bands"], "pos_bands")
    neg_bands = _validate_bands(metadata["neg_bands"], "neg_bands")
    pos_idx = band_indices_for_polar_grid(r_values, pos_bands)
    neg_idx = band_indices_for_polar_grid(r_values, neg_bands)
    signature = extract_ring_difference_signature(polar_feature, pos_idx, neg_idx)
    return signature, {"pos_r_indices": pos_idx, "neg_r_indices": neg_idx}


def circular_correlation_angle(
    signature: np.ndarray,
    angular_code: np.ndarray,
    angle_period: float = 180.0,
) -> tuple[float, float, np.ndarray, dict[str, Any]]:
    """Estimate angular shift with FFT-based circular correlation."""
    sig = normalize_zero_mean_unit_std(signature)
    code = normalize_zero_mean_unit_std(angular_code)
    if sig.ndim != 1 or code.ndim != 1:
        raise ValueError("signature and angular_code must be one-dimensional")
    if sig.shape[0] != code.shape[0]:
        raise ValueError("signature and angular_code must have the same length")
    if angle_period <= 0:
        raise ValueError("angle_period must be positive")

    corr = np.fft.ifft(np.fft.fft(sig) * np.conj(np.fft.fft(code))).real.astype(np.float32)
    best_idx = int(np.argmax(corr))
    best_score = float(corr[best_idx])
    if corr.size > 1:
        top = np.partition(corr, -2)[-2:]
        top2_score = float(np.min(top))
    else:
        top2_score = float("-inf")
    theta_full = float(best_idx / corr.size * 360.0)
    theta_mod = float(theta_full % float(angle_period))
    extra_info: dict[str, Any] = {
        "angle_bin": best_idx,
        "theta_full": theta_full,
        "top2_score": top2_score,
        "corr_margin": float(best_score - top2_score),
        "raw_shift": best_idx,
    }
    return theta_mod, best_score, corr.astype(np.float32), extra_info


def _polar_signature_from_metadata(
    img_rgb: np.ndarray,
    metadata: dict[str, Any],
    rmin: float | None = None,
    rmax: float | None = None,
    num_r: int = 64,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract a ring-difference signature using RotBind metadata."""
    pos_bands = _validate_bands(metadata["pos_bands"], "pos_bands")
    neg_bands = _validate_bands(metadata["neg_bands"], "neg_bands")
    all_bands = pos_bands + neg_bands
    if rmin is None:
        rmin = min(b[0] for b in all_bands)
    if rmax is None:
        rmax = max(b[1] for b in all_bands)
    polar, info = fft_polar_log_magnitude(
        img_rgb,
        float(rmin),
        float(rmax),
        num_r=num_r,
        num_angles=int(metadata["num_angles"]),
        normalize="per_radius",
    )
    signature, sig_info = extract_metadata_ring_difference_signature(polar, info["r_values"], metadata)
    info.update(sig_info)
    return signature, info


def detect_rotbind_angle(
    img_rgb: np.ndarray,
    metadata: dict[str, Any],
    rmin: float | None = None,
    rmax: float | None = None,
    num_r: int = 64,
    resolve_ambiguity: bool = True,
) -> tuple[float, float, np.ndarray, dict[str, Any]]:
    """Detect the RotBind rotation angle from an RGB image."""
    _check_rgb_image(img_rgb)
    angular_code = _as_float32_array(metadata["angular_code"], "metadata['angular_code']")
    signature, polar_info = _polar_signature_from_metadata(img_rgb, metadata, rmin, rmax, num_r)
    pi_periodic = bool(metadata.get("pi_periodic", True))
    angle_period = 180.0 if pi_periodic else 360.0
    theta_mod, best_score, score_curve, corr_info = circular_correlation_angle(
        signature,
        angular_code,
        angle_period=angle_period,
    )
    extra_info: dict[str, Any] = {
        **corr_info,
        "theta_mod": theta_mod,
        "angle_period": angle_period,
        "polar_info": polar_info,
    }
    theta_hat = theta_mod
    if pi_periodic and resolve_ambiguity:
        theta_hat, ambiguity_info = resolve_180_ambiguity(img_rgb, theta_mod, metadata, num_r=num_r)
        extra_info["ambiguity"] = ambiguity_info
    return float(theta_hat), float(best_score), score_curve.astype(np.float32), extra_info


def rotbind_anchor_score(
    img_rgb: np.ndarray,
    metadata: dict[str, Any],
    num_r: int = 64,
) -> tuple[float, dict[str, Any]]:
    """Return the best circular-correlation anchor score for an image."""
    _check_rgb_image(img_rgb)
    angular_code = _as_float32_array(metadata["angular_code"], "metadata['angular_code']")
    signature, polar_info = _polar_signature_from_metadata(img_rgb, metadata, num_r=num_r)
    angle_period = 180.0 if bool(metadata.get("pi_periodic", True)) else 360.0
    theta_mod, best_score, score_curve, corr_info = circular_correlation_angle(
        signature,
        angular_code,
        angle_period=angle_period,
    )
    info: dict[str, Any] = {
        **corr_info,
        "theta_mod": theta_mod,
        "angle_period": angle_period,
        "score_curve": score_curve,
        "polar_info": polar_info,
    }
    return float(best_score), info


def resolve_180_ambiguity(
    img_rgb: np.ndarray,
    theta_mod: float,
    metadata: dict[str, Any],
    num_r: int = 64,
    rotate_mode: str = "reflect",
) -> tuple[float, dict[str, Any]]:
    """Resolve pi-periodic ambiguity by scoring two corrected candidates."""
    img = _check_rgb_image(img_rgb)
    theta1 = float(theta_mod)
    theta2 = theta1 + 180.0
    img1 = rotate_image_keep_size(img, -theta1, mode=rotate_mode)
    img2 = rotate_image_keep_size(img, -theta2, mode=rotate_mode)
    score1, info1 = rotbind_anchor_score(np.clip(img1, 0.0, 1.0).astype(np.float32), metadata, num_r=num_r)
    score2, info2 = rotbind_anchor_score(np.clip(img2, 0.0, 1.0).astype(np.float32), metadata, num_r=num_r)
    if score1 >= score2:
        theta = theta1
        chosen = 1
    else:
        theta = theta2
        chosen = 2
    info: dict[str, Any] = {
        "theta1": theta1,
        "theta2": theta2,
        "score1": float(score1),
        "score2": float(score2),
        "chosen": chosen,
        "candidate1_info": info1,
        "candidate2_info": info2,
    }
    return float(theta), info


def remove_rotbind_anchor_rgb(
    img_rgb_corr: np.ndarray,
    modulation_grid: np.ndarray,
    alpha: float,
    gain_clip: tuple[float, float] = (0.1, 10.0),
) -> np.ndarray:
    """Remove the RotBind anchor by inverse Fourier-domain modulation."""
    img = _check_rgb_image(img_rgb_corr, "img_rgb_corr")
    mod = _check_2d(modulation_grid, "modulation_grid")
    if mod.shape != img.shape[:2]:
        raise ValueError("modulation_grid shape must match image height and width")
    ycbcr = rgb_to_ycbcr(img)
    gain = np.clip(1.0 + float(alpha) * mod, gain_clip[0], gain_clip[1])
    f_y = np.fft.fft2(ycbcr[..., 0])
    y_clean = np.fft.ifft2(f_y / gain).real
    ycbcr_clean = ycbcr.copy()
    ycbcr_clean[..., 0] = np.clip(y_clean, 0.0, 1.0).astype(np.float32)
    return np.clip(ycbcr_to_rgb(ycbcr_clean), 0.0, 1.0).astype(np.float32)


def make_negative_anchor_rgb(
    img_rgb_corr: np.ndarray,
    modulation_grid: np.ndarray,
    alpha: float,
    gain_clip: tuple[float, float] = (0.1, 10.0),
) -> np.ndarray:
    """Construct the negative-anchor image for later VAE symmetric cancellation."""
    img = _check_rgb_image(img_rgb_corr, "img_rgb_corr")
    mod = _check_2d(modulation_grid, "modulation_grid")
    if mod.shape != img.shape[:2]:
        raise ValueError("modulation_grid shape must match image height and width")
    ycbcr = rgb_to_ycbcr(img)
    gain_pos = np.clip(1.0 + float(alpha) * mod, gain_clip[0], gain_clip[1])
    gain_neg = np.clip(1.0 - float(alpha) * mod, gain_clip[0], gain_clip[1])
    f_y = np.fft.fft2(ycbcr[..., 0])
    y_minus = np.fft.ifft2(f_y * gain_neg / gain_pos).real
    ycbcr_minus = ycbcr.copy()
    ycbcr_minus[..., 0] = np.clip(y_minus, 0.0, 1.0).astype(np.float32)
    return np.clip(ycbcr_to_rgb(ycbcr_minus), 0.0, 1.0).astype(np.float32)


def _smoke_test() -> None:
    """Run a tiny end-to-end smoke test without evaluating accuracy."""
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    x = xx / 63.0
    y = yy / 63.0
    img = np.stack([x, y, 0.5 + 0.25 * np.sin(2.0 * np.pi * (x + y))], axis=-1)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    mask, metadata = make_ring_pair_mask(64, 64, key=0, num_angles=180)
    anchored = embed_rotbind_anchor_rgb(img, mask, alpha=0.02)
    rotated = rotate_image_keep_size(anchored, 45.0)
    theta_hat, _, _, _ = detect_rotbind_angle(rotated, metadata, num_r=32)
    corrected = rotate_image_keep_size(rotated, -theta_hat)
    removed = remove_rotbind_anchor_rgb(np.clip(corrected, 0.0, 1.0), mask, alpha=0.02)
    negative = make_negative_anchor_rgb(np.clip(corrected, 0.0, 1.0), mask, alpha=0.02)
    assert anchored.shape == rotated.shape == removed.shape == negative.shape == img.shape


if __name__ == "__main__":
    _smoke_test()
