"""Tests for RotBind circular-correlation peak refinement."""

import numpy as np

from rotbind_anchor.rotbind_anchor import circular_correlation_shift, shift_to_attack_angle


def _fractional_roll(x: np.ndarray, shift_bins: float) -> np.ndarray:
    freqs = np.fft.fftfreq(x.size)
    return np.fft.ifft(np.fft.fft(x) * np.exp(-2j * np.pi * freqs * shift_bins)).real


def _periodic_gaussian(n: int, sigma: float = 10.0) -> np.ndarray:
    xs = np.arange(n, dtype=np.float64)
    dist = np.minimum(xs, n - xs)
    return np.exp(-0.5 * (dist / sigma) ** 2).astype(np.float32)


def test_refine_peak_recovers_fractional_correlation_peak() -> None:
    code = _periodic_gaussian(360)
    signature = _fractional_roll(code, 135.4).astype(np.float32)

    corr_shift_deg, _, _, info = circular_correlation_shift(
        signature,
        code,
        angle_period=360.0,
        refine_peak=True,
    )

    assert abs(corr_shift_deg - 135.4) < 0.05
    assert info["angle_bin_int"] == 135
    assert abs(info["angle_bin_refined"] - 135.4) < 0.05


def test_refine_peak_false_returns_integer_bin() -> None:
    code = _periodic_gaussian(360)
    signature = _fractional_roll(code, 135.4).astype(np.float32)

    corr_shift_deg, _, _, info = circular_correlation_shift(
        signature,
        code,
        angle_period=360.0,
        refine_peak=False,
    )

    assert corr_shift_deg == 135.0
    assert info["angle_bin_refined"] == 135.0
    assert info["peak_refine_delta"] == 0.0


def test_fractional_corr_shift_converts_to_fractional_rotation_hat() -> None:
    assert shift_to_attack_angle(135.5, angle_period=180.0) == 44.5


def test_num_angles_720_gives_half_degree_resolution_without_refinement() -> None:
    code = _periodic_gaussian(720)
    signature = _fractional_roll(code, 271.0).astype(np.float32)

    corr_shift_deg, _, _, info = circular_correlation_shift(
        signature,
        code,
        angle_period=360.0,
        refine_peak=False,
    )

    assert info["angle_bin_int"] == 271
    assert corr_shift_deg == 135.5
