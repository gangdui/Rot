"""Regression tests for RotBind rotation-angle naming semantics."""

from rotbind_anchor.rotbind_anchor import circular_angle_error, shift_to_attack_angle, wrap_angle_signed
from rotbind_anchor.visualize_rotbind_embedding import format_rotation_display


def test_corr_shift_136_maps_to_rotation_44() -> None:
    assert shift_to_attack_angle(136.0, angle_period=180.0) == 44.0


def test_corr_shift_1_maps_to_rotation_179_and_display_minus_1() -> None:
    rotation_hat_deg = shift_to_attack_angle(1.0, angle_period=180.0)
    assert rotation_hat_deg == 179.0
    assert wrap_angle_signed(rotation_hat_deg, period=180.0) == -1.0


def test_rotation_error_45_vs_44_is_1() -> None:
    assert circular_angle_error(44.0, 45.0, period=180.0) == 1.0


def test_rotation_error_0_vs_179_is_1() -> None:
    assert circular_angle_error(179.0, 0.0, period=180.0) == 1.0


def test_visualization_display_shows_signed_rotation_with_mod_equivalence() -> None:
    assert format_rotation_display(-1.0, 179.0, 180.0) == "-1.0° (≡ 179.0° mod 180)"
    assert format_rotation_display(44.0, 44.0, 180.0) == "44.0°"
