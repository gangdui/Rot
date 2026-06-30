"""Tests for scaled-vs-raw VAE footprint metrics."""

import numpy as np

from rotbind_anchor.eval_rotbind_anchor import DiffusersVaeEncoder, compute_rotated_corrected_vae_metrics


class FakeModel:
    class Config:
        scaling_factor = 0.5

    config = Config()


class FakeScaledVaeEncoder:
    def __init__(self, scaling_factor: float = 0.18215) -> None:
        self.scaling_factor = scaling_factor

    def encode_images_raw(self, images):
        return np.stack([np.asarray(img, dtype=np.float32).reshape(-1) for img in images], axis=0)

    def encode_images(self, images):
        return self.encode_images_raw(images) * self.scaling_factor


def test_diffusers_vae_encoder_reads_model_scaling_factor() -> None:
    encoder = DiffusersVaeEncoder(FakeModel(), use_scaling_factor=True)
    assert encoder.scaling_factor == 0.5
    assert encoder.use_scaling_factor is True


def test_main_vae_mse_uses_scaled_latent_and_raw_mse_is_debug() -> None:
    x = np.ones((2, 2, 1), dtype=np.float32)
    x_anchor = x + 2.0
    x_clean_pred = x + np.float32(np.sqrt(8.32))
    x_clean_oracle = x + 1.0
    x_roundtrip = x + 0.5

    metrics = compute_rotated_corrected_vae_metrics(
        FakeScaledVaeEncoder(),
        x,
        x_anchor,
        x_anchor,
        x_anchor,
        x,
        x_clean_pred,
        x_clean_oracle,
        x,
        x,
        x,
        x_roundtrip,
    )

    expected_scaled = 8.32 * 0.18215**2
    assert abs(metrics["vae_raw_mse_clean_predcorr"] - 8.32) < 1e-5
    assert abs(metrics["vae_mse_clean_predcorr"] - expected_scaled) < 1e-5
    assert abs(metrics["vae_mse_clean_predcorr"] - 0.276138) < 1e-3


def test_relative_mse_uses_scaled_latent_energy() -> None:
    x = np.ones((2, 2, 1), dtype=np.float32)
    x_anchor = x + 2.0

    metrics = compute_rotated_corrected_vae_metrics(
        FakeScaledVaeEncoder(),
        x,
        x_anchor,
        x_anchor,
        x_anchor,
        x,
        x,
        x,
        x,
        x,
        x,
        x,
    )

    assert abs(metrics["vae_mse_anchor"] - metrics["vae_rel_mse_anchor"] * (0.18215**2)) < 1e-7
    assert abs(metrics["vae_rel_mse_anchor"] - 4.0) < 1e-7
