"""Contract tests for the real Gaussian Shading adapter boundary."""

from __future__ import annotations

from argparse import Namespace
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rotbind_anchor.gaussian_shading_adapter import detect_gaussian_shading, normalize_detection_result
from rotbind_anchor.gaussian_shading_real_adapter import (
    RealGaussianShadingPipeline,
    build_gaussian_shading_pipeline,
    _ensure_transformers_clip_feature_extractor_compat,
)


class MockVae:
    config = SimpleNamespace(scaling_factor=0.18215)

    def encode(self, tensor):
        class Dist:
            mean = tensor

            def mode(self):
                return self.mean

        return SimpleNamespace(latent_dist=Dist())


class MockSdPipeline:
    vae = MockVae()

    def __init__(self) -> None:
        self.forward_calls = 0
        self._execution_device = "cpu"

    def get_image_latents(self, image, sample=False):
        return image.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)

    def forward_diffusion(self, latents, text_embeddings, guidance_scale, num_inference_steps):
        self.forward_calls += 1
        return latents + 0.25


class MockWatermark:
    def __init__(self, tau_onebit: float = 0.5, tau_bits: float = 0.75) -> None:
        self.tau_onebit = tau_onebit
        self.tau_bits = tau_bits
        self.seen_shape = None

    def eval_watermark(self, zt):
        self.seen_shape = tuple(zt.shape)
        return 0.8


def test_build_pipeline_requires_watermark_state_or_object() -> None:
    args = Namespace(gs_watermark_state=None, gs_model_path=None, gs_config=None, gs_key=None)

    with pytest.raises(ValueError, match="watermark state"):
        build_gaussian_shading_pipeline(args)


def test_real_pipeline_exposes_inversion_detection_and_vae_encoder() -> None:
    sd_pipe = MockSdPipeline()
    watermark = MockWatermark()
    pipeline = RealGaussianShadingPipeline(
        sd_pipeline=sd_pipe,
        watermark=watermark,
        text_embeddings=None,
        num_inversion_steps=3,
        guidance_scale=1.0,
        device="cpu",
    )

    image = np.ones((8, 8, 3), dtype=np.float32) * 0.75
    zt = pipeline.invert_to_zT(image)
    det = pipeline.detect_gaussian_shading(zT=zt)

    assert hasattr(pipeline, "vae_encoder")
    assert sd_pipe.forward_calls == 1
    assert zt.shape == (1, 4, 8, 8)
    assert watermark.seen_shape == (1, 4, 8, 8)
    assert det["detector_score"] == pytest.approx(0.8)
    assert det["detector_success"] is True
    assert det["detector_threshold"] == pytest.approx(0.5)
    assert det["bit_accuracy"] == pytest.approx(0.8)
    assert det["identification_accuracy"] == pytest.approx(1.0)
    assert det["score_higher_is_better"] is True


def test_detect_result_normalizes_through_common_adapter() -> None:
    pipeline = RealGaussianShadingPipeline(
        sd_pipeline=MockSdPipeline(),
        watermark=MockWatermark(),
        text_embeddings=None,
        num_inversion_steps=1,
        guidance_scale=1.0,
        device="cpu",
    )

    result = detect_gaussian_shading(pipeline, zT=np.ones((1, 4, 4, 4), dtype=np.float32))

    normalized = normalize_detection_result(result, pipeline)
    assert set(normalized) >= {
        "detector_score",
        "detector_success",
        "detector_threshold",
        "bit_accuracy",
        "identification_accuracy",
        "score_higher_is_better",
    }


def test_real_pipeline_set_watermark_updates_thresholds() -> None:
    pipeline = RealGaussianShadingPipeline(
        sd_pipeline=MockSdPipeline(),
        watermark=MockWatermark(tau_onebit=0.5, tau_bits=0.75),
        text_embeddings=None,
        num_inversion_steps=1,
        guidance_scale=1.0,
        device="cpu",
    )

    pipeline.set_watermark(MockWatermark(tau_onebit=0.9, tau_bits=0.95))

    assert pipeline.detector_threshold == pytest.approx(0.9)
    assert pipeline.identification_threshold == pytest.approx(0.95)


def test_transformers_clip_feature_extractor_compat_aliases_top_level(monkeypatch) -> None:
    class FakeImageProcessor:
        pass

    fake_transformers = SimpleNamespace(CLIPImageProcessor=FakeImageProcessor)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    _ensure_transformers_clip_feature_extractor_compat()

    assert fake_transformers.CLIPFeatureExtractor is FakeImageProcessor


def test_transformers_clip_feature_extractor_compat_aliases_clip_submodule(monkeypatch) -> None:
    class FakeImageProcessor:
        pass

    fake_transformers = SimpleNamespace()
    fake_clip_module = SimpleNamespace(CLIPImageProcessor=FakeImageProcessor)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.models.clip.image_processing_clip", fake_clip_module)

    _ensure_transformers_clip_feature_extractor_compat()

    assert fake_transformers.CLIPFeatureExtractor is FakeImageProcessor
