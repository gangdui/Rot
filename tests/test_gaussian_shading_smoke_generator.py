"""Tests for Gaussian Shading smoke dataset generation helpers."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from rotbind_anchor.gaussian_shading_real_adapter import load_watermark_state
from rotbind_anchor.generate_gaussian_shading_smoke_dataset import (
    build_metadata_row,
    save_watermark_state,
)


class DummyWatermark:
    def __init__(self) -> None:
        self.key = "key_for_image_0"
        self.watermark = "watermark_for_image_0"
        self.tau_onebit = 0.5
        self.tau_bits = 0.75

    def eval_watermark(self, reversed_w):
        return 1.0


def test_save_watermark_state_uses_snapshot_not_later_mutation(tmp_path: Path) -> None:
    watermark = DummyWatermark()
    state_path = tmp_path / "000000_watermark_state.pt"

    save_watermark_state(watermark, state_path)
    watermark.key = "mutated_key_for_next_image"
    loaded = load_watermark_state(state_path, device="cpu")

    assert loaded.key == "key_for_image_0"
    assert loaded.watermark == "watermark_for_image_0"
    assert loaded.tau_onebit == pytest.approx(0.5)
    assert loaded.tau_bits == pytest.approx(0.75)


def test_build_metadata_row_records_generation_and_state_paths(tmp_path: Path) -> None:
    args = Namespace(
        prompt="a high quality photo of a cat",
        model_path="/models/sd15",
        num_inference_steps=50,
        num_inversion_steps=50,
        guidance_scale=7.5,
        channel_copy=1,
        hw_copy=8,
        fpr=0.000001,
        user_number=1000000,
        chacha=True,
    )
    image_path = tmp_path / "images" / "000000.png"
    state_path = tmp_path / "states" / "000000_watermark_state.pt"

    row = build_metadata_row(
        image_id="000000",
        image_path=image_path,
        state_path=state_path,
        seed=123,
        args=args,
    )

    assert row["image_id"] == "000000"
    assert row["image_path"] == str(image_path)
    assert row["state_path"] == str(state_path)
    assert row["prompt"] == args.prompt
    assert row["seed"] == 123
    assert row["model_path"] == args.model_path
    assert row["chacha"] is True
