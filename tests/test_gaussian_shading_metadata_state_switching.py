"""Tests for per-image Gaussian Shading watermark state switching."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import rotbind_anchor.eval_gaussian_shading_rotbind_canonical as gs_eval


class MetadataSwitchPipeline:
    detector_threshold = 0.25
    score_higher_is_better = True

    def __init__(self) -> None:
        self.state_paths: list[str] = []
        self.vae_encoder = FlatVaeEncoder()

    def set_watermark_state(self, state_path: str | Path) -> None:
        self.state_paths.append(str(state_path))

    def invert_to_zT(self, image: np.ndarray) -> np.ndarray:
        return np.asarray(image, dtype=np.float32).reshape(-1)

    def detect_gaussian_shading(self, zT=None, image=None):
        score = float(np.mean(np.asarray(zT if zT is not None else image, dtype=np.float32)))
        return {
            "detector_score": score,
            "detector_success": score >= self.detector_threshold,
            "detector_threshold": self.detector_threshold,
            "bit_accuracy": score,
            "identification_accuracy": score,
            "score_higher_is_better": True,
        }


class FlatVaeEncoder:
    def encode_images_raw(self, images: list[np.ndarray]) -> np.ndarray:
        return np.stack([np.asarray(img, dtype=np.float32).reshape(-1) for img in images], axis=0)

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        return self.encode_images_raw(images) * 0.18215


def write_image(path: Path, value: float) -> None:
    arr = np.ones((32, 32, 3), dtype=np.float32) * value
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


def test_metadata_lookup_matches_resolved_path_filename_and_image_id(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "000000.png"
    state_path = tmp_path / "states" / "000000_watermark_state.pt"
    image_path.parent.mkdir()
    state_path.parent.mkdir()
    state_path.write_bytes(b"state")
    rows = [
        {
            "image_id": "sample-id",
            "image_path": str(image_path),
            "state_path": str(state_path),
        }
    ]

    lookup = gs_eval.build_state_lookup(rows)

    assert gs_eval.resolve_state_for_image(image_path, lookup)["state_path"] == str(state_path)
    assert gs_eval.resolve_state_for_image(Path("000000.png"), lookup)["state_path"] == str(state_path)
    assert lookup["sample-id"]["state_path"] == str(state_path)


def test_metadata_state_switching_writes_state_seed_prompt_fields(tmp_path: Path, monkeypatch) -> None:
    image_dir = tmp_path / "images"
    states_dir = tmp_path / "states"
    image_dir.mkdir()
    states_dir.mkdir()
    paths = [image_dir / "000000.png", image_dir / "000001.png"]
    states = [states_dir / "000000_watermark_state.pt", states_dir / "000001_watermark_state.pt"]
    for idx, path in enumerate(paths):
        write_image(path, 0.4 + idx * 0.1)
        states[idx].write_bytes(b"state")
    metadata_path = tmp_path / "metadata.jsonl"
    with metadata_path.open("w") as f:
        for idx, path in enumerate(paths):
            f.write(
                json.dumps(
                    {
                        "image_id": path.stem,
                        "image_path": str(path),
                        "state_path": str(states[idx]),
                        "seed": idx,
                        "prompt": f"prompt {idx}",
                    }
                )
                + "\n"
            )
    pipeline = MetadataSwitchPipeline()
    monkeypatch.setattr(gs_eval, "load_gaussian_shading_pipeline", lambda args: pipeline)
    outdir = tmp_path / "out"

    rc = gs_eval.main(
        [
            "--image-dir",
            str(image_dir),
            "--outdir",
            str(outdir),
            "--size",
            "32",
            "--max-images",
            "2",
            "--gs-metadata",
            str(metadata_path),
            "--gs-adapter-module",
            "mock:factory",
        ]
    )

    assert rc == 0
    assert pipeline.state_paths == [str(states[0]), str(states[1])]
    with (outdir / "rotbind_gs_canonical_results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["gs_state_path"] == str(states[0])
    assert rows[0]["gs_seed"] == "0.0"
    assert rows[0]["gs_prompt"] == "prompt 0"
    assert rows[1]["gs_state_path"] == str(states[1])


def test_metadata_missing_state_path_raises_clear_error(tmp_path: Path, monkeypatch) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "000000.png"
    write_image(image_path, 0.5)
    metadata_path = tmp_path / "metadata.jsonl"
    missing_state = tmp_path / "states" / "missing.pt"
    metadata_path.write_text(
        json.dumps(
            {
                "image_id": "000000",
                "image_path": str(image_path),
                "state_path": str(missing_state),
                "seed": 0,
                "prompt": "prompt",
            }
        )
        + "\n"
    )
    monkeypatch.setattr(gs_eval, "load_gaussian_shading_pipeline", lambda args: MetadataSwitchPipeline())

    with pytest.raises(FileNotFoundError, match="watermark state"):
        gs_eval.main(
            [
                "--image-dir",
                str(image_dir),
                "--outdir",
                str(tmp_path / "out"),
                "--size",
                "32",
                "--gs-metadata",
                str(metadata_path),
                "--gs-adapter-module",
                "mock:factory",
            ]
        )
