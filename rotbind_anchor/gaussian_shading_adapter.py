"""Adapter boundary for Gaussian Shading inversion and detection.

The bundled Gaussian Shading reference code keeps detector state in the
watermark object created during generation. For externally supplied
watermarked images, this adapter intentionally requires an explicit pipeline
factory instead of guessing missing key/config state.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FakeGaussianShadingPipeline:
    """Small deterministic pipeline used only for tests and dry-run plumbing."""

    detector_threshold: float = 0.25
    score_higher_is_better: bool = True

    def __post_init__(self) -> None:
        self.vae_encoder = _FakeVaeEncoder()

    def invert_to_zT(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32)
        return arr.reshape(-1).astype(np.float32)

    def detect_gaussian_shading(self, zT: np.ndarray | None = None, image: np.ndarray | None = None) -> dict[str, Any]:
        signal = np.asarray(zT if zT is not None else image, dtype=np.float32)
        score = float(np.mean(signal))
        return {
            "detector_score": score,
            "detector_success": bool(score >= self.detector_threshold),
            "detector_threshold": self.detector_threshold,
            "bit_accuracy": score,
            "identification_accuracy": score,
            "score_higher_is_better": self.score_higher_is_better,
        }


class _FakeVaeEncoder:
    """Deterministic VAE-like encoder for tests; not a scientific metric."""

    def encode_images_raw(self, images: list[np.ndarray]) -> np.ndarray:
        return np.stack([np.asarray(img, dtype=np.float32).reshape(-1) for img in images], axis=0)

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        return self.encode_images_raw(images) * 0.18215


def load_gaussian_shading_pipeline(args: Any) -> Any:
    """Load or construct a Gaussian Shading adapter pipeline.

    Supported modes:
    - ``--use-fake-gs-pipeline``: deterministic test pipeline.
    - ``--gs-adapter-module module:function``: user-provided factory returning
      an object with ``invert_to_zT`` and ``detect_gaussian_shading`` methods.

    The local Gaussian Shading reference code does not expose enough persisted
    watermark key/config state for arbitrary image folders, so missing adapter
    configuration raises a clear error instead of silently producing NaNs.
    """
    if bool(getattr(args, "use_fake_gs_pipeline", False)):
        return FakeGaussianShadingPipeline()

    factory_spec = getattr(args, "gs_adapter_module", None)
    if factory_spec:
        pipeline = _load_factory(factory_spec)(args)
        _validate_pipeline(pipeline)
        return pipeline

    raise ValueError(
        "Gaussian Shading adapter requires gs configuration: pass "
        "--gs-adapter-module module:function with access to the original "
        "Gaussian Shading inversion/detector state, or --use-fake-gs-pipeline "
        "for tests. Required state usually includes gs model/config/key or the "
        "watermark object used to generate the input images."
    )


def invert_to_zT(pipeline: Any, image: np.ndarray) -> np.ndarray:
    """Return the detector-space z_T/noise representation from a pipeline."""
    if hasattr(pipeline, "invert_to_zT"):
        return _to_numpy(pipeline.invert_to_zT(image)).astype(np.float32)
    if hasattr(pipeline, "invert_to_zt"):
        return _to_numpy(pipeline.invert_to_zt(image)).astype(np.float32)
    raise NotImplementedError("Gaussian Shading pipeline must expose invert_to_zT(image)")


def detect_gaussian_shading(
    pipeline: Any,
    zT: np.ndarray | None = None,
    image: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run Gaussian Shading detection and normalize the returned metric dict."""
    if hasattr(pipeline, "detect_gaussian_shading"):
        result = pipeline.detect_gaussian_shading(zT=zT, image=image)
    elif hasattr(pipeline, "detect"):
        result = pipeline.detect(zT=zT, image=image)
    else:
        raise NotImplementedError(
            "Gaussian Shading pipeline must expose detect_gaussian_shading(zT=..., image=...)"
        )
    if not isinstance(result, dict):
        raise TypeError("Gaussian Shading detector must return a dict")
    return normalize_detection_result(result, pipeline)


def normalize_detection_result(result: dict[str, Any], pipeline: Any | None = None) -> dict[str, Any]:
    """Fill optional detector fields with NaN/default values."""
    score_higher = bool(result.get("score_higher_is_better", getattr(pipeline, "score_higher_is_better", True)))
    threshold = result.get("detector_threshold", getattr(pipeline, "detector_threshold", float("nan")))
    return {
        "detector_score": float(result.get("detector_score", result.get("score", float("nan")))),
        "detector_success": bool(result.get("detector_success", result.get("success", False))),
        "detector_threshold": float(threshold) if threshold is not None else float("nan"),
        "bit_accuracy": float(result.get("bit_accuracy", float("nan"))),
        "identification_accuracy": float(result.get("identification_accuracy", float("nan"))),
        "score_higher_is_better": score_higher,
    }


def _load_factory(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("--gs-adapter-module must have the form module:function")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, func_name)
    if not callable(factory):
        raise TypeError(f"Gaussian Shading adapter factory is not callable: {spec}")
    return factory


def _validate_pipeline(pipeline: Any) -> None:
    if not (hasattr(pipeline, "invert_to_zT") or hasattr(pipeline, "invert_to_zt")):
        raise NotImplementedError("Gaussian Shading adapter pipeline must expose invert_to_zT(image)")
    if not (hasattr(pipeline, "detect_gaussian_shading") or hasattr(pipeline, "detect")):
        raise NotImplementedError("Gaussian Shading adapter pipeline must expose a detector method")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)
