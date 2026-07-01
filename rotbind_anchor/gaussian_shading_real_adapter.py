"""Real Gaussian Shading adapter for RotBind canonical necessity experiments.

This module wraps the bundled Gaussian Shading reference implementation without
changing its detector math. Detection is delegated to the original watermark
object's ``eval_watermark`` method, so a saved watermark state from generation
is required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


class RealGaussianShadingPipeline:
    """Thin wrapper around the original Gaussian Shading inversion/detector."""

    score_higher_is_better = True

    def __init__(
        self,
        sd_pipeline: Any,
        watermark: Any,
        text_embeddings: Any,
        num_inversion_steps: int,
        guidance_scale: float = 1.0,
        device: str = "cpu",
        vae_encoder: Any | None = None,
    ) -> None:
        if not hasattr(sd_pipeline, "get_image_latents") or not hasattr(sd_pipeline, "forward_diffusion"):
            raise TypeError("Gaussian Shading SD pipeline must expose get_image_latents and forward_diffusion")
        if not hasattr(watermark, "eval_watermark"):
            raise TypeError("Gaussian Shading watermark state must expose eval_watermark")
        self.sd_pipeline = sd_pipeline
        self.watermark = watermark
        self.text_embeddings = text_embeddings
        self.num_inversion_steps = int(num_inversion_steps)
        self.guidance_scale = float(guidance_scale)
        self.device = str(device)
        self.detector_threshold = float(getattr(watermark, "tau_onebit", float("nan")))
        self.identification_threshold = float(getattr(watermark, "tau_bits", float("nan")))
        self.vae_encoder = vae_encoder or _GaussianShadingVaeEncoder(sd_pipeline.vae, device=self.device)

    def set_watermark(self, watermark: Any) -> None:
        """Switch only the Gaussian Shading watermark state."""
        if not hasattr(watermark, "eval_watermark"):
            raise TypeError("Gaussian Shading watermark state must expose eval_watermark")
        self.watermark = watermark
        self.detector_threshold = float(getattr(watermark, "tau_onebit", float("nan")))
        self.identification_threshold = float(getattr(watermark, "tau_bits", float("nan")))

    def set_watermark_state(self, state_path: str | Path) -> None:
        """Load and switch to a per-image Gaussian Shading watermark state."""
        watermark = load_watermark_state(Path(state_path), device=self.device)
        self.set_watermark(watermark)

    def invert_to_zT(self, image: np.ndarray) -> np.ndarray:
        """Invert an RGB float image to the z_T representation used by GS detection."""
        import torch

        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("image must have shape [H, W, 3]")
        arr = np.clip(arr, 0.0, 1.0) * 2.0 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        dtype = getattr(self.text_embeddings, "dtype", None)
        if dtype is not None:
            tensor = tensor.to(dtype)
        with torch.no_grad():
            image_latents = self.sd_pipeline.get_image_latents(tensor, sample=False)
            zt = self.sd_pipeline.forward_diffusion(
                latents=image_latents,
                text_embeddings=self.text_embeddings,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inversion_steps,
            )
        return _to_numpy(zt).astype(np.float32)

    def detect_gaussian_shading(
        self,
        zT: np.ndarray | None = None,
        image: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run the original Gaussian Shading watermark detector."""
        if zT is None:
            if image is None:
                raise ValueError("detect_gaussian_shading requires zT or image")
            zT = self.invert_to_zT(image)
        zt_tensor = _to_torch(zT, device=self.device)
        score = float(self.watermark.eval_watermark(zt_tensor))
        success = bool(score >= self.detector_threshold) if np.isfinite(self.detector_threshold) else False
        if np.isfinite(self.identification_threshold):
            identification_accuracy = float(score >= self.identification_threshold)
        else:
            identification_accuracy = float("nan")
        return {
            "detector_score": score,
            "detector_success": success,
            "detector_threshold": self.detector_threshold,
            "bit_accuracy": score,
            "identification_accuracy": identification_accuracy,
            "score_higher_is_better": self.score_higher_is_better,
        }


def build_gaussian_shading_pipeline(args: Any) -> RealGaussianShadingPipeline:
    """Build a real Gaussian Shading inversion/detector adapter.

    Required:
    - ``--gs-watermark-state`` pointing to the saved watermark object/state used
      to generate the input images.
    - ``--gs-model-path`` pointing to the Stable Diffusion model used by the
      Gaussian Shading pipeline.
    """
    state_path = getattr(args, "gs_watermark_state", None)
    if not state_path:
        raise ValueError(
            "Gaussian Shading real adapter requires watermark state: pass "
            "--gs-watermark-state saved during Gaussian Shading image generation."
        )
    model_path = getattr(args, "gs_model_path", None) or getattr(args, "model_path", None)
    if not model_path:
        raise ValueError("Gaussian Shading real adapter requires --gs-model-path")

    device = _resolve_device(getattr(args, "device", "auto"))
    _ensure_gs_code_root(getattr(args, "gs_code_root", "Gaussian-Shading-master"))
    watermark = load_watermark_state(Path(state_path), device=device)
    sd_pipeline = _load_reference_sd_pipeline(args, model_path, device)
    prompt = getattr(args, "gs_prompt", "")
    text_embeddings = sd_pipeline.get_text_embedding(prompt) if hasattr(sd_pipeline, "get_text_embedding") else None
    num_steps = int(
        getattr(args, "gs_num_inversion_steps", None)
        or getattr(args, "num_inversion_steps", None)
        or getattr(args, "num_inference_steps", 50)
    )
    vae_encoder = _GaussianShadingVaeEncoder(sd_pipeline.vae, device=device)
    return RealGaussianShadingPipeline(
        sd_pipeline=sd_pipeline,
        watermark=watermark,
        text_embeddings=text_embeddings,
        num_inversion_steps=num_steps,
        guidance_scale=1.0,
        device=device,
        vae_encoder=vae_encoder,
    )


def load_watermark_state(path: Path, device: str = "cpu") -> Any:
    """Load a saved Gaussian Shading watermark object/state."""
    if not path.exists():
        raise FileNotFoundError(f"Gaussian Shading watermark state not found: {path}")
    import torch

    state = torch.load(path, map_location=device, weights_only=False)
    if hasattr(state, "eval_watermark"):
        return state
    if isinstance(state, dict):
        for key in ("watermark", "watermark_object", "gs_watermark"):
            candidate = state.get(key)
            if hasattr(candidate, "eval_watermark"):
                return candidate
    raise ValueError(
        "Gaussian Shading watermark state must contain the original watermark "
        "object with eval_watermark; PNG images alone are insufficient."
    )


def _load_reference_sd_pipeline(args: Any, model_path: str, device: str) -> Any:
    """Load the original Gaussian Shading InversableStableDiffusionPipeline."""
    _ensure_gs_code_root(getattr(args, "gs_code_root", "Gaussian-Shading-master"))
    _ensure_transformers_clip_feature_extractor_compat()

    import torch
    from diffusers import DPMSolverMultistepScheduler
    from inverse_stable_diffusion import InversableStableDiffusionPipeline

    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        torch_dtype=dtype,
        local_files_only=bool(getattr(args, "vae_local_files_only", False)),
    )
    pipe.safety_checker = None
    return pipe.to(device)


def _ensure_gs_code_root(gs_code_root: str | Path) -> Path:
    gs_root = Path(gs_code_root).resolve()
    if not gs_root.exists():
        raise FileNotFoundError(f"Gaussian Shading code root not found: {gs_root}")
    root_text = str(gs_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return gs_root


def _ensure_transformers_clip_feature_extractor_compat() -> None:
    """Alias CLIPFeatureExtractor for newer transformers versions if needed."""
    import importlib
    import transformers

    if hasattr(transformers, "CLIPFeatureExtractor"):
        return
    clip_cls = getattr(transformers, "CLIPImageProcessor", None)
    if clip_cls is None:
        for module_name, attr_name in (
            ("transformers.models.clip.image_processing_clip", "CLIPImageProcessor"),
            ("transformers.models.clip.feature_extraction_clip", "CLIPFeatureExtractor"),
        ):
            try:
                module = importlib.import_module(module_name)
                clip_cls = getattr(module, attr_name)
                break
            except Exception:
                continue
    if clip_cls is None:
        raise ImportError(
            "Gaussian Shading requires transformers.CLIPFeatureExtractor, but this "
            "environment exposes neither CLIPFeatureExtractor nor CLIPImageProcessor."
        )
    transformers.CLIPFeatureExtractor = clip_cls


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class _GaussianShadingVaeEncoder:
    """Scaled-latent VAE encoder that reuses the Gaussian Shading VAE."""

    def __init__(self, vae: Any, device: str) -> None:
        self.vae = vae
        self.device = device
        self.scaling_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))

    def encode_images_raw(self, images: list[np.ndarray]) -> np.ndarray:
        import torch

        batch = np.stack(images, axis=0).astype(np.float32)
        batch = np.clip(batch, 0.0, 1.0) * 2.0 - 1.0
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        dtype = _module_dtype(self.vae)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        with torch.no_grad():
            latent = self.vae.encode(tensor).latent_dist.mean
        return latent.detach().float().cpu().numpy().reshape(len(images), -1).astype(np.float32)

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        return (self.encode_images_raw(images) * self.scaling_factor).astype(np.float32)


def _module_dtype(module: Any) -> Any:
    if hasattr(module, "parameters"):
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return None
    return None


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _to_torch(value: Any, device: str) -> Any:
    import torch

    if hasattr(value, "detach"):
        return value.to(device)
    return torch.from_numpy(np.asarray(value, dtype=np.float32)).to(device)
