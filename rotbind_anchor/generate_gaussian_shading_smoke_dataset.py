"""Generate a tiny Gaussian Shading dataset with per-image watermark state.

The generated images are intended for RotBind canonical necessity sanity
checks. Each image gets its own saved watermark state because the Gaussian
Shading reference code refreshes key/watermark state on every
``create_watermark_and_return_w`` call.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="datasets/gaussian_shading_sd15_512_smoke")
    parser.add_argument("--gs-code-root", default="Gaussian-Shading-master")
    parser.add_argument(
        "--model-path",
        default="/home/ysh/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
    )
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--prompt", default="a high quality photo of a cat")
    parser.add_argument("--image-length", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--num-inversion-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--gen-seed", type=int, default=0)
    parser.add_argument("--channel-copy", type=int, default=1)
    parser.add_argument("--hw-copy", type=int, default=8)
    parser.add_argument("--fpr", type=float, default=0.000001)
    parser.add_argument("--user-number", type=int, default=1000000)
    parser.add_argument("--chacha", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-failed-baseline", action="store_true")
    args = parser.parse_args(argv)
    if args.num_images <= 0:
        parser.error("--num-images must be positive")
    if args.image_length <= 0:
        parser.error("--image-length must be positive")
    return args


def save_watermark_state(watermark: Any, state_path: Path) -> None:
    """Save a per-image snapshot of the current Gaussian Shading watermark object."""
    import torch

    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"watermark": copy.deepcopy(watermark)}, state_path)


def build_metadata_row(
    image_id: str,
    image_path: Path,
    state_path: Path,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return one JSONL metadata row."""
    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "state_path": str(state_path),
        "prompt": args.prompt,
        "seed": int(seed),
        "model_path": args.model_path,
        "num_inference_steps": int(args.num_inference_steps),
        "num_inversion_steps": int(args.num_inversion_steps),
        "guidance_scale": float(args.guidance_scale),
        "channel_copy": int(args.channel_copy),
        "hw_copy": int(args.hw_copy),
        "fpr": float(args.fpr),
        "user_number": int(args.user_number),
        "chacha": bool(args.chacha),
    }


def main(argv: list[str] | None = None) -> int:
    """Generate the smoke dataset."""
    args = parse_args(argv)
    outdir = Path(args.outdir)
    images_dir = outdir / "images"
    states_dir = outdir / "states"
    images_dir.mkdir(parents=True, exist_ok=True)
    states_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = outdir / "metadata.jsonl"

    gs_modules = import_gaussian_shading_modules(args.gs_code_root)
    device = resolve_device(args.device)
    pipe = load_gaussian_shading_pipe(args, gs_modules, device)
    watermark = build_watermark(args, gs_modules)
    text_embeddings = pipe.get_text_embedding("")

    rows: list[dict[str, Any]] = []
    for index in range(int(args.num_images)):
        image_id = f"{index:06d}"
        seed = int(args.gen_seed) + index
        image_path = images_dir / f"{image_id}.png"
        state_path = states_dir / f"{image_id}_watermark_state.pt"

        gs_modules["set_random_seed"](seed)
        init_latents_w = watermark.create_watermark_and_return_w()
        outputs = pipe(
            args.prompt,
            num_images_per_prompt=1,
            guidance_scale=float(args.guidance_scale),
            num_inference_steps=int(args.num_inference_steps),
            height=int(args.image_length),
            width=int(args.image_length),
            latents=init_latents_w,
        )
        image_w = outputs.images[0]
        image_w.save(image_path)
        save_watermark_state(watermark, state_path)

        baseline_score = run_baseline_sanity_check(
            pipe=pipe,
            watermark=watermark,
            image=image_w,
            text_embeddings=text_embeddings,
            num_inversion_steps=int(args.num_inversion_steps),
            device=device,
            transform_img=gs_modules["transform_img"],
        )
        tau = float(getattr(watermark, "tau_onebit", float("nan")))
        print(f"[{image_id}] baseline_score = {baseline_score:.6f}, tau_onebit = {tau:.6f}")
        if baseline_score < tau and not bool(args.allow_failed_baseline):
            raise RuntimeError(
                f"Gaussian Shading baseline detection failed for {image_id}: "
                f"score={baseline_score:.6f}, tau_onebit={tau:.6f}"
            )

        rows.append(build_metadata_row(image_id, image_path, state_path, seed, args))

    with metadata_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"images = {images_dir}")
    print(f"states = {states_dir}")
    print(f"metadata = {metadata_path}")
    return 0


def import_gaussian_shading_modules(gs_code_root: str) -> dict[str, Any]:
    """Import the Gaussian Shading reference modules."""
    gs_root = Path(gs_code_root).resolve()
    if not gs_root.exists():
        raise FileNotFoundError(f"Gaussian Shading code root not found: {gs_root}")
    root_text = str(gs_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    from diffusers import DPMSolverMultistepScheduler
    from inverse_stable_diffusion import InversableStableDiffusionPipeline
    from image_utils import set_random_seed, transform_img
    from watermark import Gaussian_Shading, Gaussian_Shading_chacha

    return {
        "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
        "InversableStableDiffusionPipeline": InversableStableDiffusionPipeline,
        "Gaussian_Shading": Gaussian_Shading,
        "Gaussian_Shading_chacha": Gaussian_Shading_chacha,
        "set_random_seed": set_random_seed,
        "transform_img": transform_img,
    }


def resolve_device(device: str) -> str:
    """Resolve an auto/cuda/cpu device string."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_gaussian_shading_pipe(args: argparse.Namespace, gs_modules: dict[str, Any], device: str) -> Any:
    """Load the Gaussian Shading reference diffusion pipeline."""
    import torch

    scheduler_cls = gs_modules["DPMSolverMultistepScheduler"]
    pipe_cls = gs_modules["InversableStableDiffusionPipeline"]
    scheduler = scheduler_cls.from_pretrained(
        args.model_path,
        subfolder="scheduler",
        local_files_only=bool(args.local_files_only),
    )
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = pipe_cls.from_pretrained(
        args.model_path,
        scheduler=scheduler,
        torch_dtype=dtype,
        local_files_only=bool(args.local_files_only),
    )
    pipe.safety_checker = None
    return pipe.to(device)


def build_watermark(args: argparse.Namespace, gs_modules: dict[str, Any]) -> Any:
    """Construct the original Gaussian Shading watermark object."""
    cls_name = "Gaussian_Shading_chacha" if bool(args.chacha) else "Gaussian_Shading"
    cls = gs_modules[cls_name]
    return cls(int(args.channel_copy), int(args.hw_copy), float(args.fpr), int(args.user_number))


def run_baseline_sanity_check(
    pipe: Any,
    watermark: Any,
    image: Any,
    text_embeddings: Any,
    num_inversion_steps: int,
    device: str,
    transform_img: Any,
) -> float:
    """Invert the generated image and run the original GS detector once."""
    with _torch_no_grad():
        image_tensor = transform_img(image).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents = pipe.get_image_latents(image_tensor, sample=False)
        reversed_latents = pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=int(num_inversion_steps),
        )
        return float(watermark.eval_watermark(reversed_latents))


class _torch_no_grad:
    """Tiny lazy torch.no_grad context to keep module import lightweight."""

    def __enter__(self):
        import torch

        self._ctx = torch.no_grad()
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)


if __name__ == "__main__":
    raise SystemExit(main())
