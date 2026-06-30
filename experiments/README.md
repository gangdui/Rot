# RotBind Experiments

## Directory Layout

- `current/` contains the current primary experiment outputs used for comparison and reporting.
- `archive_debug/` contains historical debugging runs kept for traceability, not for primary reporting.

## Current Main Configuration

The current main configuration is:

```text
method: two_pair
rotation convention: raw corr shift converted to image rotation
alpha: 0.12, 0.15, 0.18
```

`multi_ringpair` is kept as an ablation method for comparison.

`current/vae_footprint_two_pair/` contains the current VAE footprint-only run. It does not include DDIM inversion or any original watermark detector.

## Dataset Hygiene

Synthetic orientation cards and generated test images belong under:

```text
datasets/synthetic_orientation_512/
```

Do not mix synthetic images into:

```text
real_watermarked_images_sd15_512/
```

## Angle Semantics

RotBind uses two angle quantities:

1. `corr_shift_deg`:
   the circular-correlation peak shift in Fourier-polar coordinates. It is
   internal debug information and has the opposite sign of image rotation.

2. `rotation_hat_deg`:
   the estimated image rotation angle:

```text
rotation_hat_deg = (-corr_shift_deg) mod 180
```

`rotation_hat_deg` is the only angle reported in main results. For readable
plots, `rotation_hat_display_deg` is shown in `[-90 deg, 90 deg)`, so `179 deg`
is displayed as `-1 deg`.

## VAE Footprint Metrics

Raw VAE latent mean MSE is scale-dependent and unbounded, because it is measured
directly on `latent_dist.mean`. Main reported VAE MSE fields use Stable
Diffusion scaled latents:

```text
z = scaling_factor * latent_dist.mean
```

The debug `vae_raw_mse_*` fields keep the unscaled latent mean MSE for
traceability. Cosine similarity and `vae_rel_mse_*` should be interpreted
together with the scaled MSE values.

## Rotation Attack Benchmark

Main robustness benchmarks use Tree-Ring-like rotation distortion:

- torchvision rotation
- nearest interpolation for the attack
- `expand=False`
- `fill=0`

The default correction rotation uses torchvision bilinear interpolation with
`fill=0`. Internal diagnostics may additionally report scipy bilinear reflect
rotation (`reshape=False`, `order=1`, `mode="reflect"`), but that is not the
main comparison setting.
