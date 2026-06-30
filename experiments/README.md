# RotBind Experiments

## Directory Layout

- `current/` contains the current primary experiment outputs used for comparison and reporting.
- `archive_debug/` contains historical debugging runs kept for traceability, not for primary reporting.

## Current Main Configuration

The current main configuration is:

```text
method: two_pair
rotation convention: raw corr shift converted to image rotation
primary alpha: 0.15
```

Historical alpha sweeps are kept only in archive/debug directories unless
explicitly regenerated with the current rotation and quality-metric
definitions.

`multi_ringpair` is kept as an ablation method for comparison.

`current/rotation_sync_same_size_zfill_vae/` is the reserved current directory
for scaled VAE footprint metrics under the literature-aligned same-size
zero-fill rotation attack. It does not include DDIM inversion or any original
watermark detector.

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

Main robustness benchmarks use a literature-aligned same-size zero-fill rotation attack:

- torchvision rotation
- attack interpolation = nearest
- `expand=False`
- attack fill = 0
- correction interpolation = bilinear
- correction fill = 0

This follows the common same-size zero-fill rotation setup used in prior
watermark robustness evaluations. This setup is a benchmark rotation definition rather than a method-specific setting. Internal diagnostics may additionally
report scipy bilinear reflect rotation (`reshape=False`, `order=1`,
`mode="reflect"`), but that is not the main comparison setting.

## PSNR/SSIM Quality Metrics

`psnr_anchor` / `ssim_anchor` measure before-attack anchor imperceptibility in
the original image geometry.

`psnr_anchor_rotated` / `ssim_anchor_rotated` measure after-attack
imperceptibility under matched rotated geometry: rotated original image versus
rotated anchored image.

`psnr_clean_*_vs_roundtrip_*` / `ssim_clean_*_vs_roundtrip_*` measure anchor
removal quality under matched geometry after predicted or oracle correction.
These are the preferred removal-quality fields.

`psnr_clean` / `ssim_clean` are legacy post-attack full-image quality aliases
for `psnr_postattack_clean_to_original` / `ssim_postattack_clean_to_original`.
They compare the corrected-and-cleaned image to the original image and are
dominated by rotation roundtrip artifacts, so they should not be used as anchor
quality metrics.
