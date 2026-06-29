# RotBind Experiments

## Directory Layout

- `current/` contains the current primary experiment outputs used for comparison and reporting.
- `archive_debug/` contains historical debugging runs kept for traceability, not for primary reporting.

## Current Main Configuration

The current main configuration is:

```text
method: two_pair
angle_sign: raw
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
