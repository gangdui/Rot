# Current RotBind Runs

Current experiment names describe the method and attack definition. Avoid
method-specific directory labels for this benchmark rotation setting.

## Recognized Current Directories

- `current/rotation_sync_same_size_zfill`
  Angle synchronization and matched-geometry PSNR/SSIM under the
  literature-aligned same-size zero-fill rotation attack. No VAE metrics.

- `current/rotation_sync_same_size_zfill_vae`
  Same configuration as `rotation_sync_same_size_zfill`, plus scaled VAE
  latent MSE, relative MSE, and cosine metrics. This directory should be
  regenerated only when the VAE weights are available locally or through a
  working Hugging Face endpoint. At the time of cleanup this directory may
  contain only a README placeholder until the scaled VAE run is regenerated
  with the current evaluator.

- `current/embedding_process_same_size_zfill_alpha015`
  Visualization-only diagnostics for the RotBind embedding process at
  `alpha=0.15`.

- `current/anchor_imperceptibility_two_pair_alpha015`
  Canonical two-pair anchor imperceptibility and removal quality at
  `alpha=0.15`.

- `current/gaussian_shading_canonical_rotbind_alpha015_real10`
  Gaussian Shading canonical RotBind anchor-removal necessity experiment on the
  real 10-image Gaussian Shading smoke dataset. This run has no rotation attack
  and evaluates pixel, scaled VAE latent, Gaussian Shading inversion zT, and
  detector metrics for baseline, anchor, and explicit anchor-removal variants.

- `current/gaussian_shading_quality_rotbind_alpha015_real10`
  Paper-aligned Gaussian Shading + RotBind image quality evaluation on the real
  10-image Gaussian Shading smoke dataset. This run reports RotBind incremental
  paired quality metrics and generation/distribution quality metrics.

## Rotation Attack Definition

The current main attack is a literature-aligned same-size zero-fill rotation
attack:

- torchvision rotation
- attack interpolation = nearest
- `expand=False`
- attack fill = 0
- correction interpolation = bilinear
- correction fill = 0

This follows the common same-size zero-fill rotation setup used in prior
watermark robustness evaluations. This setup is a benchmark rotation definition
rather than a method-specific setting.

## Archived Results

Deprecated runs with old angle/quality/VAE/rotation naming or semantics were
moved to:

```text
archive_debug/deprecated_before_rotation_and_quality_fix/
```

Do not use those archived runs for current tables unless the table explicitly
labels them as historical debugging results.
