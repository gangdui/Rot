# Current RotBind Runs

Current experiment names describe the method and attack definition. Avoid old
Tree-Ring-specific directory labels for this rotation setting.

## Recognized Current Directories

- `current/rotation_sync_same_size_zfill`
  Angle synchronization and matched-geometry PSNR/SSIM under the
  literature-aligned same-size zero-fill rotation attack. No VAE metrics.

- `current/rotation_sync_same_size_zfill_vae`
  Same configuration as `rotation_sync_same_size_zfill`, plus scaled VAE
  latent MSE, relative MSE, and cosine metrics. This directory should be
  regenerated only when the VAE weights are available locally or through a
  working Hugging Face endpoint.

- `current/embedding_process_same_size_zfill_alpha015`
  Visualization-only diagnostics for the RotBind embedding process at
  `alpha=0.15`.

- `current/anchor_imperceptibility_two_pair_alpha015`
  Canonical two-pair anchor imperceptibility and removal quality at
  `alpha=0.15`.

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
watermark robustness evaluations. It is not specific to Tree-Ring.

## Archived Results

Deprecated runs with old angle/quality/VAE/rotation naming or semantics were
moved to:

```text
archive_debug/deprecated_before_rotation_and_quality_fix/
```

Do not use those archived runs for current tables unless the table explicitly
labels them as historical debugging results.
