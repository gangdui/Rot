# Deprecated RotBind Runs

This archive contains runs moved out of `experiments/current/` during the
rotation naming and quality-metric cleanup.

Reasons include one or more of:

- old PSNR/SSIM semantics where post-attack cleaned images were compared
  directly to the original image;
- old unscaled raw VAE latent MSE emphasis;
- scipy reflect rotation used as a main run rather than an internal diagnostic;
- directory names using `treering_like`;
- historical alpha sweeps or ablations produced before the current
  rotation/quality conventions.

Keep these files only for debugging provenance. Current reporting should use
the directories listed in `experiments/CURRENT_RUNS.md`.
