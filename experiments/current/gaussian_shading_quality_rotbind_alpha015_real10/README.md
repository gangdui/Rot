# Gaussian Shading + RotBind Quality Evaluation

This directory reports paper-aligned image quality metrics for Gaussian Shading
baseline images and RotBind variants.

Metric semantics:

- PSNR/SSIM/LPIPS are RotBind incremental paired quality metrics. They compare
  `x_w` against `x_anchor` and `x_clean`, so they are not a direct full-pipeline
  comparison with external latent watermark methods.
- FID/CLIP are closer to latent watermark paper generation quality metrics.
- If `reference-image-dir` is not provided, `fid_*_vs_reference` is `nan`.
- `fid_anchor_vs_baseline` and `fid_clean_vs_baseline` only measure additional
  RotBind distribution drift relative to the Gaussian Shading baseline. They
  should not be used for direct external-method comparison.
- LPIPS is lower-is-better, CLIP Score is higher-is-better, and FID is
  lower-is-better.
- PSNR can be `inf`; formal reporting should prefer median PSNR or finite mean
  PSNR when exact reconstruction occurs.

Configuration:

```text
alpha = 0.15
method = two_pair
key = 0
clip_model = ViT-g-14
clip_pretrained = checkpoints/open_clip/ViT-g-14_laion2b_s34b_b88k_open_clip_pytorch_model.bin
clip_score_scale = raw cosine-like similarity
reference_image_dir = not provided
compute_lpips = False
compute_fid = False
```
