# Gaussian Shading Canonical RotBind Necessity

Gaussian Shading canonical RotBind anchor-removal necessity experiment.

- no rotation attack
- input images are Gaussian Shading watermarked images
- evaluates whether RotBind anchor removal is necessary before Gaussian Shading inversion/detection
- VAE MSE uses Stable Diffusion scaled latents
- zT metrics compare Gaussian Shading inversion outputs
- detector score direction is recorded by `score_higher_is_better`

Configuration:

```text
alpha = 0.15
method = two_pair
key = 0
score_higher_is_better = True
```
