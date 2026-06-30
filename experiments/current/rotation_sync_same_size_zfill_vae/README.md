# rotation_sync_same_size_zfill_vae

This is the reserved current directory for the rotation synchronization run
with scaled VAE footprint metrics.

It should be regenerated with the current evaluator when the VAE weights are
available locally or through a working Hugging Face endpoint. Do not copy old
raw-latent VAE outputs into this directory.

Expected configuration:

```text
method = two_pair
alpha = 0.15
attack rotation = torchvision nearest, expand=False, fill=0
correction rotation = torchvision bilinear, fill=0
VAE metrics = scaled latent MSE, relative MSE, cosine
```
