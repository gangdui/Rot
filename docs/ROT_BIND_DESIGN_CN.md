# RotBind：面向潜空间扩散水印的 VAE 感知旋转同步锚点

## 1. 项目目标

我们希望设计一个用于 latent diffusion watermark 的可插拔旋转同步模块。

原始 latent watermark 方法不被替换。Tree-Ring、RingID、Gaussian Shading 或其他 latent-based diffusion watermark 仍然负责身份标识、来源认证或用户追踪。

我们的模块只是在生成图像之后额外加入一个辅助旋转同步锚点。它的目标是在图像进入 VAE 编码和 DDIM inversion 之前，先估计旋转角度并恢复检测同步关系。

典型 latent diffusion watermark 的检测路径是：

```text
attacked image
-> VAE encoder
-> DDIM inversion
-> recovered noise / latent
-> original watermark detector
```

旋转攻击会破坏这个路径，因为 VAE encoder 不是严格旋转等变的：

```text
E(R_theta x) != R_theta E(x)
```

因此，旋转后的水印图像经过 VAE 编码和 DDIM inversion 后，可能得到一个和原始水印噪声空间差异很大的 latent/noise，导致原始检测器失败。

但是，我们不希望只做普通的像素空间旋转校正。单纯估计角度再把图像旋转回来，在传统鲁棒水印和几何同步中已经很常见，创新性不够。

我们的目标是设计一个 VAE-aware synchronization anchor。这个锚点应该同时满足：

1. 在像素频域中可被同步检测器观测，用于估计旋转角度；
2. 对 VAE latent encoding 的影响尽量小；
3. 在进入 VAE 之前可以通过频域逆调制显式去除；
4. 必要时可以在 VAE latent 层通过 symmetric encoding 进一步抵消锚点 footprint。

简而言之，我们的方法是：

```text
pixel-frequency synchronization
+ pre-VAE inverse anchor removal
+ optional VAE latent cancellation
+ original latent watermark detector
```

## 2. 我们的方法不是什么

本方法不是新的身份水印。

本方法不替代 Tree-Ring、Gaussian Shading、RingID 或其他 latent watermark 方法。

本方法不是普通图像旋转校正。

本方法不是暴力枚举角度搜索。

本方法不是随机像素纹理水印。

本方法是一个专门为 latent diffusion watermark detection 设计的 pre-inversion synchronization adapter。

## 3. 总体思路

给定一个已经由 latent watermark 方法生成的水印图像：

```text
x_w
```

我们加入一个辅助旋转同步锚点：

```text
x_a = AnchorEmbed(x_w, K_a)
```

其中 `K_a` 是锚点密钥。

如果图像受到旋转攻击：

```text
x_att = R_theta(x_a)
```

检测端首先估计旋转角度：

```text
theta_hat = AnchorDetect(x_att, K_a)
```

然后校正图像方向：

```text
x_corr = R_{-theta_hat}(x_att)
```

方向校正后，在进入 VAE 编码之前去除锚点：

```text
x_clean = AnchorRemove(x_corr, K_a)
```

最后再调用原始 latent watermark 检测器：

```text
x_clean
-> VAE encoder
-> DDIM inversion
-> original watermark detector
```

## 4. 锚点设计

锚点设计在像素频域中，具体作用在亮度通道上。

先将 RGB 图像转为 YCbCr，只修改 Y 通道。

设亮度通道的 Fourier 表示为：

```text
F_Y(r, phi) = |F_Y(r, phi)| exp(j phase(r, phi))
```

我们构造一个由密钥控制的角向编码：

```text
P_K(phi)
```

第一版可以使用 Rademacher code：

```text
P_K(phi) in {-1, +1}
```

并做 zero-mean / unit-std normalization。

锚点采用 differential ring-pair magnitude modulation。

选择正向环带：

```text
R+
```

和负向环带：

```text
R-
```

在正向环带中：

```text
|F'_Y(r, phi)| = |F_Y(r, phi)| * (1 + alpha * P_K(phi))
```

在负向环带中：

```text
|F'_Y(r, phi)| = |F_Y(r, phi)| * (1 - alpha * P_K(phi))
```

保持相位不变：

```text
phase'(r, phi) = phase(r, phi)
```

其他频率区域不修改。

等价地：

```text
F'_Y = F_Y * (1 + alpha * M_K)
```

其中：

```text
M_K(r, phi) =
  +P_K(phi), r in R+
  -P_K(phi), r in R-
  0, otherwise
```

最后对修改后的频谱做 inverse FFT，得到新的 Y 通道，再转回 RGB 图像。

## 5. 为什么使用差分环带

如果直接在 raw Fourier magnitude 中检测角向模板：

```text
A(r, phi) = log(1 + |FFT(Y)|)
```

真实图像自身的频谱结构会非常强，容易淹没锚点信号。

因此我们使用正负相邻环带进行差分检测。

检测时提取 Fourier-polar log magnitude：

```text
A(r, phi) = log(1 + |FFT(Y)|)
```

然后构造 ring-difference signature：

```text
D(phi) = mean_{r in R+} A(r, phi) - mean_{r in R-} A(r, phi)
```

由于相邻环带的自然图像频谱趋势相似，做差后可以部分抵消 host image content。

同时，因为我们的锚点在两个环带中以相反符号嵌入，做差会增强锚点：

```text
(+P_K) - (-P_K) = 2 P_K
```

这就是差分环带设计的核心原因。

## 6. 旋转角检测

图像旋转在 Fourier-polar 表示中会变成角向平移。

旋转 theta 后：

```text
D_rot(phi) approximately equals D(phi - theta)
```

因此可以通过 circular correlation 估计旋转角：

```text
C(k) = Corr(D(phi), P_K(phi - k))
```

这里必须使用 FFT-based circular correlation：

```text
C = IFFT(FFT(D) * conj(FFT(P_K)))
```

然后：

```text
theta_hat = argmax_k C(k)
```

不要逐角度旋转模板。

不要对所有角度做 brute-force NCC 搜索。

如果由于实值图像频谱的 Hermitian symmetry，构造的模板是 pi-periodic，那么检测可能只能得到 theta modulo 180 degrees。这种情况下，只允许在两个候选角度中做轻量验证：

```text
theta_mod
theta_mod + 180 degrees
```

不允许重新退化成全角度暴力搜索。

## 7. 锚点去除

方向校正之后，锚点回到 canonical orientation。

因为嵌入时：

```text
F'_Y = F_Y * (1 + alpha * M_K)
```

所以可以在进入 VAE 前通过 inverse modulation 去除锚点：

```text
F_clean = F'_corr / (1 + alpha * M_K)
```

然后 inverse FFT 得到 clean Y 通道。

这一步非常重要，因为同步锚点不应该不必要地扰动 VAE encoder 或原始 latent watermark detector。

这个 inverse modulation 可以理解为 pixel-frequency 层面的 unbinding 操作。

## 8. 可选的 VAE latent cancellation

我们还希望验证锚点在 VAE latent 层是否可以进一步抵消。

给定方向校正后的带锚点图像 `x_plus`，构造一个反锚点图像 `x_minus`：

```text
F_minus = F_plus * (1 - alpha * M_K) / (1 + alpha * M_K)
```

然后分别经过 VAE encoder：

```text
z_plus = E(x_plus)
z_minus = E(x_minus)
```

做 symmetric cancellation：

```text
z_sym = (z_plus + z_minus) / 2
```

如果 VAE encoder 在局部近似线性：

```text
E(x + Delta) approximately E(x) + J_E Delta
E(x - Delta) approximately E(x) - J_E Delta
```

那么：

```text
(E(x + Delta) + E(x - Delta)) / 2 approximately E(x)
```

这一步是我们区别于普通像素空间旋转校正的关键。我们不是只把图像旋转回来，而是进一步研究并抵消锚点在 VAE 编码中的 footprint。

## 9. 第一阶段实现范围

第一阶段不要实现 DDIM inversion。

第一阶段不要实现原始 watermark detector。

第一阶段只实现和评估：

1. anchor embedding；
2. rotation detection；
3. rotation correction；
4. inverse anchor removal；
5. VAE latent footprint；
6. VAE symmetric cancellation。

第一阶段输出指标：

```text
angle error
image quality
VAE latent difference before removal
VAE latent difference after inverse removal
VAE latent difference after symmetric cancellation
runtime
```

我们希望看到的趋势是：

```text
VAE error after symmetric cancellation
<
VAE error after inverse removal
<
VAE error with anchor directly
```

如果这个趋势出现，就说明这个锚点不是普通像素同步模板，而是 VAE-aware / VAE-cancellable synchronization anchor。

## 10. 实现计划

请新建文件夹：

```text
rotbind_anchor/
```

包含：

```text
rotbind_anchor.py
eval_rotbind_anchor.py
vae_hooks.py
metrics.py
plots.py
README_rotbind.md
```

第一阶段不要修改旧的 V0/V1 代码。

不要删除旧实验。

新实现应该尽量独立、清晰、可复现。

## 11. 第一阶段实验数据

使用：

```text
real_watermarked_images_sd15_512
```

作为第一批输入图像目录。

第一轮参数：

```text
alphas: 0.005,0.01,0.02,0.03
angles: 5,10,15,30,45,60,75,90,120,150,180
image size: 512
```

## 12. 成功标准

如果满足以下条件，说明方法值得继续：

```text
mean angle error < 1-2 degrees
failure rate above 3 degrees < 5%
PSNR reasonably high
SSIM reasonably high
VAE symmetric cancellation improves over direct anchoring
```

最重要的 VAE 结果是：

```text
d_symmetric < d_clean < d_anchor
```

其中：

```text
d_anchor = distance(E(x_anchor), E(x_original))
d_clean = distance(E(x_removed), E(x_original))
d_symmetric = distance((E(x_plus)+E(x_minus))/2, E(x_original))
```

如果这个关系成立，就说明我们的锚点不仅是像素空间同步模板，而是一个 VAE-aware synchronization adapter。
