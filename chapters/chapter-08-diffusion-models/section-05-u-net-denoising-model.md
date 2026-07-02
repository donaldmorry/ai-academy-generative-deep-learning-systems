# Section 8.5: U-Net Denoising Model

> **Source inheritance:** Foster, Ch. 8 — "The U-Net Denoising Model"  
> **Enhanced with:** Encoder-decoder skips, sinusoidal time embedding, DownBlock/UpBlock, ResidualBlock  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

The [reverse diffusion](./section-04-reverse-diffusion-process.md) network $\epsilon_\theta(x_t, t)$ must map a **noisy image** to a **same-shaped noise tensor**. Foster uses a **U-Net** — encoder-decoder with **skip connections** linking same-resolution layers. Unlike a [VAE](../chapter-03-variational-autoencoders/section-01-autoencoder-architecture.md), skips preserve spatial detail needed to predict pixel-level noise.

**Time embedding** injects the noise level: sinusoidal encoding of the variance scalar, upsampled and concatenated with image channels so the network knows *how much* static to remove.

> **Readable form:** Hourglass CNN with teleporter cables across the middle — plus a clock telling each layer how noisy the input is.

---

## U-Net Topology

```
Input x_t (64×64×3) + time embedding
    ↓ DownBlock ×3  (save skips)
    ↓ ResidualBlock (bottleneck)
    ↑ UpBlock ×3    (concat skips)
    ↓ Conv 1×1 → noise prediction (64×64×3)
```

| Half | Operation | Channels | Spatial |
|------|-----------|----------|---------|
| Down | Conv stride-2 + residual blocks | Increase | Halve |
| Bottleneck | ResidualBlock | 128 | 8×8 |
| Up | Upsample + concat skip + residual | Decrease | Double |

Output shape **matches** input — essential for $\epsilon$-prediction.

---

## Sinusoidal Time Embedding

Same idea as Transformers' positional encoding — map scalar noise level to high-D vector:

```python
import math
import tensorflow as tf
from tensorflow.keras import layers

def sinusoidal_embedding(x):
  frequencies = tf.exp(
    tf.linspace(tf.math.log(1.0), tf.math.log(1000.0), 64)
  )
  angles = frequencies * x * 2.0 * math.pi
  return tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)

# In U-Net:
noise_variances = layers.Input(shape=(1, 1, 1))  # noise_rates**2
noise_embedding = layers.Lambda(sinusoidal_embedding)(noise_variances)
noise_embedding = layers.UpSampling2D(size=64, interpolation="nearest")(noise_embedding)
x = layers.Concatenate()([image_features, noise_embedding])
```

Broadcast embedding across all spatial positions — every pixel "knows" the global noise level.

---

## DownBlock and UpBlock

```python
class DownBlock(layers.Layer):
  def __init__(self, filters, block_depth=2):
    super().__init__()
    self.res_blocks = [ResidualBlock(filters) for _ in range(block_depth)]
    self.pool = layers.Conv2D(filters, 3, strides=2, padding="same")

  def call(self, inputs, skips):
    x = inputs
    for block in self.res_blocks:
      x = block(x)
    skips.append(x)
    return self.pool(x), skips


class UpBlock(layers.Layer):
  def __init__(self, filters, block_depth=2):
    super().__init__()
    self.upsample = layers.UpSampling2D(2)
    self.res_blocks = [ResidualBlock(filters) for _ in range(block_depth)]

  def call(self, inputs):
    x, skips = inputs
    x = self.upsample(x)
    x = layers.Concatenate()([x, skips.pop()])
    for block in self.res_blocks:
      x = block(x)
    return x, skips
```

**Skip connections** concatenate encoder features before each upsample — preserves edges, petal boundaries, color gradients that bottleneck alone would blur.

---

## ResidualBlock

```python
class ResidualBlock(layers.Layer):
  def __init__(self, filters):
    super().__init__()
    self.conv1 = layers.Conv2D(filters, 3, padding="same", activation="swish")
    self.conv2 = layers.Conv2D(filters, 3, padding="same")

  def call(self, x):
    h = self.conv1(x)
    h = self.conv2(h)
    if x.shape[-1] != filters:
      x = layers.Conv2D(filters, 1)(x)
    return layers.Activation("swish")(x + h)
```

**Swish** activation (from [EBM Chapter](../chapter-07-energy-based-models/section-02-mnist-energy-function.md)) — smooth, works well in deep generative nets.

---

## Full U-Net Assembly (Foster)

```python
from tensorflow.keras import Model

noisy_images = layers.Input(shape=(64, 64, 3))
x = layers.Conv2D(32, kernel_size=1)(noisy_images)

noise_variances = layers.Input(shape=(1, 1, 1))
noise_embedding = layers.Lambda(sinusoidal_embedding)(noise_variances)
noise_embedding = layers.UpSampling2D(size=64, interpolation="nearest")(noise_embedding)
x = layers.Concatenate()([x, noise_embedding])

skips = []
x, skips = DownBlock(32, block_depth=2)([x, skips])
x, skips = DownBlock(64, block_depth=2)([x, skips])
x, skips = DownBlock(96, block_depth=2)([x, skips])

x = ResidualBlock(128)(x)

x, skips = UpBlock(96, block_depth=2)([x, skips])
x, skips = UpBlock(64, block_depth=2)([x, skips])
x, skips = UpBlock(32, block_depth=2)([x, skips])

x = layers.Conv2D(3, kernel_size=1)(x)
unet = Model(inputs=[noisy_images, noise_variances], outputs=x)
```

Two inputs: noisy RGB image + scalar noise variance. Output: predicted noise $\hat{\epsilon}$.

---

## U-Net vs VAE Autoencoder

| | VAE | DDPM U-Net |
|---|-----|------------|
| Skip connections | Usually none | **Yes** — core design |
| Latent | Low-D vector | None — full res |
| Conditioning | $z$ vector | Time / noise embedding |
| Output | Reconstructed image | Noise tensor |
| Stochastic layers | Often yes | No |

U-Net was pioneered for biomedical segmentation (Ronneberger et al., 2015) — diffusion repurposed it for generative denoising.

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Skip channel mismatch | Concat error | Match filters in UpBlock |
| Wrong embedding size | 64 vs 32 mismatch | Upsample to H×W |
| Missing noise input | Same output all $t$ | Concatenate variance |
| ReLU only | Dead neurons | Use swish |

---

## Connection to Other Sections

| Concept | Link |
|---------|------|
| Denoise math | [Section 8.4](./section-04-reverse-diffusion-process.md) |
| Training wrapper | [Section 8.6](./section-06-training-the-diffusion-model.md) |
| Conv2DTranspose in VAE | [Section 3.1](../chapter-03-variational-autoencoders/section-01-autoencoder-architecture.md) |
| Residual blocks | [Section 5.7](../chapter-05-autoregressive-models/section-07-residual-blocks-and-training-the-pixelcnn.md) |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **U-Net** | Encoder-decoder CNN with skip connections |
| **Skip connection** | Concatenate encoder features into decoder |
| **Sinusoidal embedding** | Fourier features for scalar timestep |
| **DownBlock / UpBlock** | Strided down / upsample up chapters |
| **Noise variance input** | Tells network the corruption level |

---

## Reflection Questions

1. Why must the U-Net output match input spatial shape?
2. How do skip connections help predict fine petal edges on flowers?
3. Why embed noise level instead of passing raw scalar $t$ only once?
4. What breaks if you use a plain autoencoder without skips?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 8 — U-Net. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Ronneberger, O. et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
- Foster's notebook: `notebooks/08_diffusion/01_ddm/ddm.ipynb`

---

**Previous:** [Section 8.4 — Reverse Diffusion Process](./section-04-reverse-diffusion-process.md)  
**Next:** [Section 8.6 — Training the Diffusion Model](./section-06-training-the-diffusion-model.md)
