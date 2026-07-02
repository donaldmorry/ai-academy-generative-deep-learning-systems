# Section 5.6: PixelCNN

> **Source inheritance:** Foster, Ch. 5 — "PixelCNN"  
> **Enhanced with:** Masked convolutions, raster pixel ordering, causal image generation, and Fashion-MNIST quantization  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

[Autoregressive models](../../GLOSSARY.md#autoregressive-model) factor the joint distribution of data into a product of conditionals — one step at a time. For text, LSTMs predict the next token; for images, **PixelCNN** (van den Oord et al., 2016) predicts the next **pixel** given all pixels that came before it in a fixed raster order.

The challenge: standard `Conv2D` layers see neighbors in all directions — they violate causality. **Masked convolutions** zero out filter weights that would peek at future pixels, turning CNNs into causal predictors. The result is a tractable likelihood model over images: train with maximum likelihood, sample one pixel at a time.

> **Readable form:** Read the image like a book — top-left to bottom-right — and at each position predict the next pixel from everything already written.

---

## From Text Tokens to Pixel Tokens

In [Section 5.1](./section-01-autoregressive-framework.md), the chain rule gave us:

$$
P(x) = \prod_{i=1}^{n} P(x_i \mid x_{<i})
$$
> **Readable form:** Joint probability equals the product of each pixel's probability given all earlier pixels.

For a $H \times W$ grayscale image flattened in row-major order, $n = H \times W$. Each $x_i$ is a discrete pixel level (or a distribution over levels). Unlike words, pixel values live on a continuous color spectrum — but we **quantize** them into bins so softmax outputs remain tractable.

| Text (LSTM) | Images (PixelCNN) |
|-------------|-------------------|
| Token = word/character | Token = pixel (or channel value) |
| Order = left-to-right sequence | Order = raster scan (rows, then columns) |
| Embedding layer | Raw pixel values as input channels |
| Sequential RNN | Parallel masked convolutions |

Foster quantizes Fashion-MNIST to **4 levels** per pixel so the output layer needs only 4 softmax logits instead of 256 — a practical trade-off between fidelity and training speed.

---

## Raster Ordering and RGB Channels

A sensible pixel order: scan rows top-to-bottom, columns left-to-right within each row. For RGB images, van den Oord et al. also order **channels** within each spatial position — e.g., R before G before B — so the model predicts R, then G given R, then B given R and G.

```
Row 0: (0,0) → (0,1) → ... → (0,W-1)
Row 1: (1,0) → ...
...
```

At position $(r, c)$, the model may use all pixels with $r' < r$, or $r' = r$ and $c' < c$, plus earlier channels at $(r, c)$. This ordering is arbitrary but must stay **fixed** at train and sample time.

---

## Masked Convolutional Layers

A standard $3 \times 3$ conv at pixel $(r, c)$ aggregates a neighborhood centered on $(r, c)$ — including **future** pixels. Masked conv fixes this by element-wise multiplying the filter kernel with a binary mask before each forward pass.

**Type A mask** — used on the **first** layer applied directly to the image:
- Mask out the center pixel (the prediction target)
- Mask out all pixels at or after the current position in raster order

**Type B mask** — used in **deeper** layers:
- Include the center pixel (it was computed only from past pixels in earlier layers)
- Still mask all strictly future spatial positions

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class MaskedConv2D(layers.Layer):
  def __init__(self, mask_type, **kwargs):
    super().__init__()
    self.mask_type = mask_type
    self.conv = layers.Conv2D(**kwargs)

  def build(self, input_shape):
    self.conv.build(input_shape)
    k = self.conv.kernel
    kh, kw = k.shape[0], k.shape[1]
    mask = np.zeros(k.shape)
    mask[: kh // 2, ...] = 1.0
    mask[kh // 2, : kw // 2, ...] = 1.0
    if self.mask_type == "B":
      mask[kh // 2, kw // 2, ...] = 1.0
    self.mask = tf.constant(mask, dtype=k.dtype)

  def call(self, inputs):
    self.conv.kernel.assign(self.conv.kernel * self.mask)
    return self.conv(inputs)
```

The first layer often uses a larger kernel (e.g., $7 \times 7$ Type A) to give each pixel broad context from the past. Deeper layers use $3 \times 3$ Type B masks.

---

## Full PixelCNN Stack (Simplified)

Foster's architecture: Type A masked conv → residual blocks → Type B $1 \times 1$ masked convs → softmax over 4 pixel levels.

```python
from tensorflow.keras import models, optimizers

def build_pixelcnn(input_shape=(16, 16, 1), n_residual=5, n_filters=128):
  inputs = layers.Input(shape=input_shape)
  x = MaskedConv2D(
    mask_type="A", filters=n_filters, kernel_size=7,
    activation="relu", padding="same"
  )(inputs)
  for _ in range(n_residual):
    x = ResidualBlock(filters=n_filters)(x)  # see Section 5.7
  for _ in range(2):
    x = MaskedConv2D(
      mask_type="B", filters=n_filters, kernel_size=1,
      activation="relu", padding="valid"
    )(x)
  out = layers.Conv2D(
    filters=4, kernel_size=1, activation="softmax", padding="valid"
  )(x)
  return models.Model(inputs, out)

pixel_cnn = build_pixelcnn()
pixel_cnn.compile(
  optimizer=optimizers.Adam(learning_rate=5e-4),
  loss="sparse_categorical_crossentropy",
)
```

**Training targets:** input images scaled to $[0, 1]$ (float); labels are the same images quantized to integers $\{0, 1, 2, 3\}$. The network outputs a $4$-way distribution at every spatial location — like an autoencoder that reconstructs its input, but causally masked so no pixel sees itself or the future.

---

## Causal Generation Loop

Sampling is **sequential** — the main weakness of PixelCNN. For a $16 \times 16$ image you need $256$ forward passes (one per pixel), each reading the partially filled canvas.

```python
def sample_pixel(pixel_cnn, temperature=1.0, batch_size=1):
  shape = (batch_size,) + pixel_cnn.input_shape[1:]
  img = np.zeros(shape, dtype=np.float32)
  rows, cols, channels = img.shape[1], img.shape[2], img.shape[3]
  for row in range(rows):
    for col in range(cols):
      for ch in range(channels):
        probs = pixel_cnn.predict(img, verbose=0)[:, row, col, :]
        probs = probs ** (1.0 / temperature)
        probs /= probs.sum(axis=-1, keepdims=True)
        levels = [np.random.choice(4, p=p) for p in probs]
        img[:, row, col, ch] = np.array(levels, dtype=np.float32) / 4.0
  return img
```

Compare to a VAE: **one** decoder forward pass produces the full image. PixelCNN trades speed for exact likelihood and sharp local predictions. Modern systems (Part III Transformers, latent autoregressive models) mitigate this cost — but the masked-conv principle remains foundational.

---

## PixelCNN vs Other Generative Families

| Property | PixelCNN | VAE | GAN |
|----------|----------|-----|-----|
| Likelihood | Exact (tractable) | ELBO bound | Implicit |
| Sampling speed | Slow (sequential) | Fast | Fast |
| Output sharpness | Good locally | Often blurry | Often sharpest |
| Architecture | Masked CNN | Encoder-decoder | Generator only |

PixelCNN sits in the **explicit density, slow sampling** quadrant of Foster's taxonomy ([Chapter 01](../chapter-01-generative-modeling/section-07-generative-model-taxonomy.md)). [Section 5.8](./section-08-mixture-distributions-for-pixelcnn.md) upgrades the output head with mixture distributions so we need not quantize to 4 levels.

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Wrong mask type on first layer | Future pixels leak into predictions | Type A on input; Type B deeper |
| Forgetting channel order (RGB) | Color fringing at generation | Fix R→G→B order in sampling loop |
| 256-way softmax on full MNIST | Extremely slow training | Quantize or use mixture output |
| Mismatched train/sample order | Blurry or incoherent images | Same raster order everywhere |

---

## Connection to Prior Sections

| Concept | Where | Role here |
|---------|-------|-----------|
| Chain rule / MLE | [Section 5.1](./section-01-autoregressive-framework.md) | Defines training objective |
| Conv2D | [Chapter 02](../chapter-02-deep-learning/section-06-convolutional-layers.md) | Base layer before masking |
| Residual blocks | [Section 5.7](./section-07-residual-blocks-and-training-the-pixelcnn.md) | Depth without vanishing gradients |
| Fashion-MNIST prep | [Chapter 03](../chapter-03-variational-autoencoders/section-01-autoencoder-architecture.md) | Same dataset, different generator |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **PixelCNN** | Autoregressive image model using masked convolutions |
| **Masked convolution** | Conv filter with zeroed weights to enforce causality |
| **Type A / B mask** | Excludes / includes center pixel in the receptive field |
| **Raster order** | Fixed top-left-to-bottom-right pixel sequence |
| **Quantization** | Bucketing continuous pixels into discrete levels for softmax |
| **Causal generation** | Sampling each token only from past context |

---

## Reflection Questions

1. Why cannot a standard Conv2D stack be trained autoregressively without masking?
2. What is the computational cost of generating a $32 \times 32$ image vs a VAE sample?
3. Why does Foster use 4 pixel levels instead of 256 for the softmax output?
4. How does Type B differ from Type A, and why is the distinction needed?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). O'Reilly. Ch. 5 — PixelCNN. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- van den Oord, A. et al. (2016). Conditional Image Generation with PixelCNN Decoders. [https://arxiv.org/abs/1606.05328](https://arxiv.org/abs/1606.05328)
- Foster's notebook: `notebooks/05_autoregressive/02_pixelcnn/pixelcnn.ipynb`
- ADMoreau Keras PixelCNN tutorial (adapted in Foster's repo)

---

**Previous:** [Section 5.5 — RNN Extensions](./section-05-rnn-extensions.md)  
**Next:** [Section 5.7 — Residual Blocks & Training](./section-07-residual-blocks-and-training-the-pixelcnn.md)



