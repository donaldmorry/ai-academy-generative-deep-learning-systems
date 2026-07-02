# Section 5.7: Residual Blocks & Training the PixelCNN

> **Source inheritance:** Foster, Ch. 5 — "Residual Blocks" / "Training the PixelCNN"  
> **Enhanced with:** Skip connections in causal CNNs, Fashion-MNIST quantization, training loop, and generation callbacks  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Deep [PixelCNN](./section-06-pixelcnn.md) networks need many masked convolution layers to capture long-range image structure — sleeves relate to shirt bodies, horizons align across pixels. Stacking plain conv layers risks **vanishing gradients** and makes identity mappings hard to learn. **Residual blocks** add a skip connection: output = transformed input + input.

Foster stacks five residual blocks between the initial Type A mask and the final softmax head. This section implements `ResidualBlock`, prepares quantized Fashion-MNIST data, trains the full PixelCNN, and monitors generation with a Keras callback.

> **Readable form:** Each block may pass the input straight through the shortcut; the conv path only needs to learn the *residual* change.

---

## Residual Block Architecture

A PixelCNN residual block (Figure 5-14 in Foster):

1. **$1 \times 1$ Conv2D** — halve channels (bottleneck)
2. **Type B MaskedConv2D** ($3 \times 3$) — causal spatial mixing
3. **$1 \times 1$ Conv2D** — restore channel count
4. **Add** input to output (skip connection)

```python
import tensorflow as tf
from tensorflow.keras import layers

class ResidualBlock(layers.Layer):
  def __init__(self, filters, **kwargs):
    super().__init__(**kwargs)
    self.conv1 = layers.Conv2D(filters // 2, 1, activation="relu")
    self.pixel_conv = MaskedConv2D(
      mask_type="B", filters=filters // 2,
      kernel_size=3, activation="relu", padding="same"
    )
    self.conv2 = layers.Conv2D(filters, 1, activation="relu")

  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.pixel_conv(x)
    x = self.conv2(x)
    return layers.add([inputs, x])
```

The Type B masked conv in the middle sees only **five** past pixels for a $3 \times 3$ kernel: three in the row above, one to the left, and the center (already computed from past context in earlier layers). Stacking five blocks expands the effective receptive field without breaking causality.

---

## Why Skip Connections Help

Without the shortcut, the network must approximate the identity $f(x) = x$ through stacked nonlinear layers — difficult when the optimal transformation is "mostly keep the input." With residuals:

$$
y = x + \mathcal{F}(x)
$$
> **Readable form:** Output equals input plus a learned residual correction.

If $\mathcal{F}(x) \to 0$, the block passes $x$ through unchanged. Gradients flow directly along the skip path during backpropagation. This pattern — introduced in ResNet for classification — transfers cleanly to generative masked CNNs.

---

## Data Preparation: Quantized Fashion-MNIST

Foster trains on $16 \times 16$ grayscale Fashion-MNIST with 4 discrete levels per pixel.

```python
import numpy as np
from tensorflow.keras import datasets

(x_train, _), (x_test, _) = datasets.fashion_mnist.load_data()

def preprocess(imgs, size=16, levels=4):
  imgs = imgs.astype("float32") / 255.0
  # center-crop or resize to 16x16 (Foster uses 16x16 for speed)
  imgs = tf.image.resize(imgs[..., None], (size, size)).numpy()
  input_data = imgs  # floats in [0, 1]
  output_data = np.floor(imgs * levels).astype("int32")
  output_data = np.clip(output_data, 0, levels - 1)
  return input_data, output_data

x_in, y_out = preprocess(x_train)
x_test_in, y_test_out = preprocess(x_test)
```

**Input** (`input_data`): float pixels in $[0, 1]$. **Target** (`output_data`): integer levels $\{0, 1, 2, 3\}$ for `sparse_categorical_crossentropy`. The model predicts a distribution over 4 bins at every spatial location.

---

## Assembling and Training the Full Model

```python
from tensorflow.keras import models, optimizers, callbacks

inputs = layers.Input(shape=(16, 16, 1))
x = MaskedConv2D(
  mask_type="A", filters=128, kernel_size=7,
  activation="relu", padding="same"
)(inputs)

for _ in range(5):
  x = ResidualBlock(filters=128)(x)

for _ in range(2):
  x = MaskedConv2D(
    mask_type="B", filters=128, kernel_size=1,
    activation="relu", padding="valid"
  )(x)

out = layers.Conv2D(4, 1, activation="softmax", padding="valid")(x)
pixel_cnn = models.Model(inputs, out)

pixel_cnn.compile(
  optimizer=optimizers.Adam(learning_rate=5e-4),
  loss="sparse_categorical_crossentropy",
)

history = pixel_cnn.fit(
  x_in, y_out,
  batch_size=128,
  epochs=150,
  validation_data=(x_test_in, y_test_out),
  callbacks=[img_generator_callback],  # defined below
)
```

Training resembles an autoencoder (predict the image from itself) but the causal mask ensures each position's loss only uses legal context. Foster reports ~150 epochs on this small setup; loss should steadily decrease as the model learns garment shapes and textures.

---

## ImageGenerator Callback

Monitor qualitative progress during training by sampling partial images each epoch:

```python
class ImageGenerator(callbacks.Callback):
  def __init__(self, model, num_img=10):
    super().__init__()
    self.model = model
    self.num_img = num_img

  def sample_from(self, probs, temperature=1.0):
    probs = probs ** (1.0 / temperature)
    probs = probs / np.sum(probs)
    return np.random.choice(len(probs), p=probs)

  def generate(self, temperature=1.0):
    shape = (self.num_img,) + self.model.input_shape[1:]
    generated = np.zeros(shape, dtype=np.float32)
    _, rows, cols, ch = generated.shape
    for row in range(rows):
      for col in range(cols):
        for c in range(ch):
          probs = self.model.predict(generated, verbose=0)[:, row, col, :]
          generated[:, row, col, c] = [
            self.sample_from(p, temperature) / 4.0 for p in probs
          ]
    return generated

  def on_epoch_end(self, epoch, logs=None):
    imgs = self.generate(temperature=1.0)
    # save or display imgs — e.g., matplotlib grid
```

**Temperature** $> 1$ flattens distributions (more random); $< 1$ sharpens them (more deterministic). Start with $T = 1.0$.

---

## Analysis: What to Expect

After training, generated $16 \times 16$ Fashion-MNIST samples should capture **global shape** (trousers vs shirts vs bags) even if fine detail is soft. Foster's Figure 5-15 compares training set images with PixelCNN outputs — the model learns local color correlations and silhouette statistics from the chain-rule decomposition.

**Limitations at this scale:**

| Issue | Cause |
|-------|-------|
| Slow sampling | $H \times W$ sequential model calls |
| Limited resolution | 16×16 for tractable generation demos |
| Quantization banding | Only 4 gray levels |
| No global latent | No single vector summarizes "a shoe" |

[Section 5.8](./section-08-mixture-distributions-for-pixelcnn.md) addresses quantization via **mixture distributions**; Part III covers latent autoregressive models (VQ-VAE + PixelCNN) for high-resolution synthesis.

---

## Loss and Likelihood Connection

Minimizing sparse categorical cross-entropy over all positions is equivalent to maximizing:

$$
\log P_\theta(x) = \sum_{i} \log P_\theta(x_i \mid x_{<i})
$$
> **Readable form:** Total log-likelihood is the sum of per-pixel log-probabilities under the causal factorization.

Unlike GANs, you get a principled density estimate (on the quantized domain). Unlike VAEs, there is no encoder or ELBO — just direct MLE on pixels.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| `categorical_crossentropy` with integer labels | Shape mismatch | Use `sparse_categorical_crossentropy` |
| Training on 32×32 without patience | Hours of sampling per eval | Use 16×16 for coursework |
| No validation split | Overfitting unnoticed | Hold out `x_test` |
| Callback every batch | Training crawls | `on_epoch_end` only |

---

## Connection to Other Chapters

| Concept | Link |
|---------|------|
| ResNet skip connections | [Course 3 CNNs](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-09-convolutional-networks/README.md) |
| MLE / cross-entropy | [Course 3 probability](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-03-probability-information-theory/README.md) |
| Fashion-MNIST | [Section 3.1](../chapter-03-variational-autoencoders/section-01-autoencoder-architecture.md) |
| Mixture output upgrade | [Section 5.8](./section-08-mixture-distributions-for-pixelcnn.md) |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Residual block** | Layer group with skip connection $y = x + \mathcal{F}(x)$ |
| **Skip connection** | Shortcut path bypassing intermediate layers |
| **Sparse categorical cross-entropy** | Classification loss with integer class indices |
| **Quantization** | Mapping continuous $[0,1]$ pixels to discrete levels |
| **Generation callback** | Keras hook to sample images during training |

---

## Reflection Questions

1. Why does the residual block use a $1 \times 1$ conv before and after the masked $3 \times 3$ conv?
2. How many forward passes does generating one $16 \times 16$ image require?
3. Why is validation loss a better early-stop signal than saved sample images alone?
4. What would change if you replaced sparse CE with MSE on raw pixel values?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 5 — Residual Blocks, Training. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
- Foster's notebook: `notebooks/05_autoregressive/02_pixelcnn/pixelcnn.ipynb`

---

**Previous:** [Section 5.6 — PixelCNN](./section-06-pixelcnn.md)  
**Next:** [Section 5.8 — Mixture Distributions](./section-08-mixture-distributions-for-pixelcnn.md)
