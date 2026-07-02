# Section 7.2: MNIST Energy Function

> **Source inheritance:** Foster, Ch. 7 — "The MNIST Dataset" / "The Energy Function"  
> **Enhanced with:** MNIST preprocessing, swish activation, Conv2D energy network, and scalar output design  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

The **energy function** $E_\theta(x)$ is a neural network mapping an image to a single scalar. Foster trains on **MNIST** — 28×28 handwritten digits padded to 32×32, pixels scaled to $[-1, 1]$. Architecture: stacked strided Conv2D layers with **swish** activation, ending in `Dense(1)` with linear output.

Low energy on real digits, high energy on noise — the network is a **critic of images**, not a classifier. No softmax over 10 classes; one unbounded energy value per image.

> **Readable form:** CNN shrinks the image to one number — how "MNIST-like" is this input?

---

## Loading and Preprocessing MNIST

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets

(x_train, _), (x_test, _) = datasets.mnist.load_data()

def preprocess(imgs):
  imgs = (imgs.astype("float32") - 127.5) / 127.5  # [-1, 1]
  imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=-1.0)
  imgs = np.expand_dims(imgs, -1)  # (N, 32, 32, 1)
  return imgs

x_train = preprocess(x_train)
x_test = preprocess(x_test)

train_ds = tf.data.Dataset.from_tensor_slices(x_train).batch(128)
test_ds = tf.data.Dataset.from_tensor_slices(x_test).batch(128)
```

**Why $[-1, 1]$?** Matches Langevin sampling initialization ([Section 7.3](./section-03-langevin-dynamics-sampling.md)) — fake samples start uniform in $[-1,1]$ and gradient-descend toward digit manifolds.

**Why pad to 32×32?** Stride-2 conv stacks divide evenly; same convention as [DCGAN](../chapter-04-generative-adversarial-networks/section-02-dcgan-architecture.md) and Foster's EBM notebook.

---

## Swish Activation

Google's **swish** (2017) replaces ReLU in EBMs:

$$
\text{swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$
> **Readable form:** Smooth, non-monotonic — small negative inputs get small negative outputs instead of hard zero.

| | ReLU | Swish |
|---|------|-------|
| Smoothness | Kink at 0 | Smooth everywhere |
| Negative inputs | Clipped to 0 | Small negative values pass |
| EBM training | Vanishing gradients in flat regions | Better gradient flow for Langevin |

```python
from tensorflow.keras import activations

x = layers.Conv2D(
  16, kernel_size=5, strides=2, padding="same",
  activation=activations.swish
)(ebm_input)
```

---

## Energy Network Architecture

Foster's $E_\theta(x)$ (Example 7-3):

| Layer | Output shape | Notes |
|-------|--------------|-------|
| Input | (32, 32, 1) | Grayscale digit |
| Conv2D 16, 5×5, stride 2, swish | (16, 16, 16) | |
| Conv2D 32, 3×3, stride 2, swish | (8, 8, 32) | |
| Conv2D 64, 3×3, stride 2, swish | (4, 4, 64) | |
| Conv2D 64, 3×3, stride 2, swish | (2, 2, 64) | |
| Flatten | (256,) | |
| Dense 64, swish | (64,) | |
| Dense 1, linear | (1,) | **Energy scalar** |

```python
from tensorflow.keras import layers, models

ebm_input = layers.Input(shape=(32, 32, 1))
x = layers.Conv2D(16, 5, strides=2, padding="same", activation=activations.swish)(ebm_input)
x = layers.Conv2D(32, 3, strides=2, padding="same", activation=activations.swish)(x)
x = layers.Conv2D(64, 3, strides=2, padding="same", activation=activations.swish)(x)
x = layers.Conv2D(64, 3, strides=2, padding="same", activation=activations.swish)(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation=activations.swish)(x)
ebm_output = layers.Dense(1)(x)  # linear — energy unbounded
energy_model = models.Model(ebm_input, ebm_output)
energy_model.summary()
```

**No sigmoid/softmax** — energy can be any real number. Regularization on energy magnitudes comes in the training loss ([Section 7.5](./section-05-training-the-ebm.md)).

---

## Energy vs Discriminator

| | GAN Discriminator $D(x)$ | EBM Energy $E(x)$ |
|---|--------------------------|-------------------|
| Output | Probability in $[0,1]$ | Unbounded scalar |
| Training signal | Real/fake labels | Contrastive energy gap |
| Sampling | Generator only | Langevin on $x$ |
| Interpretation | "Realness" logit | Physical "energy" |

Both distinguish real from fake — EBMs make the score an energy landscape you can **climb down** in pixel space.

---

## Forward Pass Usage

```python
# Random noise — expect HIGH energy before training
noise = tf.random.uniform((4, 32, 32, 1)) * 2 - 1
print(energy_model(noise).numpy())  # large positive values typical

# Real digit — expect lower energy after training
print(energy_model(x_train[:4]).numpy())
```

Before contrastive training, energies are random. After training ([Section 7.6](./section-06-analysis-of-the-ebm.md)), real batch mean $\ll$ fake batch mean.

---

## Design Choices

**Strided convolutions** — spatial downsampling like a discriminator pyramid; receptive field covers full 32×32 at the end.

**Linear final layer** — energies must span wide range to separate noise ($E \gg 0$) from data ($E \ll 0$).

**Single channel** — MNIST grayscale; RGB EBMs add channels and deepen stacks.

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Pixels in $[0,1]$ not $[-1,1]$ | Langevin clip mismatch | Rescale with `(x-127.5)/127.5` |
| ReLU only | Slow Langevin convergence | Use swish |
| Sigmoid on output | Bounded energy — weak contrast | Linear Dense(1) |
| Forgetting padding | Shape errors in conv stack | Pad 28→32 |

---

## Connection to Other Chapters

| Concept | Link |
|---------|------|
| MNIST / padding | [Chapter 04 DCGAN](../chapter-04-generative-adversarial-networks/section-02-dcgan-architecture.md) |
| Conv2D strides | [Chapter 02](../chapter-02-deep-learning/section-06-convolutional-layers.md) |
| Langevin sampling | [Section 7.3](./section-03-langevin-dynamics-sampling.md) |
| EBM intro | [Section 7.1](./section-01-energy-based-models-introduction.md) |

---

## Parameter Count Estimate

Rough order for Foster's energy CNN:

| Block | Approx params |
|-------|---------------|
| Conv 16×5×5 | 16×25 ≈ 400 |
| Conv stacks | ~50K |
| Dense 64 + 1 | ~16K |
| **Total** | ~70K — tiny vs diffusion U-Net |

Small footprint — EBM training time dominated by Langevin loops, not parameter count.

---

## Inspecting Intermediate Activations

```python
layer_model = models.Model(
  inputs=energy_model.input,
  outputs=energy_model.layers[-3].output  # last conv block
)
feat = layer_model(x_train[:1])
print(feat.shape)  # spatial map before scalar energy
```

Useful when digits get high energy despite looking correct — spatial features may be OOD even if pixels look fine.

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Energy function** | Neural network $E_\theta(x) \in \mathbb{R}$ |
| **Swish** | Smooth activation $x \cdot \sigma(x)$ |
| **Scalar output** | One energy per image, not per-pixel |
| **Stride-2 conv** | Downsampling without pooling |
| **Pixel range $[-1,1]$** | Standard scaling for EBM + Langevin |

---

## Reflection Questions

1. Why does the final layer use linear activation instead of sigmoid?
2. How does swish help gradients flow during Langevin sampling?
3. What output shape would distinguish an EBM from a 10-class MNIST classifier?
4. Why pad digits to 32×32 instead of training at 28×28?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 7 — MNIST, Energy Function. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Ramachandran, P. et al. (2017). Searching for Activation Functions (swish). [https://arxiv.org/abs/1710.05941](https://arxiv.org/abs/1710.05941)
- Foster's notebook: `notebooks/07_ebm/01_ebm/ebm.ipynb`

---

**Previous:** [Section 7.1 — EBM Introduction](./section-01-energy-based-models-introduction.md)  
**Next:** [Section 7.3 — Langevin Dynamics Sampling](./section-03-langevin-dynamics-sampling.md)



