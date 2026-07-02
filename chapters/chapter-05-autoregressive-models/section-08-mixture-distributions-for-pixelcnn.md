# Section 5.8: Mixture Distributions for PixelCNN

> **Source inheritance:** Foster, Ch. 5 — "Mixture Distributions"  
> **Enhanced with:** Logistic mixtures, TensorFlow Probability PixelCNN, and escaping coarse quantization  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

The simplified PixelCNN from [Sections 5.6–5.7](./section-06-pixelcnn.md) predicts a **4-way softmax** over quantized pixel levels. That keeps training fast but throws away most of the gray-scale spectrum — unacceptable for color images with millions of distinct values.

**Mixture distributions** solve this: instead of 256 independent logits per pixel, the network outputs parameters for $K$ **sub-distributions** (e.g., logistics) plus a **categorical** over which sub-distribution to use. A handful of parameters can represent multimodal, continuous-valued pixel densities. Salimans et al. extended PixelCNN this way; Foster wraps TensorFlow Probability's built-in `PixelCNN` distribution.

> **Readable form:** Predict a weighted blend of simple curves (logistics) instead of a single histogram with 256 bins.

---

## Why Softmax Over 256 Levels Fails

A categorical distribution over pixel values $0, \ldots, 255$ needs 255 free parameters per location (per channel). The model must learn independently that level 200 is "close" to 201 — no inductive bias for smoothness. Training on full-resolution MNIST or CIFAR with raw softmax is prohibitively slow.

Quantizing to 4 levels ([Section 5.7](./section-07-residual-blocks-and-training-the-pixelcnn.md)) is a hack. **Mixture models** retain continuous structure: each logistic has location and scale; nearby values share statistical mass.

---

## Mixture Distribution Math

A mixture of $K$ distributions:

$$
p(x) = \sum_{k=1}^{K} \pi_k \, p_k(x \mid \theta_k)
$$
> **Readable form:** Probability at x equals the weighted sum of K component densities.

where $\pi_k$ are mixture weights ($\sum_k \pi_k = 1$) from a categorical, and each $p_k$ might be a logistic:

$$
p_k(x) = \frac{\exp(-(x - \mu_k) / s_k)}{s_k (1 + \exp(-(x - \mu_k) / s_k))^2}
$$
> **Readable form:** Component $k$ assigns logistic density to value $x$ using its own location $\mu_k$ and scale $s_k$.

**Sampling:** draw $k \sim \text{Categorical}(\pi)$, then $x \sim p_k$. **Eight parameters** can define a 3-component Gaussian mixture (Foster's Figure 5-16) vs 255 for a full categorical — dramatic compression with expressive power.

For PixelCNN, each spatial location outputs all mixture parameters conditioned on past pixels — still autoregressive, but with a richer output head.

---

## TensorFlow Probability PixelCNN

Foster's second notebook (`pixelcnn_md.ipynb`) uses `tfp.distributions.PixelCNN`:

```python
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, models, optimizers

tfd = tfp.distributions

dist = tfd.distributions.PixelCNN(
  image_shape=(32, 32, 1),
  num_resnet=1,
  num_hierarchies=2,
  num_filters=32,
  num_logistic_mix=5,
  dropout_p=0.3,
)

image_input = layers.Input(shape=(32, 32, 1))

# Log-likelihood loss: maximize log p(x)
log_prob = dist.log_prob(image_input)
loss = -tf.reduce_mean(log_prob)

trainable_vars = (
  dist.trainable_variables
  if hasattr(dist, "trainable_variables")
  else []
)

# Foster wires a custom train step; simplified pattern:
class PixelCNNModel(models.Model):
  def __init__(self, dist):
    super().__init__()
    self.dist = dist

  def call(self, x):
    return self.dist.log_prob(x)

  def train_step(self, data):
    with tf.GradientTape() as tape:
      nll = -tf.reduce_mean(self.dist.log_prob(data))
    grads = tape.gradient(nll, self.dist.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.dist.trainable_variables))
    return {"nll": nll}
```

Key hyperparameters:

| Parameter | Role |
|-----------|------|
| `num_logistic_mix` | Number of logistic components per pixel |
| `num_hierarchies` | Multi-scale autoregressive stacks |
| `num_resnet` | Residual blocks per hierarchy |
| `dropout_p` | Regularization |

Training maximizes $\log p(x)$ directly — the exact generative objective for [autoregressive models](../../GLOSSARY.md#autoregressive-model).

---

## Hierarchical PixelCNN

`num_hierarchies=2` builds a **coarse-to-fine** factorization: lower-resolution latents are generated first, then refined. This mirrors how human artists sketch structure before detail. The TFP implementation hides mask and residual details inside the distribution class — you trade pedagogical transparency ([Section 5.6](./section-06-pixelcnn.md)) for production-ready mixture outputs.

For coursework: implement masked convs by hand first; then appreciate the TFP shortcut for full 32×32 grayscale with 5 logistic mixtures.

---

## Sampling from the Mixture PixelCNN

```python
# After training
samples = dist.sample(16)  # batch of 16 images
samples = tf.clip_by_value(samples, 0.0, 1.0)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for ax, img in zip(axes.flat, samples):
  ax.imshow(img.numpy().squeeze(), cmap="gray")
  ax.axis("off")
plt.savefig("pixelcnn_md_samples.png")
```

Sampling remains sequential at the autoregressive level — TFP handles the inner mixture draw per pixel. Expect sharper tonal gradients than 4-level quantization, at the cost of longer training.

---

## Comparison: Quantized vs Mixture PixelCNN

| Aspect | 4-level softmax | Logistic mixture (TFP) |
|--------|-----------------|------------------------|
| Output params | 4 per pixel | $3K + K$ (mix weights + logistics) |
| Color depth | Banding | Smooth grayscale / color |
| Implementation | Hand-rolled Keras | `tfp.distributions.PixelCNN` |
| Image size in Foster | 16×16 | 32×32 |
| Training loss | Sparse CE | Negative log-likelihood |

---

## Connection to Broader Generative Modeling

Mixture outputs appear beyond PixelCNN:

- **MDN-RNNs** — mixture density networks for continuous sequences
- **GMM priors** in some VAE variants
- **Audio models** — logistic mixtures over $\mu$-law samples

The theme: **factor complex distributions into parameterized components** instead of brute-force discretization. Diffusion models ([Chapter 08](../chapter-08-diffusion-models/section-01-diffusion-models-introduction.md)) take a different path — iterative denoising — but autoregressive + mixtures remains central in codecs (VQ-VAE + Transformer) and language modeling (softmax over 50k tokens is itself a huge categorical).

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Missing `tensorflow_probability` | Import error | `pip install tensorflow-probability` |
| Optimizing wrong variables | Loss flatlines | Train `dist.trainable_variables` |
| Forgetting input scale | Logistic mismatch | Keep pixels in $[0, 1]$ as Foster |
| Expecting fast sampling | Hours for 32×32 | Use fewer hierarchies for demos |

---

## PixelCNN++ Improvements (Context)

Salimans et al. added beyond mixture outputs:

- **Gated activation** units in masked conv stacks
- **Shortcut between color channels** at each position
- **Residual connections** at full resolution (you built these in [Section 5.7](./section-07-residual-blocks-and-training-the-pixelcnn.md))
- **Skip connections** from input to intermediate layers

TFP `PixelCNN` bundles many upgrades — when reading PixelCNN++ papers, map each trick to the hand-rolled components you already understand from Foster's first notebook.

---

## Choosing K Mixture Components

| `num_logistic_mix` | Trade-off |
|--------------------|-----------|
| 1 | Near single logistic — fast, limited |
| 5 | Foster default — good grayscale |
| 10+ | Color CIFAR — more expressivity, slower |

Monitor NLL on validation — if NLL plateaus high, increase `num_filters` before increasing K.

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Mixture distribution** | Weighted combination of simpler distributions |
| **Logistic mixture** | Mixture of logistic PDFs for continuous pixels |
| **Categorical mixing weights** | $\pi_k$ selecting which component generates |
| **Negative log-likelihood (NLL)** | Loss $-\log p(x)$ for explicit density models |
| **Hierarchical PixelCNN** | Multi-scale autoregressive generation |

---

## Reflection Questions

1. How many parameters does a 3-component Gaussian mixture need vs a 256-bin categorical?
2. Why does a mixture model encode "closeness" between pixel values better than raw softmax?
3. What does `num_hierarchies=2` change about generation order?
4. When would you still prefer a VAE or diffusion model over PixelCNN with mixtures?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 5 — Mixture Distributions. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Salimans, T. et al. (2017). PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood. [https://arxiv.org/abs/1701.05517](https://arxiv.org/abs/1701.05517)
- TensorFlow Probability: [PixelCNN distribution](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/PixelCNN)
- Foster's notebook: `notebooks/05_autoregressive/03_pixelcnn_md/pixelcnn_md.ipynb`

---

**Previous:** [Section 5.7 — Residual Blocks & Training](./section-07-residual-blocks-and-training-the-pixelcnn.md)  
**Next:** [Lab 05](./section-lab-05-lstm-recipe-generator-and-pixelcnn.md)
