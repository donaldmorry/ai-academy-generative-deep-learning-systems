# Section 6.4: Two Moons Dataset

> **Source inheritance:** Foster, Ch. 6 — "The Two Moons Dataset"  
> **Enhanced with:** sklearn data generation, normalization, visualization, and why 2D flows teach invertibility  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Before scaling [RealNVP](./section-03-realnvp-architecture.md) to images, Foster trains on the **two moons** dataset — 3,000 noisy 2D points forming interlocking crescents. In two dimensions you can **plot** the data, the learned Gaussian latent space, and generated samples side by side. Every Jacobian determinant and coupling mask becomes visible.

The goal: learn a bijection $f: \mathbb{R}^2 \to \mathbb{R}^2$ mapping moons to $\mathcal{N}(0, I)$, then sample $z \sim \mathcal{N}(0,I)$ and invert to new moon points.

> **Readable form:** Two banana-shaped clouds in the plane → squeeze to a round Gaussian blob → sample the blob and unsqueeze to new bananas.

---

## Creating the Dataset

```python
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.keras import layers

# 3000 points, small noise
raw = datasets.make_moons(n_samples=3000, noise=0.05)[0].astype("float32")

# Keras Normalization: zero mean, unit variance per axis
norm = layers.Normalization()
norm.adapt(raw)
data = norm(raw).numpy()
```

`make_moons` produces two semicircles offset horizontally — a classic **multimodal** distribution. Noise=0.05 prevents exact overlap and mimics real measurement error. Normalization centers the cloud and unit-scales each axis so coupling layers operate in a stable range.

---

## Why Two Moons for Flows?

| Property | Why it matters for flows |
|----------|--------------------------|
| 2D | Plot $p(x)$, $p(z)$, samples on paper |
| Multimodal | Single Gaussian cannot fit — needs expressive $f$ |
| Smooth manifolds | Invertible maps can untangle curves |
| Fast training | Seconds per epoch on CPU |

Contrast with [Fashion-MNIST](../chapter-03-variational-autoencoders/section-01-autoencoder-architecture.md): 784 dimensions hide geometric intuition. Moons are the flow equivalent of XOR for neural nets — minimal but revealing.

---

## Visualizing the Raw Distribution

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
plt.scatter(data[:, 0], data[:, 1], s=5, alpha=0.5)
plt.title("Normalized two moons")
plt.xlabel("$x_1$"); plt.ylabel("$x_2$")
plt.axis("equal")
plt.savefig("moons_raw.png")
```

You should see two crescents centered near the origin. The RealNVP's job: warp this into a circular Gaussian cloud in latent space.

---

## Data Pipeline for Training

```python
batch_size = 128
train_ds = (
  tf.data.Dataset.from_tensor_slices(data)
  .shuffle(3000)
  .batch(batch_size)
  .prefetch(tf.data.AUTOTUNE)
)
```

Foster trains entirely on this slice — no held-out test set for the toy example. Evaluation is **visual**: do generated points land on the moon manifolds? Does latent $z$ look Gaussian?

For coursework, reserve 20% for log-likelihood monitoring:

```python
n = len(data)
split = int(0.8 * n)
train_data, val_data = data[:split], data[split:]
```

---

## Target Latent Distribution

The base distribution is standard bivariate Gaussian:

$$
p_Z(z) = \mathcal{N}(z; 0, I)
$$
> **Readable form:** Latent space is a 2D bell curve centered at origin with unit variance per axis.

After training, forward pass $z = f(x)$ should map moon points to roughly circular scatter. Inverse $x = f^{-1}(z)$ maps Gaussian draws back to crescents. The [change-of-variables](./section-02-change-of-variables.md) equation ties them:

$$
\log p_X(x) = \log p_Z(f(x)) + \log \left| \det \frac{\partial f}{\partial x} \right|
$$
> **Readable form:** This derivative measures local sensitivity: a larger magnitude means the output changes more for a small input change.

---

## Coupling Layer Masks in 2D

With $D = 2$, alternating masks are simple:

| Layer | Mask on input | Updated dims |
|-------|---------------|--------------|
| Odd | `[1, 0]` | $x_2$ |
| Even | `[0, 1]` | $x_1$ |

```python
mask_a = np.array([1.0, 0.0])
mask_b = np.array([0.0, 1.0])
```

Each mask zeros one coordinate before the coupling net, ensuring the affine update applies only to the complementary half. After 4–8 stacked layers, both coordinates are thoroughly mixed.

---

## Expected Training Outcomes

After sufficient epochs ([Section 6.5](./section-05-training-realnvp.md)):

1. **Forward:** Moon points → tight Gaussian blob in $z$-space
2. **Inverse samples:** Gaussian noise → new points on both crescents
3. **Log-likelihood:** Steady increase on training data
4. **Failure mode:** Collapsed map (all $z$ identical) → check masks and learning rate

Foster's Figure 6-4 shows the raw moons; Figures 6-9–6-11 show the stacked coupling architecture and transformed space.

---

## From Moons to Images

The two-moons experiment uses `Dense` coupling networks. Image flows ([GLOW](./section-07-glow.md)) replace:

- Points $\mathbb{R}^2$ → tensors $\mathbb{R}^{H \times W \times C}$
- Coordinate masks → checkerboard / channel-split masks
- Dense → Conv2D stacks

The **math** (affine coupling, log-det, alternating masks) is unchanged. Master moons first; image flows become an engineering scale-up.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Skipping normalization | Training unstable | `layers.Normalization().adapt()` |
| Too few coupling layers | Poor Gaussianization | Stack 6+ layers |
| Huge learning rate | NaN in $\exp(s)$ | Start ~1e-3 Adam |
| noise=0 in make_moons | Degenerate overlaps | Use noise ≥ 0.05 |

---

## Connection to Other Chapters

| Concept | Link |
|---------|------|
| RealNVP layers | [Section 6.3](./section-03-realnvp-architecture.md) |
| NLL training | [Section 6.5](./section-05-training-realnvp.md) |
| sklearn datasets | [Course 1](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-01-machine-learning/README.md) |
| Multimodality | [Chapter 01 taxonomy](../chapter-01-generative-modeling/section-07-generative-model-taxonomy.md) |

---

## sklearn Parameters Explained

```python
datasets.make_moons(n_samples=3000, noise=0.05, random_state=42)
```

| Argument | Effect |
|----------|--------|
| `n_samples` | 3000 points — enough density to learn two curves |
| `noise` | Gaussian jitter — prevents zero-width manifolds |
| `random_state` | Reproducible splits for grading |

Try `noise=0.0` once — flow still trains but overfits thin curves; `noise=0.2` — moons merge visually, harder density task.

---

## Exporting Latent Scatter for Reports

```python
z, _ = flow_model(data, inverse=False)
np.savetxt("moons_latent.csv", z.numpy(), delimiter=",", header="z1,z2")
```

Include this CSV + scatter PNG in lab submission — demonstrates Gaussianization quantitatively (eyeball + optional 2D KS test on radial distance).

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Two moons** | sklearn benchmark: two interleaved 2D crescents |
| **Normalization layer** | Keras layer for feature-wise mean/var scaling |
| **Multimodal distribution** | Multiple separated regions of high density |
| **Latent Gaussian** | Simple base distribution flows map onto |
| **Manifold** | Low-dimensional structure in higher-dimensional space |

---

## Reflection Questions

1. Why does Foster normalize the moons before training RealNVP?
2. How would a single linear transform fail on this dataset?
3. What visual evidence confirms the flow has learned a valid bijection?
4. How many coupling layers are needed before both coordinates are updated at least once?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 6 — Two Moons. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- scikit-learn: `sklearn.datasets.make_moons`
- Foster's notebook: `notebooks/06_normflow/01_realnvp/realnvp.ipynb`

---

**Previous:** [Section 6.3 — RealNVP Architecture](./section-03-realnvp-architecture.md)  
**Next:** [Section 6.5 — Training RealNVP](./section-05-training-realnvp.md)
