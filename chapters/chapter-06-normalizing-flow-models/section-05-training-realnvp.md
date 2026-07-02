# Section 6.5: Training RealNVP

> **Source inheritance:** Foster, Ch. 6 — "Training the RealNVP Model"  
> **Enhanced with:** Negative log-likelihood loss, forward/inverse passes, custom train step, and log-det accumulation  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

[RealNVP](./section-03-realnvp-architecture.md) training maximizes **exact log-likelihood** via the [change-of-variables](./section-02-change-of-variables.md) formula. For each batch of moon points $x$, forward through coupling layers to get $z = f(x)$ and accumulate $\log |\det J|$. The loss is **negative** log-likelihood:

$$
\mathcal{L} = -\mathbb{E}_{x \sim \text{data}}\left[ \log p_Z(f(x)) + \log \left| \det \frac{\partial f}{\partial x} \right| \right]
$$
> **Readable form:** Penalize data that maps to low-density regions of the Gaussian or uses extreme scaling.

Unlike VAEs (ELBO) or GANs (adversarial game), this is plain gradient ascent on a principled density — MLE for flows.

---

## Building the Flow Model

Foster stacks alternating coupling layers with shared hyperparameters:

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers

class RealNVP(tf.keras.Model):
  def __init__(self, n_layers=6, input_dim=2):
    super().__init__()
    self.layers_list = []
    masks = [np.array([1., 0.]), np.array([0., 1.])]
    for i in range(n_layers):
      mask = masks[i % 2]
      self.layers_list.append(
        CouplingLayer(Coupling(input_dim), mask)
      )

  def call(self, x, inverse=False):
    log_det = 0.0
    if not inverse:
      for layer in self.layers_list:
        x, ld = layer(x, inverse=False)
        log_det += ld
      return x, log_det
    else:
      for layer in reversed(self.layers_list):
        x, ld = layer(x, inverse=True)
        log_det += ld
      return x, log_det
```

Each `CouplingLayer` returns transformed $z$ (or $x$) plus per-sample `log_det_jacobian`.

---

## Log-Likelihood Computation

Base log-density for $\mathcal{N}(0, I)$ in $d$ dimensions:

$$
\log p_Z(z) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\|z\|^2
$$
> **Readable form:** Under a standard normal prior, log probability decreases with dimensionality and squared distance from the origin.

```python
def log_normal(z):
  d = tf.cast(tf.shape(z)[-1], tf.float32)
  return -0.5 * d * tf.math.log(2.0 * np.pi) - 0.5 * tf.reduce_sum(z ** 2, axis=-1)
```

Full batch log-likelihood:

```python
def compute_log_likelihood(model, x):
  z, log_det = model(x, inverse=False)
  return log_normal(z) + log_det
```

Minimize `nll = -tf.reduce_mean(compute_log_likelihood(model, x))`.

---

## Custom train_step

```python
class FlowTrainer(tf.keras.Model):
  def __init__(self, flow):
    super().__init__()
    self.flow = flow
    self.nll_tracker = tf.keras.metrics.Mean(name="nll")

  @property
  def metrics(self):
    return [self.nll_tracker]

  def train_step(self, data):
    with tf.GradientTape() as tape:
      z, log_det = self.flow(data, inverse=False)
      log_pz = log_normal(z)
      log_px = log_pz + log_det
      nll = -tf.reduce_mean(log_px)
    grads = tape.gradient(nll, self.flow.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.flow.trainable_variables))
    self.nll_tracker.update_state(nll)
    return {m.name: m.result() for m in self.metrics}
```

Foster trains for hundreds of epochs on the [two moons](./section-04-two-moons-dataset.md) with Adam (~1e-3). NLL should decrease monotonically early on, then plateau.

---

## Sampling (Inverse Pass)

```python
flow_trainer = FlowTrainer(RealNVP(n_layers=8))
flow_trainer.compile(optimizer=optimizers.Adam(1e-3))
flow_trainer.fit(train_ds, epochs=200)

# Generate 1000 new moon points
z_sample = tf.random.normal((1000, 2))
x_gen, _ = flow_trainer.flow(z_sample, inverse=True)

plt.scatter(x_gen[:, 0], x_gen[:, 1], s=5, alpha=0.5)
plt.title("RealNVP samples")
```

One draw from $\mathcal{N}(0,I)$, one inverse pass — **fast sampling**, a key flow advantage over PixelCNN.

---

## Monitoring Training

| Metric | Healthy trend |
|--------|-----------------|
| `nll` | Decreasing, then stable |
| Latent scatter | Circular, centered at 0 |
| Generated moons | Both crescents populated |
| $\|z\|$ histogram | Matches $\chi^2$ with 2 dof roughly |

Plot latent codes every 50 epochs:

```python
z_val, _ = flow_trainer.flow(val_data, inverse=False)
plt.scatter(z_val[:, 0], z_val[:, 1], s=3, alpha=0.4)
```

If latent points form crescents (not Gaussians), add layers or train longer.

---

## RealNVP vs Other Training Objectives

| Model | Training loss | Exact $p(x)$? |
|-------|---------------|---------------|
| RealNVP | $-\log p_X(x)$ | Yes |
| VAE | $-\text{ELBO}$ | Lower bound |
| GAN | Adversarial | No |
| PixelCNN | $-\sum \log P(x_i|x_{<i})$ | Yes (discrete) |

Flows occupy the sweet spot: **exact likelihood + fast sampling**. Cost: architectural restrictions (invertibility) and weaker SOTA image quality vs diffusion on some benchmarks.

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Wrong sign on log-det | NLL increases | Add log-det, not subtract |
| Training in inverse mode | Nonsense gradients | Forward for likelihood |
| No shuffle | Biased batches | `dataset.shuffle()` |
| Exploding $\exp(s)$ | NaN mid-training | tanh on $s$, lower LR |

---

## Connection to Prior Sections

| Concept | Link |
|---------|------|
| Coupling math | [Section 6.3](./section-03-realnvp-architecture.md) |
| Moons data | [Section 6.4](./section-04-two-moons-dataset.md) |
| Analysis plots | [Section 6.6](./section-06-analysis-of-realnvp.md) |
| GLOW images | [Section 6.7](./section-07-glow.md) |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Negative log-likelihood (NLL)** | Loss $-\log p(x)$ for density estimation |
| **Log-determinant** | $\log |\det J|$ from coupling scales |
| **Forward pass** | Data → latent (encoding) |
| **Inverse pass** | Latent → data (generation) |
| **Maximum likelihood estimation** | Fit $\theta$ to maximize $\prod p_\theta(x_i)$ |

---

## Reflection Questions

1. Why do we add `log_det` to `log_pz` rather than treat them as separate losses?
2. What happens to NLL if the flow maps all data to $z = 0$?
3. How does training complexity compare to one GAN epoch on the same data?
4. Why is the inverse pass used at inference but not in the NLL computation?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 6 — Training RealNVP. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Dinh, L. et al. (2017). Density Estimation using Real NVP. [https://arxiv.org/abs/1605.08803](https://arxiv.org/abs/1605.08803)
- Foster's notebook: `notebooks/06_normflow/01_realnvp/realnvp.ipynb`

---

**Previous:** [Section 6.4 — Two Moons Dataset](./section-04-two-moons-dataset.md)  
**Next:** [Section 6.6 — Analysis of RealNVP](./section-06-analysis-of-realnvp.md)
