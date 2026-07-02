# Section 8.2: Forward Diffusion Process

> **Source inheritance:** Foster, Ch. 8 — "The Forward Diffusion Process"  
> **Enhanced with:** Markov noising chain, variance preservation, closed-form $q(x_t \mid x_0)$, and flower dataset setup  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

The **forward diffusion process** gradually corrupts data with Gaussian noise over $T$ timesteps. Define $\beta_t \in (0, 1)$ as noise variance at step $t$. Each step:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\bigl(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I\bigr)
$$
> **Readable form:** Each forward step samples a slightly faded previous image plus Gaussian noise with variance beta.

Equivalently:

$$
x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$
> **Readable form:** Each step: shrink the image slightly, add a little Gaussian noise.

After $T$ steps (Foster uses $T=1000$), $x_T$ is approximately standard Gaussian if $x_0$ has zero mean and unit variance.

---

## Variance Preservation

The $\sqrt{1-\beta_t}$ scaling keeps $\text{Var}(x_t) = 1$ when $\text{Var}(x_{t-1}) = 1$:

$$
\text{Var}(\sqrt{1-\beta_t}\, x_{t-1}) = (1-\beta_t), \quad \text{Var}(\sqrt{\beta_t}\, \epsilon) = \beta_t
$$
> **Readable form:** The signal keeps variance $1-\beta_t$ while the newly added noise contributes variance $\beta_t$.

Sum: $(1-\beta_t) + \beta_t = 1$. By induction, if $x_0$ is normalized to unit variance, every $x_t$ stays unit variance — $x_T$ is a clean target for reverse sampling.

---

## Cumulative Notation

Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Jump from $x_0$ to any $x_t$ in one shot ([Section 8.3](./section-03-reparameterization-and-diffusion-schedules.md) expands):

$$
q(x_t \mid x_0) = \mathcal{N}\bigl(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1-\bar{\alpha}_t) I\bigr)
$$

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon
$$
> **Readable form:** Noisy image is a weighted blend of clean image and noise — weights depend only on timestep t.

This **reparameterization** is what makes training efficient — sample random $t$, noise once, form $x_t$ directly.

---

## Visualizing Forward Corruption

```python
import tensorflow as tf
import matplotlib.pyplot as plt

def forward_sample(x0, alpha_bar_t):
  eps = tf.random.normal(tf.shape(x0))
  xt = tf.sqrt(alpha_bar_t) * x0 + tf.sqrt(1.0 - alpha_bar_t) * eps
  return xt, eps

# Show corruption at t = 0, 250, 500, 750, 1000 (conceptual alpha_bar values)
timesteps = [0.0, 0.25, 0.5, 0.75, 0.99]
fig, axes = plt.subplots(1, len(timesteps), figsize=(12, 3))
for ax, t_frac in zip(axes, timesteps):
  ab = 1.0 - t_frac  # illustrative
  xt, _ = forward_sample(x0, ab)
  ax.imshow(tf.clip_by_value(xt[0], 0, 1))
  ax.set_title(f"t ≈ {int(t_frac*1000)}")
  ax.axis("off")
```

Early $t$: faint static. Late $t$: TV snow. Foster's Figure 8-3 shows this chain for flowers.

---

## Oxford 102 Flowers Dataset

Foster trains on 64×64 RGB flower images:

```python
import tensorflow as tf

train_data = tf.keras.utils.image_dataset_from_directory(
  "/app/data/pytorch-challange-flower-dataset/dataset",
  labels=None,
  image_size=(64, 64),
  batch_size=None,
  shuffle=True,
  seed=42,
)

def preprocess(img):
  return tf.cast(img, "float32") / 255.0

train = train_data.map(preprocess).repeat(5).batch(64, drop_remainder=True)
```

Download via Foster's script: `bash scripts/download_kaggle_data.sh nunenuh pytorch-challange-flower-dataset`. Repeat dataset 5× to lengthen epochs.

**Normalization note:** Foster's training code uses `layers.Normalization` to zero-mean unit-variance **inside** the diffusion model ([Section 8.6](./section-06-training-the-diffusion-model.md)) — forward math assumes $x_0$ with $\mathbb{E}[x]=0$, $\text{Var}(x)=1$ after that layer.

---

## Markov Chain Structure

The forward process factorizes:

$$
q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1})
$$
> **Readable form:** Multiply local factors to form the joint quantity or sequence probability.

Only **adjacent** pairs matter for the generative story — but training uses the closed form $q(x_t \mid x_0)$ to skip the loop.

The forward process has **no learnable parameters** — $\beta_t$ (or $\bar{\alpha}_t$) comes from a fixed **schedule**.

---

## Forward vs VAE Encoder

| | VAE encoder | Diffusion forward |
|---|-------------|-------------------|
| Learned? | Yes $q_\phi(z \mid x)$ | No — fixed Gaussians |
| Latent | Low-dimensional $z$ | Same shape as $x$, noisy |
| Purpose | Compress for ELBO | Destroy info so reverse can learn denoise |
| Invertible? | Approximate | Exact in distribution |

Diffusion keeps full resolution through the chain — $x_t$ is an image, not a 100-D code.

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Missing $\sqrt{1-\beta_t}$ scale | Variance drift | Use Ho et al. parameterization |
| $x_0$ in $[0,1]$ only | Schedule mismatch | Apply `Normalization` layer |
| Wrong $\bar{\alpha}_t$ indexing | Off-by-one noise level | `cumprod` from $t=1$ |
| Too large $\beta_t$ early | Instant mush | Cosine/linear schedules |

---

## Connection to Other Sections

| Concept | Link |
|---------|------|
| Schedules $\beta_t$ | [Section 8.3](./section-03-reparameterization-and-diffusion-schedules.md) |
| Reverse $p_\theta$ | [Section 8.4](./section-04-reverse-diffusion-process.md) |
| Training uses $x_t$ | [Section 8.6](./section-06-training-the-diffusion-model.md) |

---

## Step-by-Step Corruption Demo

```python
import matplotlib.pyplot as plt

def show_forward_strip(x0, schedule_fn, n_panels=6):
  times = tf.linspace(0.0, 1.0, n_panels)
  fig, axes = plt.subplots(1, n_panels, figsize=(14, 3))
  for ax, t in zip(axes, times):
    nr, sr = schedule_fn(t[None, None, None, None])
    xt, _ = forward_sample(x0, sr ** 2)  # use signal fraction squared if matching Foster
    ax.imshow(tf.clip_by_value(xt[0], 0, 1))
    ax.set_title(f"t={float(t):.2f}")
    ax.axis("off")
```

Run on a single flower — builds intuition before training the U-Net.

---

## Inductive Argument for Unit Variance

Base: $\text{Var}(x_0)=1$. Step: if $\text{Var}(x_{t-1})=1$, then $x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon$ has variance $(1-\beta_t)+\beta_t=1$. By induction all $x_t$ unit variance — critical so $x_T \approx \mathcal{N}(0,I)$ without per-step renormalization hacks.

---

## Connection to ELBO (Preview)

The forward process defines $q(x_{1:T}|x_0)$ used in the variational bound derived in Ho et al. — Foster focuses on the simplified $\epsilon$-loss rather than full ELBO algebra, but the forward formulas in this section are the same $q$ terms appearing in the bound. [Section 8.4](./section-04-reverse-diffusion-process.md) explains how optimizing noise prediction relates to maximizing that bound.

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Forward process** | Fixed noising Markov chain $q$ |
| **$\beta_t$** | Noise variance schedule at step $t$ |
| **$\bar{\alpha}_t$** | Cumulative signal retention $\prod(1-\beta_s)$ |
| **Markov chain** | Each step depends only on previous $x_{t-1}$ |
| **Variance preservation** | Scaling keeps unit variance across $t$ |

---

## Reflection Questions

1. Why multiply $x_{t-1}$ by $\sqrt{1-\beta_t}$ before adding noise?
2. What does $x_T$ look like when $T=1000$ and the schedule is well-tuned?
3. How is $q(x_t \mid x_0)$ more useful at training time than iterating $q(x_t \mid x_{t-1})$?
4. Why does Foster use flowers instead of MNIST for diffusion?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 8 — Forward Process. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Ho, J. et al. (2020). DDPM. [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- Foster's notebook: `notebooks/08_diffusion/01_ddm/ddm.ipynb`

---

**Previous:** [Section 8.1 — Diffusion Introduction](./section-01-diffusion-models-introduction.md)  
**Next:** [Section 8.3 — Reparameterization & Schedules](./section-03-reparameterization-and-diffusion-schedules.md)
