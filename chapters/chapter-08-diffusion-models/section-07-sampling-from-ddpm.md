# Section 8.7: Sampling from DDPM

> **Source inheritance:** Foster, Ch. 8 — reverse sampling / generation  
> **Enhanced with:** Iterative denoising loop, EMA network, DDIM preview, and flower generation  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

After [training](./section-06-training-the-diffusion-model.md), generate flowers by reversing the diffusion chain: sample $x_T \sim \mathcal{N}(0, I)$, then for $t = T, T-1, \ldots, 1$ predict noise with the **EMA U-Net**, subtract a denoising update, add small Gaussian jitter (except final step). Foster uses $T=1000$ steps — quality scales with step count; **DDIM** can reduce steps later.

> **Readable form:** Start with TV static; run 1000 denoise micro-steps; a new flower appears.

---

## Reverse Step Mathematics

Given $x_t$ and predicted $\hat{\epsilon} = \epsilon_\theta(x_t, t)$, estimate clean image:

$$
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\, \hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}
$$
> **Readable form:** Estimate the clean image by subtracting predicted noise from the noisy sample and dividing by the remaining signal strength.

Then sample $x_{t-1}$ from DDPM posterior (simplified; Foster uses `noise_rates` / `signal_rates` from schedule):

```python
import tensorflow as tf
import math

T = 1000
diffusion_times = tf.linspace(0.0, 1.0, T)

def reverse_step(x, pred_noise, noise_rate, signal_rate, next_noise_rate, next_signal_rate):
  pred_x0 = (x - noise_rate * pred_noise) / signal_rate
  # DDPM: blend toward next timestep (Foster's implementation details in notebook)
  x_prev = next_signal_rate * pred_x0 + next_noise_rate * tf.random.normal(tf.shape(x))
  return x_prev, pred_x0
```

Final step often omits added noise for sharper output.

---

## Full Sampling Loop

```python
def sample(diffusion_model, n_images=16):
  x = tf.random.normal((n_images, 64, 64, 3))

  noise_rates, signal_rates = offset_cosine_diffusion_schedule(diffusion_times)
  # iterate timesteps from high noise to low
  for i in reversed(range(T)):
    t = diffusion_times[i]
    nr = tf.reshape(noise_rates[i], (1, 1, 1, 1))
    sr = tf.reshape(signal_rates[i], (1, 1, 1, 1))
    pred_noise, _ = diffusion_model.denoise(
      x, nr, sr, training=False  # uses EMA network
    )
    if i > 0:
      nr_next = tf.reshape(noise_rates[i - 1], (1, 1, 1, 1))
      sr_next = tf.reshape(signal_rates[i - 1], (1, 1, 1, 1))
      x, _ = reverse_step(x, pred_noise, nr, sr, nr_next, sr_next)
    else:
      x = (x - nr * pred_noise) / sr
  return tf.clip_by_value(x, -2.0, 2.0)  # after denormalize if needed
```

`training=False` routes through `ema_network` — critical for visual quality.

---

## Denormalization for Display

Training uses `layers.Normalization`. Invert before saving PNGs:

```python
def denormalize(x, normalizer):
  mean = normalizer.mean.numpy()
  var = normalizer.variance.numpy()
  return x * tf.sqrt(var) + mean

samples = sample(diffusion_model, 16)
samples = denormalize(samples, diffusion_model.normalizer)
samples = tf.clip_by_value(samples, 0.0, 1.0)
```

---

## Cost of Sampling

| Steps | Quality | Time (64×64, GPU) |
|-------|---------|-------------------|
| 1000 | Best (Foster) | Minutes per grid |
| 250 | Good | ~4× faster |
| 50 (DDIM) | Usable | Near real-time batch |

PixelCNN: $HW$ sequential passes. GAN: **1** pass. DDPM: **T** passes — the price of stable MLE-style training.

**DDIM** (Song et al., 2020): non-Markovian reverse — skip timesteps with deterministic updates. Foster mentions in [Section 8.8](./section-08-analysis-and-connections.md); production systems use DDIM or distillation to 4–8 steps.

---

## Classifier-Free Guidance Preview

Not in Foster's base notebook but essential in modern systems: train with random dropout of text/label conditioning; at sample time blend conditional and unconditional noise predictions:

$$
\hat{\epsilon} = \hat{\epsilon}_u + s\, (\hat{\epsilon}_c - \hat{\epsilon}_u)
$$
> **Readable form:** Classifier-free guidance starts with unconditional noise and adds a scaled difference toward the conditional noise estimate.

Guidance scale $s > 1$ sharpens prompt alignment — powers Stable Diffusion.

---

## Qualitative Evaluation

Inspect sample grids for:

- **Color diversity** — not all pink roses
- **Petal structure** — not mushy blobs
- **Center symmetry** — flower anatomy
- **Background** — green foliage hints

Compare epoch 20 vs 100 — early samples show color clouds; late samples show distinct species shapes.

---

## Sampling vs Training Noise

Training uses **random** $t$ per image. Sampling walks **all** $t$ sequentially on the **same** image — asymmetric compute. Cannot sample without full loop (unless DDIM/distilled).

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Train network at sample | Blurry / noisy | `training=False` → EMA |
| Wrong schedule order | Abstract noise | Reverse `diffusion_times` |
| Forgot denormalize | Black/white clip | Invert Normalization |
| Too few steps | Blobby flowers | Use T≥250 minimum |

---

## Connection to Other Sections

| Concept | Link |
|---------|------|
| EMA | [Section 8.6](./section-06-training-the-diffusion-model.md) |
| Reverse math | [Section 8.4](./section-04-reverse-diffusion-process.md) |
| Schedules | [Section 8.3](./section-03-reparameterization-and-diffusion-schedules.md) |
| Fast sampling | [Section 8.8](./section-08-analysis-and-connections.md) |

---

## Sampling Checklist

Before declaring DDPM training successful:

- [ ] EMA network loaded (`training=False` in `denoise`)
- [ ] Schedule indices run high-noise → low-noise
- [ ] `normalizer` inverted before `imshow`
- [ ] At least T=250 steps for demo grids (1000 for best)
- [ ] Multiple random seeds — diversity check
- [ ] Side-by-side with real flowers from training set

Save intermediate $x_t$ every 100 steps as a GIF — excellent lab deliverable showing DiffuseTV walk-in effect.

---

## Batch Sampling on GPU

```python
@tf.function
def sample_batch(diffusion_model, n, T):
  x = tf.random.normal((n, 64, 64, 3))
  # ... loop — tf.function may unroll partially; watch compile time
  return x
```

For large `n`, sample in chunks to avoid OOM — each reverse step holds full batch activations in U-Net.

---

## Storing Sample Trajectories

```python
import imageio

frames = []
x = tf.random.normal((1, 64, 64, 3))
for i in reversed(range(0, T, 50)):
  # ... denoise step ...
  frames.append((denormalize(x)[0].numpy() * 255).astype("uint8"))
imageio.mimsave("denoise.gif", frames, fps=4)
```

Animations communicate the DiffuseTV story better than a single final PNG for presentations.

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Reverse sampling** | Iterative $x_T \to x_0$ generation |
| **Posterior step** | $p(x_{t-1} \mid x_t, \hat{x}_0)$ update |
| **DDIM** | Accelerated deterministic sampler |
| **Guidance scale** | Strength of conditional vs unconditional blend |
| **Denoormalize** | Undo training normalization for display |

---

## Reflection Questions

1. Why use the EMA network instead of the freshly trained weights?
2. How does sampling cost compare to a single VAE decoder pass?
3. What visual artifact appears if you use only 50 reverse steps?
4. Why is $x_T$ sampled as standard Gaussian?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 8 — Sampling. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Song, J. et al. (2020). Denoising Diffusion Implicit Models (DDIM). [https://arxiv.org/abs/2010.02502](https://arxiv.org/abs/2010.02502)
- Foster's notebook: `notebooks/08_diffusion/01_ddm/ddm.ipynb`

---

**Previous:** [Section 8.6 — Training the Diffusion Model](./section-06-training-the-diffusion-model.md)  
**Next:** [Section 8.8 — Analysis & Connections](./section-08-analysis-and-connections.md)
