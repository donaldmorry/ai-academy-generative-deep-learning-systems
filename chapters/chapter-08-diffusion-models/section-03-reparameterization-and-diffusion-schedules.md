# Section 8.3: Reparameterization & Diffusion Schedules

> **Source inheritance:** Foster, Ch. 8 — "The Reparameterization Trick" / "Diffusion Schedules"  
> **Enhanced with:** Jump sampling $x_t$, linear vs cosine schedules, signal/noise rates, and Foster's offset cosine  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Training needs noisy images $x_t$ at **random** timesteps without simulating $t$ sequential noising steps. The **reparameterization trick** (same spirit as VAEs) writes:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$
> **Readable form:** Build a noisy image at timestep $t$ in one step by mixing the clean image with Gaussian noise according to the signal/noise schedule.

The **diffusion schedule** chooses how $\bar{\alpha}_t$ (or $\beta_t$) evolves from $t=1$ to $T$. Schedule choice strongly affects training stability and sample quality — Foster implements **linear**, **cosine**, and **offset cosine** schedules.

> **Readable form:** Pick how fast signal dies and noise grows across 1000 steps — cosine dies slower at first, works better than linear.

---

## Deriving the Closed Form

Start from $x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1-\alpha_t}\, \epsilon_{t-1}$ with $\alpha_t = 1 - \beta_t$. Unrolling:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \bar{\epsilon}
$$
> **Readable form:** A noisy sample is a weighted mixture of clean image signal and accumulated Gaussian noise.

where $\bar{\epsilon}$ is a merged Gaussian (sum of independent normals). Hence $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\, x_0,\, (1-\bar{\alpha}_t)I)$.

**Signal rate:** $\sqrt{\bar{\alpha}_t}$ — fraction of $x_0$ retained.  
**Noise rate:** $\sqrt{1-\bar{\alpha}_t}$ — fraction of pure noise.

Foster's code uses `signal_rates` and `noise_rates` directly (squared forms in some implementations).

---

## Linear Schedule (Ho et al.)

$$
\beta_t \text{ linear from } 0.0001 \text{ to } 0.02 \text{ over } T=1000
$$
> **Readable form:** The linear schedule increases per-step noise variance steadily from a tiny value to a larger value across all timesteps.

```python
import tensorflow as tf

T = 1000

def linear_diffusion_schedule(diffusion_times):
  min_rate = 0.0001
  max_rate = 0.02
  betas = min_rate + tf.convert_to_tensor(diffusion_times) * (max_rate - min_rate)
  alphas = 1.0 - betas
  alpha_bars = tf.math.cumprod(alphas)
  signal_rates = tf.sqrt(alpha_bars)
  noise_rates = tf.sqrt(1.0 - alpha_bars)
  return noise_rates, signal_rates

diffusion_times = tf.linspace(0.0, 1.0, T)
linear_noise, linear_signal = linear_diffusion_schedule(diffusion_times)
```

Small $\beta$ early → gentle initial corruption. Larger $\beta$ late → finish destroying signal when image is already noisy.

---

## Cosine Schedule

Nichol & Dhariwal (2021) proposed slower noise ramp at the start:

$$
\bar{\alpha}_t \propto \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)
$$
> **Readable form:** The score compares vector representations; closer or more aligned vectors receive a better match score.

Foster's simplified **pure cosine** for visualization:

```python
import math

def cosine_diffusion_schedule(diffusion_times):
  signal_rates = tf.cos(diffusion_times * math.pi / 2.0)
  noise_rates = tf.sin(diffusion_times * math.pi / 2.0)
  return noise_rates, signal_rates
```

Uses identity $\cos^2 + \sin^2 = 1$ so signal² + noise² = 1 when squared rates sum to 1.

---

## Offset Cosine (Foster's Training Default)

Prevents tiny noise steps at $t \approx 0$ that waste capacity:

```python
def offset_cosine_diffusion_schedule(diffusion_times):
  min_signal_rate = 0.02
  max_signal_rate = 0.95
  start_angle = tf.acos(max_signal_rate)
  end_angle = tf.acos(min_signal_rate)
  diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
  signal_rates = tf.cos(diffusion_angles)
  noise_rates = tf.sin(diffusion_angles)
  return noise_rates, signal_rates
```

**Training** in [Section 8.6](./section-06-training-the-diffusion-model.md) samples continuous `diffusion_times` in $[0,1]$ and passes through this schedule — not discrete integer $t$ only.

---

## Comparing Schedules

| Schedule | Early noise | Late noise | Foster use |
|----------|-------------|------------|------------|
| Linear | Moderate | Aggressive | Baseline (Ho) |
| Cosine | **Gentle** | Moderate | Better FID |
| Offset cosine | Tuned start/end | Tuned | **Training default** |

Figure 8-4 in Foster plots signal vs noise across $t$. Cosine keeps more signal longer — corrupted images at mid-$t$ retain more structure (Figure 8-5), helping the network learn coarse layout before fine detail.

---

## Training-Time Usage

```python
def corrupt(x0, diffusion_times):
  noise_rates, signal_rates = offset_cosine_diffusion_schedule(diffusion_times)
  # broadcast rates to (batch, 1, 1, 1) for images
  noise_rates = noise_rates[:, None, None, None]
  signal_rates = signal_rates[:, None, None, None]
  eps = tf.random.normal(tf.shape(x0))
  xt = signal_rates * x0 + noise_rates * eps
  return xt, eps, noise_rates, signal_rates
```

Sample `diffusion_times ~ Uniform(0,1)` per image in batch — each example sees a random noise level (continuous-time flavor).

---

## Connection to VAE Reparameterization

| VAE | Diffusion |
|-----|-----------|
| $z = \mu + \sigma \odot \epsilon$ | $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon$ |
| Learn $\mu, \sigma$ | Fixed schedule |
| Low-D $z$ | Same shape as $x$ |

Both enable backprop through stochastic nodes by expressing samples as deterministic functions of $\epsilon$.

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| `alpha_bar` not cumprod | Wrong noise level | `tf.math.cumprod(alphas)` |
| Integer $t$ only | Coarse training | Continuous times in $[0,1]$ |
| Linear schedule only | Worse flowers | Use offset cosine |
| Forgetting broadcast | Shape error | Expand rates to 4D |

---

## Signal-to-Noise Ratio View

Define SNR at step $t$:

$$
\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t} = \frac{\text{signal}^2}{\text{noise}^2}
$$
> **Readable form:** Signal-to-noise ratio compares remaining clean signal power against accumulated noise power at timestep $t$.

Plot SNR vs $t$ for linear vs offset cosine — cosine maintains higher SNR early, so mid-corruption images retain global flower layout longer, giving the U-Net easier coarse structure to latch onto before fine denoising.

---

## Discrete vs Continuous Time

Foster samples `diffusion_times ~ U(0,1)` continuously — not only integers $\{1,\ldots,T\}$. Benefits:

- Smoother coverage during training
- Single schedule function for any resolution of inference steps
- Matches improved DDPM practice and score-SDE formulations

At inference you still discretize into $T$ bins — training-time continuity does not force integer $t$ per batch.

---

## Plotting Schedules for Your Report

```python
import matplotlib.pyplot as plt

t = tf.linspace(0., 1., 500)
for name, fn in [("linear", linear_diffusion_schedule), ("offset_cosine", offset_cosine_diffusion_schedule)]:
  nr, sr = fn(t[:, None, None, None])
  plt.plot(t, sr[:,0,0,0]**2, label=f"{name} signal²")
plt.legend(); plt.xlabel("diffusion time"); plt.ylabel("signal fraction²")
plt.savefig("schedule_compare.png")
```

Include this figure in Chapter 08 lab — examiners expect you to justify Foster's offset cosine default over linear.

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **$\bar{\alpha}_t$** | Cumulative product of $(1-\beta_s)$ |
| **Diffusion schedule** | How $\beta_t$ or signal/noise rates vary with $t$ |
| **Signal rate** | Weight on $x_0$ in $x_t$ |
| **Noise rate** | Weight on $\epsilon$ in $x_t$ |
| **Offset cosine** | Schedule with clamped signal at endpoints |

---

## Reflection Questions

1. Why is sampling $x_t$ directly faster than $t$ sequential noise steps?
2. How does a cosine schedule change mid-training corruption visually?
3. What problem does the offset term solve at $t \approx 0$?
4. Why are `noise_rates` fed to the U-Net alongside $x_t$?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 8 — Reparameterization, Schedules. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Nichol, A. Q. & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. [https://arxiv.org/abs/2102.09672](https://arxiv.org/abs/2102.09672)

---

**Previous:** [Section 8.2 — Forward Diffusion Process](./section-02-forward-diffusion-process.md)  
**Next:** [Section 8.4 — Reverse Diffusion Process](./section-04-reverse-diffusion-process.md)
