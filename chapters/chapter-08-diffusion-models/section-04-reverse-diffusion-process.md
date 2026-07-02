# Section 8.4: Reverse Diffusion Process

> **Source inheritance:** Foster, Ch. 8 — "The Reverse Diffusion Process"  
> **Enhanced with:** Learned denoising, noise prediction objective, VAE analogy, and EMA network preview  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Generation requires reversing the [forward chain](./section-02-forward-diffusion-process.md): start from $x_T \sim \mathcal{N}(0, I)$, iteratively sample $x_{t-1}$ from a learned model $p_\theta(x_{t-1} \mid x_t)$ until you reach $x_0$.

Ho et al. showed this reverse process can be trained by predicting the **noise** $\epsilon$ that was added at each step — equivalent to a simplified variational bound. The neural network $\epsilon_\theta(x_t, t)$ takes the noisy image and timestep, outputs same-shaped noise estimate.

> **Readable form:** Given snowy TV picture and how snowy it should be, predict the static pattern — subtract it to see clearer image.

---

## Reverse as Denoising

Forward (fixed):

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I)
$$
> **Readable form:** the fixed forward process samples the next noisy image from a Gaussian centered at a slightly faded previous image, with variance controlled by beta.

Reverse (learned):

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\bigl(\mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t)\bigr)
$$
> **Readable form:** the learned reverse process predicts a Gaussian distribution for the previous, slightly cleaner image given the current noisy image.

DDPM uses a simplified parameterization: predict $\epsilon$, then reconstruct:

$$
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$
> **Readable form:** subtract the predicted noise component from the noisy image, then divide by the remaining signal strength to estimate the original clean image.

Foster's `denoise` method uses `noise_rates` and `signal_rates` from the [schedule](./section-03-reparameterization-and-diffusion-schedules.md):

```python
def denoise(noisy_images, noise_rates, signal_rates, network, training=True):
  pred_noises = network([noisy_images, noise_rates ** 2], training=training)
  pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
  return pred_noises, pred_images
```

---

## Training Objective (Simplified)

Instead of full ELBO, DDPM optimizes:

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$
> **Readable form:** Mean squared error between true noise and predicted noise.

where $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon$.

Foster uses **mean absolute error** (L1) in Keras — slightly sharper gradients in practice:

```python
loss = tf.reduce_mean(tf.abs(noises - pred_noises))
```

This connects to **score matching**: $\epsilon_\theta \propto -\nabla_{x_t} \log p(x_t)$ under Gaussian perturbations — bridge to [Chapter 07](../chapter-07-energy-based-models/section-07-other-ebms-and-connections.md).

---

## VAE Decoder Analogy

| | VAE decoder | Diffusion reverse |
|---|-------------|-------------------|
| Input | Latent $z \sim \mathcal{N}(0,I)$ | Noisy $x_t$ at many noise levels |
| Output | Image $x$ | Denoised $x_{t-1}$ or $\epsilon$ |
| Forward | Learned stochastic encoder | **Fixed** forward noise |
| Steps | One shot | $T$ iterative steps |

Diffusion is like a VAE decoder that runs at **every noise level** with shared weights — conditioned on how noisy the input is.

---

## EMA Network for Generation

Training network weights fluctuate batch-to-batch. Foster maintains an **exponential moving average (EMA)** copy:

$$
\theta_{\text{EMA}} \leftarrow 0.999\, \theta_{\text{EMA}} + 0.001\, \theta
$$
> **Readable form:** update the smoothed generation weights by keeping almost all of the previous EMA value and adding a small amount of the latest training weights.

Use $\theta_{\text{EMA}}$ at **inference** — smoother denoising, better sample quality. Training uses the fast-updating weights; generation uses EMA ([Section 8.6](./section-06-training-the-diffusion-model.md)).

---

## Reverse Sampling Loop (Preview)

At inference ([Section 8.7](./section-07-sampling-from-ddpm.md)):

```python
# Pseudocode — full loop in Section 8.7
x = tf.random.normal((batch, 64, 64, 3))  # x_T
for t in reversed(range(T)):
  noise_rates, signal_rates = schedule_at(t)
  pred_eps = ema_network([x, noise_rates ** 2], training=False)
  x = denoise_step(x, pred_eps, noise_rates, signal_rates)
```

Each step removes a small amount of predicted noise — walking back up the DiffuseTV aisle.

---

## Why Predict Noise Not $x_0$?

Equivalent reparameterizations exist (predict $x_0$, predict $x_{t-1}$, predict score). Noise prediction:

- Stable gradients across timesteps
- Natural MSE/L1 target
- Matches Ho et al. reference implementation

Some schedulers convert $\epsilon_\theta$ to $\hat{x}_0$ internally for clipping.

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Train net at sample time | Noisy outputs | Use EMA network |
| Wrong noise rate input | Blurry flowers | Pass `noise_rates**2` as Foster |
| MSE only on $x_0$ pred | Unstable | Standard $\epsilon$-prediction |
| No variance in reverse | Over-smooth | DDPM uses fixed $\beta_t$ variance |

---

## Connection to Other Sections

| Concept | Link |
|---------|------|
| Forward $q$ | [Section 8.2](./section-02-forward-diffusion-process.md) |
| Schedules | [Section 8.3](./section-03-reparameterization-and-diffusion-schedules.md) |
| U-Net $\epsilon_\theta$ | [Section 8.5](./section-05-u-net-denoising-model.md) |
| Full train loop | [Section 8.6](./section-06-training-the-diffusion-model.md) |

---

## Equivalence of Prediction Targets

DDPM papers show these training targets are closely related (up to schedule weighting):

| Target | Network predicts | Recover $\hat{x}_0$ |
|--------|------------------|---------------------|
| Noise $\epsilon$ | $\epsilon_\theta(x_t, t)$ | Closed form (Foster `denoise`) |
| Clean $x_0$ | $x_{0,\theta}(x_t, t)$ | Direct |
| Previous $x_{t-1}$ | Mean $\mu_\theta(x_t, t)$ | From posterior formula |
| Score | $\nabla_{x_t} \log p(x_t)$ | $-\epsilon / \sigma_t$ |

Foster standardizes on $\epsilon$-prediction for code clarity — match the reference notebook when comparing loss curves.

---

## Variance in the Reverse Process

Ho et al. fix reverse variance to $\beta_t$ or learned interpolation — Foster uses the schedule-derived `noise_rates` and `signal_rates` rather than exposing a separate variance head. Simpler U-Net (one output tensor, same shape as image) at the cost of less flexible posteriors. Improved DDPM (Nichol & Dhariwal) learns variance — optional upgrade after baseline works.

---

## Debugging Reverse Training

| Symptom | Check |
|---------|-------|
| Loss stuck high | Normalizer adapted? Images zero-mean? |
| Loss NaN | LR too high; clip grads |
| Pred noise all zeros | Dead activations — swish helps |
| Good loss, bad samples | Using train net not EMA at sample time |

---

## Posterior Mean Parameterization (Reference)

The DDPM paper writes the reverse mean as a weighted blend of $\hat{x}_0$ and $x_t$. Foster's `pred_images` from `denoise()` is the $\hat{x}_0$ route — algebraically equivalent to $\epsilon$-prediction when schedules are consistent. Stick to one parameterization in code to avoid mixing formulas from different papers.

---

## Variance Schedule at Sample Time

Use the **same** `offset_cosine_diffusion_schedule` at train and sample — mismatched schedules (train cosine, sample linear) destroy sample quality even with a perfect U-Net. Serialize schedule tensors or function name in checkpoints.

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Reverse process** | Learned $p_\theta$ denoising Markov chain |
| **Noise prediction** | Network target $\epsilon_\theta \approx \epsilon$ |
| **ELBO / VLB** | Variational bound motivating $\epsilon$-loss |
| **EMA weights** | Smoothed copy for stable generation |
| **Denoising step** | One $x_t \to x_{t-1}$ update |

---

## Reflection Questions

1. Why is predicting $\epsilon$ equivalent to denoising $x_t$?
2. How does the reverse process differ from a VAE encoder?
3. Why maintain separate EMA weights?
4. What role does the timestep / noise rate play in $\epsilon_\theta$?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 8 — Reverse Process. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Ho, J. et al. (2020). DDPM. [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)

---

**Previous:** [Section 8.3 — Reparameterization & Schedules](./section-03-reparameterization-and-diffusion-schedules.md)  
**Next:** [Section 8.5 — U-Net Denoising Model](./section-05-u-net-denoising-model.md)

