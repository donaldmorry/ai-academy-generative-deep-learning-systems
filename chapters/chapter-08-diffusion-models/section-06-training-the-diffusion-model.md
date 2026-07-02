# Section 8.6: Training the Diffusion Model

> **Source inheritance:** Foster, Ch. 8 — training step / DiffusionModel class  
> **Enhanced with:** Custom train_step, normalization, random timesteps, EMA update, and flower dataset loop  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Foster wraps the [U-Net](./section-05-u-net-denoising-model.md) in a `DiffusionModel` Keras subclass. Each `train_step`: normalize images, sample random diffusion times, corrupt batch with [offset cosine schedule](./section-03-reparameterization-and-diffusion-schedules.md), predict noise, L1 loss, gradient update, **EMA weight sync**.

This is the full DDPM training recipe on Oxford 102 Flowers — 64×64 RGB, batch 64, many epochs until petals emerge from noise.

> **Readable form:** Each batch: random static level, ask U-Net what static was added, penalize wrong guesses, nudge EMA copy toward main network.

---

## DiffusionModel Class Structure

```python
import tensorflow as tf
from tensorflow.keras import layers, models, metrics, optimizers

class DiffusionModel(models.Model):
  def __init__(self, unet, schedule_fn=offset_cosine_diffusion_schedule):
    super().__init__()
    self.normalizer = layers.Normalization()
    self.network = unet
    self.ema_network = models.clone_model(self.network)
    self.ema_network.set_weights(self.network.get_weights())
    self.schedule_fn = schedule_fn
    self.noise_loss_tracker = metrics.Mean(name="noise_loss")

  @property
  def metrics(self):
    return [self.noise_loss_tracker]

  def denoise(self, noisy_images, noise_rates, signal_rates, training):
    net = self.network if training else self.ema_network
    pred_noises = net([noisy_images, noise_rates ** 2], training=training)
    pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
    return pred_noises, pred_images
```

`clone_model` creates EMA twin with identical architecture — weights diverge during training, reunite slowly via EMA.

---

## train_step

```python
  def train_step(self, images):
    images = self.normalizer(images, training=True)
    noises = tf.random.normal(tf.shape(images))

    batch_size = tf.shape(images)[0]
    diffusion_times = tf.random.uniform(
      shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
    )
    noise_rates, signal_rates = self.schedule_fn(diffusion_times)
    noisy_images = signal_rates * images + noise_rates * noises

    with tf.GradientTape() as tape:
      pred_noises, _ = self.denoise(
        noisy_images, noise_rates, signal_rates, training=True
      )
      noise_loss = tf.reduce_mean(tf.abs(noises - pred_noises))

    grads = tape.gradient(noise_loss, self.network.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.network.trainable_weights))
    self.noise_loss_tracker.update_state(noise_loss)

    # EMA update (decay 0.999)
    for w, ema_w in zip(self.network.weights, self.ema_network.weights):
      ema_w.assign(0.999 * ema_w + 0.001 * w)

    return {m.name: m.result() for m in self.metrics}
```

**Key design choices:**

| Choice | Foster value | Why |
|--------|--------------|-----|
| Loss | L1 on $\epsilon$ | Robust to outliers |
| Times | Continuous $[0,1]$ | Smooth schedule coverage |
| Network input | `noise_rates ** 2` | Variance conditioning |
| EMA decay | 0.999 | Stable sampler |

---

## Normalization Layer

Adapt on first batches:

```python
diffusion = DiffusionModel(unet)
# Before fit — adapt on training data
for batch in train.take(50):
  diffusion.normalizer(batch)

diffusion.compile(optimizer=optimizers.Adam(learning_rate=2e-4))
diffusion.fit(train, epochs=100)
```

Maps flower pixels to ~zero mean, unit variance — matches [forward process](./section-02-forward-diffusion-process.md) assumption that $x_0$ has controlled variance before noising.

---

## Training Loop Overview

```
Batch of flowers x0
  → normalize
  → sample t ~ U[0,1], ε ~ N(0,I)
  → x_t = signal(t)*x0 + noise(t)*ε
  → ε_θ = U-Net(x_t, noise(t)²)
  → loss = |ε - ε_θ|
  → backprop U-Net
  → EMA ← 0.999*EMA + 0.001*θ
```

No discriminator, no ELBO, no Langevin — pure supervised regression on noise at random corruption levels. Simplicity is a major reason diffusion scaled to Stable Diffusion.

---

## Monitoring Training

| Signal | Healthy |
|--------|---------|
| `noise_loss` | Down then plateau |
| Sample grids (callback) | Petal structure by epoch 50+ |
| EMA vs train weights | EMA produces sharper samples |

```python
class FlowerSampleCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if epoch % 10 == 0:
      # sample using ema_network — see Section 8.7
      pass
```

GPU memory: 64×64×3 U-Net with batch 64 fits most 8GB cards; reduce batch if OOM.

---

## Hyperparameters (Foster / DDPM)

| Hyperparameter | Typical |
|----------------|---------|
| Image size | 64×64 |
| Batch size | 64 |
| Learning rate | 2e-4 Adam |
| EMA decay | 0.999 |
| T (implicit) | 1000 steps at inference |
| Epochs | 100+ (flowers) |

---

## Comparison to Other Training Loops

| Model | train_step complexity |
|-------|----------------------|
| VAE | Reconstruction + KL |
| GAN | D then G alternating |
| EBM | Langevin + CD |
| RealNVP | NLL + log-det |
| **DDPM** | **Single MSE/L1 on noise** |

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Forgot `normalizer.adapt` | Wrong scale | Adapt before fit |
| Train net at sample | Grainy output | `training=False` → EMA |
| No EMA update | Unstable samples | Loop after each step |
| Discrete $t$ only | Gappy training | Uniform continuous times |

---

## Connection to Other Sections

| Concept | Link |
|---------|------|
| U-Net arch | [Section 8.5](./section-05-u-net-denoising-model.md) |
| Schedules | [Section 8.3](./section-03-reparameterization-and-diffusion-schedules.md) |
| Sampling | [Section 8.7](./section-07-sampling-from-ddpm.md) |
| Custom EBM train | [Section 7.5](../chapter-07-energy-based-models/section-05-training-the-ebm.md) |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **DiffusionModel** | Keras wrapper for training + EMA |
| **Noise loss** | L1/L2 between $\epsilon$ and $\epsilon_\theta$ |
| **Continuous timestep** | $t \in [0,1]$ not only integers |
| **EMA decay** | Smoothing factor for inference weights |
| **Normalizer** | Per-feature mean/var scaling layer |

---

## Reflection Questions

1. Why normalize images before applying the forward diffusion formula?
2. What does the EMA network buy you at sampling time?
3. Why pass `noise_rates ** 2` instead of `noise_rates` to the U-Net?
4. How does this train_step compare to a VAE's reconstruction loss?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 8 — Training. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Ho, J. et al. (2020). DDPM. [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- Foster's notebook: `notebooks/08_diffusion/01_ddm/ddm.ipynb`

---

**Previous:** [Section 8.5 — U-Net Denoising Model](./section-05-u-net-denoising-model.md)  
**Next:** [Section 8.7 — Sampling from DDPM](./section-07-sampling-from-ddpm.md)
