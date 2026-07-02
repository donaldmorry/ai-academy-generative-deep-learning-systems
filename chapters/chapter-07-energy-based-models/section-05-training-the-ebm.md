# Section 7.5: Training the EBM

> **Source inheritance:** Foster, Ch. 7 — "EBM trained using contrastive divergence"  
> **Enhanced with:** Custom Keras Model, train_step/test_step, metrics, and full MNIST training loop  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Foster wraps the [energy network](./section-02-mnist-energy-function.md), [Langevin sampler](./section-03-langevin-dynamics-sampling.md), and [replay buffer](./section-04-contrastive-divergence.md) in a custom `tf.keras.Model` subclass with overridden `train_step` and `test_step`. Standard `model.compile(loss=...)` cannot express CD — you need explicit control over positive/negative phases.

This section walks through the `EBM` class, metric tracking, and a full `fit()` on MNIST.

> **Readable form:** Each batch: jitter reals, Langevin-fake negatives from buffer, contrastive loss, update weights.

---

## EBM Wrapper Class

```python
import tensorflow as tf
from tensorflow.keras import models, metrics, optimizers

class EBM(models.Model):
  def __init__(self, energy_model):
    super().__init__()
    self.model = energy_model
    self.buffer = Buffer(self.model)
    self.alpha = 0.1
    self.loss_metric = metrics.Mean(name="loss")
    self.reg_loss_metric = metrics.Mean(name="reg")
    self.cdiv_loss_metric = metrics.Mean(name="cdiv")
    self.real_out_metric = metrics.Mean(name="real")
    self.fake_out_metric = metrics.Mean(name="fake")

  @property
  def metrics(self):
    return [
      self.loss_metric, self.reg_loss_metric, self.cdiv_loss_metric,
      self.real_out_metric, self.fake_out_metric,
    ]
```

Separate metrics let you monitor whether `real` energy is dropping and `fake` energy rising — the health check for CD training.

---

## train_step Implementation

```python
  def train_step(self, real_imgs):
    # Slight noise on reals (data augmentation)
    real_imgs += tf.random.normal(tf.shape(real_imgs), mean=0, stddev=0.005)
    real_imgs = tf.clip_by_value(real_imgs, -1.0, 1.0)

    # Negative samples via buffer + Langevin
    fake_imgs = self.buffer.sample_new_exmps(
      steps=60, step_size=10, noise=0.005
    )

    inp_imgs = tf.concat([real_imgs, fake_imgs], axis=0)

    with tf.GradientTape() as tape:
      energies = self.model(inp_imgs)
      real_out, fake_out = tf.split(energies, 2, axis=0)
      cdiv_loss = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out)
      reg_loss = self.alpha * tf.reduce_mean(real_out ** 2 + fake_out ** 2)
      loss = reg_loss + cdiv_loss

    grads = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    self.loss_metric.update_state(loss)
    self.reg_loss_metric.update_state(reg_loss)
    self.cdiv_loss_metric.update_state(cdiv_loss)
    self.real_out_metric.update_state(tf.reduce_mean(real_out))
    self.fake_out_metric.update_state(tf.reduce_mean(fake_out))
    return {m.name: m.result() for m in self.metrics}
```

**Batch size 128:** 128 reals + 128 fakes concatenated → split energies down the middle.

---

## test_step (Simplified Evaluation)

```python
  def test_step(self, real_imgs):
    batch_size = tf.shape(real_imgs)[0]
    fake_imgs = tf.random.uniform((batch_size, 32, 32, 1)) * 2 - 1
    inp_imgs = tf.concat([real_imgs, fake_imgs], axis=0)
    real_out, fake_out = tf.split(self.model(inp_imgs), 2, axis=0)
    cdiv = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out)
    self.cdiv_loss_metric.update_state(cdiv)
    self.real_out_metric.update_state(tf.reduce_mean(real_out))
    self.fake_out_metric.update_state(tf.reduce_mean(fake_out))
    return {m.name: m.result() for m in self.metrics}
```

Validation uses **random noise** fakes (no Langevin) for speed — a weaker but fast sanity check. Large `fake - real` gap on val indicates generalization.

---

## Launching Training

```python
ebm = EBM(energy_model)
ebm.compile(optimizer=optimizers.Adam(learning_rate=1e-4))

history = ebm.fit(
  train_ds,
  epochs=50,
  validation_data=test_ds,
)

# Save energy weights
energy_model.save_weights("ebm_mnist.weights.h5")
```

Foster trains tens of epochs; each epoch is **slow** because every batch runs 60 Langevin steps × 128 images. Use GPU; expect minutes per epoch on CPU.

---

## Reading the Metrics

| Metric | Healthy trend |
|--------|----------------|
| `real` | Decreasing (more negative) |
| `fake` | Increasing (more positive) |
| `cdiv` | Positive and stable/growing |
| `reg` | Small relative to cdiv |
| `loss` | Overall decrease early |

**Red flags:**

- `real` and `fake` converge → model not contrasting
- `fake` below `real` → inverted loss or broken Langevin
- `reg` dominates → reduce $\alpha$ or check energy scale

---

## Checkpointing and Sample Callback

```python
class SampleCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    x0 = tf.random.uniform((16, 32, 32, 1)) * 2 - 1
    samples, _ = generate_samples(
      self.model.model, x0, steps=200, step_size=10, noise=0.005
    )
    # save grid — more steps than training for prettier visuals
```

Training uses 60 Langevin steps; visualization may use 200 for clearer digits.

---

## Training Cost Breakdown

Per batch:

1. Forward energy on 256 images (128+128)
2. Backward through energy net
3. 60 × forward + backward through net for Langevin (expensive!)

EBM training is among the **slowest** in Part II — comparable to long PixelCNN epochs, worse per-batch than GAN/VAE.

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| `compile(loss=...)` only | CD never runs | Custom `train_step` |
| LR too high | Oscillating energies | 1e-4 Adam |
| Batch size 32 | Noisy CD signal | Use 128 if memory allows |
| Skipping validation | Overfit unnoticed | `validation_data=test_ds` |

---

## Connection to Other Sections

| Concept | Link |
|---------|------|
| CD math | [Section 7.4](./section-04-contrastive-divergence.md) |
| Analysis plots | [Section 7.6](./section-06-analysis-of-the-ebm.md) |
| Custom GAN train_step | [Chapter 04](../chapter-04-generative-adversarial-networks/section-03-training-the-dcgan.md) |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Custom train_step** | Override Keras training logic |
| **Contrastive loss** | $\mathbb{E}[E(fake)] - \mathbb{E}[E(real)]$ |
| **Energy regularization** | Penalty $\alpha E^2$ for stability |
| **Metric tracking** | Separate real/fake energy means |
| **Persistent negatives** | Buffer-fed Langevin samples |

---

## Reflection Questions

1. Why can't standard `fit(x, y)` work for EBMs?
2. What is the most expensive operation inside `train_step`?
3. Why add noise to real images before scoring?
4. How would you early-stop based on `cdiv` vs `loss`?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 7 — Training EBM. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Foster's notebook: `notebooks/07_ebm/01_ebm/ebm.ipynb`

---

**Previous:** [Section 7.4 — Contrastive Divergence](./section-04-contrastive-divergence.md)  
**Next:** [Section 7.6 — Analysis of the EBM](./section-06-analysis-of-the-ebm.md)
