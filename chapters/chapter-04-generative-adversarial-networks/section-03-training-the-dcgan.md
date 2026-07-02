# Section 4.3: Training the DCGAN

> **Source inheritance:** Foster, Ch. 4 — "Training the DCGAN"  
> **Enhanced with:** Custom `train_step`, alternating updates, label smoothing, and sample monitoring  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

DCGAN architectures from [Section 4.2](./section-02-dcgan-architecture.md) are straightforward — the **training loop** is where [GANs](../../GLOSSARY.md#generative-adversarial-network-gan) live or die. Foster's `DCGAN` model class alternates discriminator and generator updates inside one `train_step`, freezes the inactive network's weights via separate `GradientTape` contexts, and adds **label noise** so the discriminator cannot overpower the generator too quickly.

Understanding gray-box diagrams (Figure 4-5) — which weights are frozen when — is essential before debugging mode collapse in [Section 4.4](./section-04-gan-tips-and-tricks.md).

---

## Loss Functions

Both networks use `BinaryCrossentropy`:

**Discriminator** (minimize):

$$
\mathcal{L}_D = -\frac{1}{2}\mathbb{E}[\log D(x)] - \frac{1}{2}\mathbb{E}[\log(1 - D(G(z)))]
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

**Generator** (minimize):

$$
\mathcal{L}_G = -\mathbb{E}[\log D(G(z))]
$$
> **Readable form:** D learns real=1, fake=0; G learns to make D say fake=1.

In Keras, the generator is trained by comparing `fake_predictions` to `ones_like` — pushing fakes toward "real."

---

## Label Smoothing / Noisy Labels

Foster adds uniform noise to labels:

```python
real_labels = tf.ones_like(real_predictions)
real_noisy = real_labels + 0.1 * tf.random.uniform(tf.shape(real_predictions))
fake_labels = tf.zeros_like(fake_predictions)
fake_noisy = fake_labels - 0.1 * tf.random.uniform(tf.shape(fake_predictions))
```

| Label type | Target | Effect |
|------------|--------|--------|
| Real | $1 + \mathcal{U}(0, 0.1)$ | Prevents overconfident D |
| Fake | $0 - \mathcal{U}(0, 0.1)$ | Softens decision boundary |

This **label smoothing** tames the discriminator — a practical stabilizer when D loss collapses toward zero.

---

## Full DCGAN Model Class

```python
import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers, models

class DCGAN(models.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.loss_fn = losses.BinaryCrossentropy()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_metric = metrics.Mean(name="d_loss")
        self.g_loss_metric = metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal((batch_size, self.latent_dim))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            real_predictions = self.discriminator(real_images, training=True)
            fake_predictions = self.discriminator(generated_images, training=True)

            real_noisy = tf.ones_like(real_predictions) + 0.1 * tf.random.uniform(tf.shape(real_predictions))
            fake_noisy = tf.zeros_like(fake_predictions) - 0.1 * tf.random.uniform(tf.shape(fake_predictions))

            d_real_loss = self.loss_fn(real_noisy, real_predictions)
            d_fake_loss = self.loss_fn(fake_noisy, fake_predictions)
            d_loss = (d_real_loss + d_fake_loss) / 2.0

            g_loss = self.loss_fn(tf.ones_like(fake_predictions), fake_predictions)

        d_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = gen_tape.gradient(g_loss, self.generator.trainable_variables)

        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {m.name: m.result() for m in self.metrics}
```

**Key detail:** Both tapes record in one forward pass, but gradients apply to **separate** variable lists — only D weights get `d_grads`, only G weights get `g_grads`.

---

## Compilation and Training

Foster uses Adam with non-default $\beta_1$:

```python
dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=100)

dcgan.compile(
    d_optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999),
    g_optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999),
)

dcgan.fit(train, epochs=300)
```

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Learning rate | 0.0002 | DCGAN standard |
| $\beta_1$ | 0.5 | Less momentum than default 0.9 |
| Epochs | 300 | Long run — log samples periodically |
| Batch size | 128 | From `image_dataset_from_directory` |

---

## Monitoring Training

**Do not trust generator loss alone** — Foster Figure 4-6 shows G loss **rising** while image quality improves. The discriminator constantly evolves; G loss is relative.

Better monitoring:

```python
import matplotlib.pyplot as plt

class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, num_img=16):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        z = tf.random.normal((self.num_img, 100))
        samples = self.model.generator(z, training=False)
        # save grid to ./output/epoch_XXX.png
```

| Metric | Interpretation |
|--------|----------------|
| `d_loss` → 0 | D winning — risk vanishing G gradients |
| `d_loss` ≈ 0.69 | Balanced (random guessing) |
| `g_loss` trend | Not directly comparable across epochs |
| Visual samples | **Ground truth** for quality |

---

## Epoch-by-Epoch Sample Quality

Foster Figure 4-7: early epochs = noise blobs; mid training = brick-like shapes; late = shadows, studs, 3D perspective. The generator learns **high-level graphics concepts** from pixels alone — no explicit 3D model.

---

## Memorization Check

Good generative models produce **novel** samples. Foster compares L1 distance to nearest training image:

```python
def compare_images(img1, img2):
    return np.mean(np.abs(img1 - img2))
```

Figure 4-8: generated bricks resemble training set but are not copies — G learned features, not lookup.

---

## Training Dynamics Diagram

```
Batch of real images ──> D ──> d_loss (update D only)
                z ~ N(0,I) ──> G ──> fake ──> D ──> d_loss / g_loss
```

Gray boxes in Foster Figure 4-5 = frozen weights during that sub-step.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Updating both nets with combined loss | D cheats / G ignored | Separate tapes and optimizers |
| No label noise | D dominates instantly | Add 0.1 uniform noise |
| Judging by g_loss only | False alarms | Plot sample grids |
| Too few epochs | Noise bricks | Train 100+ epochs minimum |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Alternating training** | Update D and G in separate optimizer steps |
| **Label smoothing** | Noisy real/fake targets softening D |
| **Vanishing gradients** | G receives no useful signal when D is perfect |
| **GANMonitor** | Callback saving sample images per epoch |
| **Implicit training** | No pixel-wise target for G — only D feedback |

---

## Reflection Questions

1. Why does Foster add noise to labels instead of using hard 0/1?
2. Why can generator loss increase while samples improve?
3. What does `disc_tape.gradient` exclude from generator variables?
4. How does Foster verify the model is not just memorizing training images?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 4 — Training the DCGAN. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Foster's codebase: `notebooks/04_gan/01_dcgan/dcgan.ipynb`
- Goodfellow, I. et al. (2014). Generative Adversarial Nets.

---

**Previous:** [Section 4.2 — DCGAN Architecture](./section-02-dcgan-architecture.md)  
**Next:** [Section 4.4 — GAN Tips and Tricks](./section-04-gan-tips-and-tricks.md)
