# Section 3.5: VAE Loss & Reparameterization

> **Source inheritance:** Foster, Ch. 3 — "The Loss Function", "The Reparameterization Trick", "Training the Variational Autoencoder"  
> **Enhanced with:** Custom `train_step`, KL closed form, GradientTape, and loss balancing  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Backpropagation through randomness sounds impossible — until you **repackage** it. The reparameterization trick writes $z = \mu + \sigma \odot \epsilon$ with $\epsilon$ as the only random source, making gradients w.r.t. $\mu$ and $\sigma$ deterministic. Combined with a custom Keras `train_step` that sums weighted reconstruction BCE and analytic KL, you get a fully trainable [VAE](../../GLOSSARY.md#variational-autoencoder) without leaving TensorFlow.

This section is the implementation bridge between [Section 3.4](./section-04-variational-autoencoders.md) theory and [Section 3.6](./section-06-training-the-vae.md) training runs.

---

## Reparameterization Trick

Direct sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ blocks gradients to $\mu, \sigma$. Instead:

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

$$
\sigma = \exp\left(\tfrac{1}{2} \log \sigma^2\right)
$$
> **Readable form:** Draw standard normal noise, then scale and shift — randomness isolated in epsilon.

**Why log-variance?** Variance must be positive. The network outputs `z_log_var` $\in (-\infty, \infty)$; exponentiation recovers $\sigma^2 > 0$.

```python
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
```

Foster's derivation: $\sigma = \exp(\log\sigma^2 / 2)$ because $\log\sigma^2 = 2\log\sigma$.

---

## Reconstruction Loss

For Fashion-MNIST with sigmoid decoder outputs, Foster uses **binary cross-entropy** summed over spatial dimensions, then averaged over batch:

$$
\mathcal{L}_{\text{recon}} = -\frac{1}{B}\sum_{b=1}^{B}\sum_{i}\left[x_{b,i}\log\hat{x}_{b,i} + (1-x_{b,i})\log(1-\hat{x}_{b,i})\right]
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

In code (with $\beta$ scaling factor 500):

```python
reconstruction_loss = tf.reduce_mean(
    500 * tf.keras.losses.binary_crossentropy(data, reconstruction, axis=(1, 2, 3))
)
```

The factor 500 is **not** sacred — it balances reconstruction against KL on this architecture. Without scaling, KL often dominates or is ignored.

---

## KL Divergence Loss (Analytic)

For diagonal Gaussian $q$ and standard normal $p$:

$$
\mathcal{L}_{\text{KL}} = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

Per-sample, then batch-averaged:

```python
kl_loss = tf.reduce_mean(
    tf.reduce_sum(
        -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
        axis=1,
    )
)
total_loss = reconstruction_loss + kl_loss
```

> **Readable form:** Push each latent dimension toward mean 0, variance 1.

**Monitoring both terms separately** is essential — a falling `total_loss` can hide KL collapse or reconstruction neglect.

---

## Custom VAE Model Class

Foster subclasses `keras.Model` to inject KL into `train_step`:

```python
from tensorflow.keras import models, metrics

class VAE(models.Model):
    def __init__(self, encoder, decoder, beta=500.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data, training=True)
            reconstruction_loss = tf.reduce_mean(
                self.beta * tf.keras.losses.binary_crossentropy(
                    data, reconstruction, axis=(1, 2, 3)
                )
            )
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(
                    -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                    axis=1,
                )
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}
```

**GradientTape** records operations for `tape.gradient()` — same mechanism Foster previews for GAN training in Chapter 4.

---

## Wiring and Compilation

```python
# decoder built as in Section 3.1 (identical architecture)
vae = VAE(encoder, decoder, beta=500.0)
vae.compile(optimizer="adam")
vae.fit(x_train, epochs=5, batch_size=100, validation_data=(x_test, x_test))
```

Note: `fit` targets are still `x` itself — the VAE is self-supervised. KL needs no labels.

---

## Interpreting Loss Curves

| Curve pattern | Diagnosis |
|---------------|-------------|
| `kl_loss` → 0 immediately | Posterior collapse — raise $\beta$ or reduce decoder capacity |
| `reconstruction_loss` flat, high | Under-training or LR too low |
| `kl_loss` large, blurry images | $\beta$ too low — latent unstructured |
| Both decreasing smoothly | Healthy VAE training |

```python
history = vae.fit(x_train, epochs=20, batch_size=100, validation_data=(x_test, x_test))
import matplotlib.pyplot as plt
for key in ["total_loss", "reconstruction_loss", "kl_loss"]:
    plt.plot(history.history[key], label=key)
plt.legend()
plt.xlabel("Epoch")
plt.show()
```

---

## Inference: Encode vs Generate

**Reconstruction path** — use sampled $z$ during training; at eval can use $\mu$ only for sharper reconstructions:

```python
z_mean, z_log_var, z = encoder.predict(x_test[:1], verbose=0)
recon_from_mu = decoder.predict(z_mean, verbose=0)
```

**Generation path** — always sample prior:

```python
z_new = np.random.normal(size=(16, 2))
images = decoder.predict(z_new, verbose=0)
```

Mixing these up is a common bug: using $\mu$ for generation reduces diversity.

---

## Subclassing `Layer` vs `Model`

| Pattern | When to use |
|---------|-------------|
| `Sampling` as `Layer` | Differentiable transform in graph |
| `VAE` as `Model` | Custom `train_step` with multiple losses |
| Functional API only | Insufficient for KL without `add_loss` hacks |

Foster's approach (subclass `Model`) is the clearest for teaching and debugging.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| `exp(z_log_var)` without 0.5 factor | Wrong variance scale | Use `exp(0.5 * z_log_var)` for $\sigma$ |
| KL sum over wrong axis | Scalar shape errors | Sum over latent dim, mean over batch |
| No $\beta$ tuning | Holes or blur | Grid-search $\beta \in \{100, 500, 1000\}$ |
| `add_loss` + `compile` confusion | KL not in gradients | Prefer custom `train_step` |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Reparameterization trick** | Express $z$ as deterministic function of $\epsilon$ for backprop |
| **z_log_var** | Network output encoding $\log\sigma^2$ per latent dimension |
| **GradientTape** | TF context recording ops for automatic differentiation |
| **train_step** | Keras hook for custom training logic per batch |
| **Posterior collapse** | KL → 0; encoder ignores latent (ignores input structure) |
| **β factor** | Weight on reconstruction term balancing KL |

---

## Reflection Questions

1. Why must randomness be confined to $\epsilon$ for backpropagation through the Sampling layer?
2. What happens to KL loss when `z_mean = 0` and `z_log_var = 0` for all dimensions?
3. Why does Foster multiply reconstruction loss by 500 but not KL loss?
4. When would you use `z_mean` instead of sampled `z` at inference?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 3 — Loss Function & Training. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
- Chollet, F. Keras VAE example: [https://keras.io/examples/generative/vae/](https://keras.io/examples/generative/vae/)
- Foster's codebase: `notebooks/03_vae/02_vae_fashion/vae_fashion.ipynb`

---

**Previous:** [Section 3.4 — Variational Autoencoders](./section-04-variational-autoencoders.md)  
**Next:** [Section 3.6 — Training the VAE](./section-06-training-the-vae.md)
