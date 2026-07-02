# Section 3.4: Variational Autoencoders

> **Source inheritance:** Foster, Ch. 3 — "Variational Autoencoders"  
> **Enhanced with:** Probabilistic encoder, ELBO intuition, wardrobe analogy formalized, and comparison to plain autoencoders  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

A plain autoencoder stores each garment at a **single wardrobe coordinate**. A [variational autoencoder (VAE)](../../GLOSSARY.md#variational-autoencoder) stores each garment in a **small cloud** centered near that coordinate, with clouds pulled toward the wardrobe center. Foster's two rule changes — stochastic placement and penalty for straying from center — transform a reconstruction network into a **principled generative model** with a defined prior $p(z) = \mathcal{N}(0, I)$.

The VAE does not just compress; it learns an approximate posterior $q_\phi(z|x)$ that can be sampled and regularized.

---

## From Point Embeddings to Distributions

| Plain autoencoder | Variational autoencoder |
|-------------------|-------------------------|
| $z = f_\phi(x)$ deterministic | $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \mathrm{diag}(\sigma^2_\phi(x)))$ |
| No generative prior | Prior $p(z) = \mathcal{N}(0, I)$ |
| Loss = reconstruction only | Loss = reconstruction + KL divergence |

The encoder outputs two vectors per image:

- $\mu$ — mean of the approximate posterior
- $\log\sigma^2$ — log-variance (ensures $\sigma^2 > 0$)

Sample via the **reparameterization trick** ([Section 3.5](./section-05-vae-loss-and-reparameterization.md)):

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$
> **Readable form:** Pick random noise, scale and shift it to match the encoder's Gaussian cloud.

---

## Why Stochasticity Enforces Continuity

Foster's sandal example: if $(-2, 2)$ decodes well, plain autoencoders do not require $(-2.1, 2.1)$ to look similar. With a VAE, the decoder sees **random perturbations** around $\mu$ during training. To keep reconstruction loss low, nearby points in latent space must decode to similar images.

This is local **Lipschitz-like** smoothness emerging from the training objective — not an explicit architectural constraint.

---

## The ELBO: Generative Objective

The VAE maximizes the **evidence lower bound (ELBO)** on $\log p(x)$:

$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}\big(q_\phi(z|x)\,\|\,p(z)\big)
$$
> **Readable form:** Reconstruct well from samples of z, while keeping the encoder's distribution close to a standard normal.

| Term | Role |
|------|------|
| $\mathbb{E}[\log p_\theta(x|z)]$ | Reconstruction quality (decoder) |
| $D_{\mathrm{KL}}(q\|p)$ | Regularize latent codes toward $\mathcal{N}(0,I)$ |

Maximizing ELBO ≡ minimizing:

$$
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{KL}}
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

Foster implements this directly in Keras — no variational calculus required at implementation time, but the decomposition explains **every knob** you will tune.

---

## KL Divergence: Closed Form for Gaussians

When $q = \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))$ and $p = \mathcal{N}(0, I)$:

$$
D_{\mathrm{KL}}(q\|p) = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
$$
> **Readable form:** KL is zero when each dimension has mean 0 and variance 1.

**What KL fixes from [Section 3.3](./section-03-visualizing-latent-space.md):**

1. **Defined sampling distribution** — draw $z \sim \mathcal{N}(0,I)$ at generation time
2. **Reduced holes** — encodings spread around origin instead of arbitrary far-flung clusters
3. **Balanced occupancy** — less extreme class-volume imbalance (visible in Foster Figure 3-14)

---

## β-VAE Weighting

Foster's Fashion-MNIST VAE uses a reconstruction weight of **500** — a $\beta$-VAE style scaling:

$$
\mathcal{L} = \beta \cdot \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{KL}}
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

| $\beta$ too low | $\beta$ too high |
|-----------------|------------------|
| KL ignored → plain autoencoder pathologies | Blurry reconstructions, underfitting detail |
| Holes return in latent space | Posterior collapses toward prior |

Tuning $\beta$ is dataset- and architecture-specific. CelebA uses $\beta = 2000$ in Foster's faces model ([Section 3.7](./section-07-celeba-faces.md)).

---

## Architecture Changes from Autoencoder to VAE

The **decoder is identical** to [Section 3.1](./section-01-autoencoder-architecture.md). Only the encoder head and loss change:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

class Sampling(layers.Layer):
    """Reparameterization trick: z = mu + sigma * epsilon."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

encoder_input = layers.Input(shape=(32, 32, 1))
x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Flatten()(x)
z_mean = layers.Dense(2, name="z_mean")(x)
z_log_var = layers.Dense(2, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
```

Notebook: `notebooks/03_vae/02_vae_fashion/vae_fashion.ipynb` (adapted from François Chollet's Keras VAE tutorial).

---

## Generative Sampling After VAE Training

```python
# At inference: sample z from prior, decode — no input image needed
z_sample = np.random.normal(size=(16, 2))
generated = decoder.predict(z_sample, verbose=0)
```

Compare to plain autoencoder generation: no min-max bounding box, no uniform guesswork — the prior is **specified**.

Foster's Figure 3-13 shows $z_{\text{mean}}$ values clustered near origin with far fewer incoherent decodes than Figure 3-8.

---

## VAE vs Plain Autoencoder vs GAN (Preview)

| Property | Autoencoder | VAE | GAN (Chapter 04) |
|----------|-------------|-----|-----------------|
| Latent distribution | Undefined | $\mathcal{N}(0,I)$ | $\mathcal{N}(0,I)$ input to generator |
| Training objective | Reconstruction | ELBO | Adversarial game |
| Sample sharpness | Moderate | Often blurry | Often sharp |
| Likelihood | No | Lower bound | Implicit |

Blur is the VAE's famous trade-off: Gaussian decoder assumptions and KL pressure favor average-looking outputs.

---

## Connection to Prior Courses

| Concept | Link |
|---------|------|
| KL divergence | [Course 3, Chapter 03](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-03-probability-information-theory/README.md) |
| Latent variable models | [Course 2, Chapter 13](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-13-probabilistic-reasoning/README.md) |
| Reparameterization | [Section 3.5](./section-05-vae-loss-and-reparameterization.md) |

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Forgetting `log_var` head | Cannot sample | Two Dense outputs from flatten |
| Sampling $\mu$ at inference (no noise) | Less diverse outputs | Sample $z \sim \mathcal{N}(0,I)$ for generation |
| Ignoring $\beta$ tuning | Holes or blur | Track both recon and KL losses separately |
| Expecting GAN sharpness | Disappointment | Different model class — see Chapter 04 |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **VAE** | Generative model with probabilistic encoder and KL-regularized latent space |
| **ELBO** | Evidence lower bound — tractable objective maximizing data likelihood |
| **Approximate posterior** | $q_\phi(z|x)$ — encoder's distribution over latents |
| **Prior** | $p(z)$ — usually $\mathcal{N}(0,I)$ |
| **KL divergence** | Penalty measuring distribution mismatch between $q$ and $p$ |
| **β-VAE** | VAE with weighted KL term for disentanglement / balance |

---

## Reflection Questions

1. In Foster's wardrobe story, what do "area around each item" and "paying Brian for straying" correspond to mathematically?
2. Why does the decoder architecture stay unchanged when upgrading autoencoder → VAE?
3. What three plain-autoencoder sampling problems does the KL term address?
4. Why might a VAE produce blurrier images than a GAN on the same dataset?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 3 — Variational Autoencoders. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
- Foster's codebase: `notebooks/03_vae/02_vae_fashion/vae_fashion.ipynb`
- Higgins, I. et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.

---

**Previous:** [Section 3.3 — Visualizing Latent Space](./section-03-visualizing-latent-space.md)  
**Next:** [Section 3.5 — VAE Loss & Reparameterization](./section-05-vae-loss-and-reparameterization.md)



