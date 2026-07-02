# Section 3.6: Training the VAE

> **Source inheritance:** Foster, Ch. 3 — "Training the Variational Autoencoder" & "Analysis of the Variational Autoencoder"  
> **Enhanced with:** Fashion-MNIST training protocol, ELBO monitoring, latent-space before/after comparison  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Training a [VAE](../../GLOSSARY.md#variational-autoencoder) on Fashion-MNIST is your first end-to-end run of the **ELBO optimization loop**: forward pass with stochastic $z$, reconstruction + KL backward pass, repeat. Foster trains five epochs at batch size 100 — modest by modern standards, but enough to see latent space reorganize from the chaotic plain-autoencoder map ([Section 3.3](./section-03-visualizing-latent-space.md)) into a cloud centered near the origin with coherent random samples.

This section focuses on **protocol**: data flow, hyperparameters, checkpoints, and the analysis plots that tell you whether KL is actually working.

---

## End-to-End Training Setup

Prerequisites from [Sections 3.4–3.5](./section-04-variational-autoencoders.md):

```python
import numpy as np
from tensorflow.keras import datasets

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

def preprocess(imgs):
    imgs = imgs.astype("float32") / 255.0
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    return np.expand_dims(imgs, -1)

x_train = preprocess(x_train)
x_test = preprocess(x_test)

# encoder, decoder, VAE class from Section 3.5
vae = VAE(encoder, decoder, beta=500.0)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

history = vae.fit(
    x_train,
    epochs=5,
    batch_size=100,
    shuffle=True,
    validation_data=(x_test, x_test),
)
```

| Hyperparameter | Foster value | Notes |
|----------------|--------------|-------|
| Optimizer | Adam | Default works for small CNN VAE |
| Batch size | 100 | Stable KL estimates |
| Epochs | 5 | Increase if KL still falling |
| $\beta$ | 500 | Reconstruction weight |
| Latent dim | 2 | Visualization-friendly |

---

## What Each Epoch Should Show

**Epoch 1:** High reconstruction loss, moderate KL — encoder learning coarse structure.

**Epochs 2–3:** Recon improves; KL may spike then settle — encodings compress toward prior.

**Epochs 4–5:** Diminishing returns; compare validation curves for overfitting (unlikely at 2D).

```python
print({k: f"{history.history[k][-1]:.4f}" for k in history.history})
```

Track three metrics from custom `train_step`:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{KL}}
$$
> **Readable form:** Total loss is reconstruction (weighted) plus KL — never watch only the sum.

---

## Reconstruction Quality Check

After training, compare VAE reconstructions to plain autoencoder ([Section 3.2](./section-02-reconstructing-images.md)):

```python
vae_recons = vae.decoder(vae.encoder(x_test[:8])[2]).numpy()
# Or full forward:
z_mean, z_log_var, z = vae.encoder.predict(x_test[:8], verbose=0)
recons = vae.decoder.predict(z, verbose=0)
```

VAE reconstructions may be **slightly blurrier** than plain autoencoder at the same 2D bottleneck — KL pulls codes toward a restrictive prior, sacrificing fine detail for global latent structure.

---

## Latent Space Analysis (Foster Figure 3-13)

Plot $z_{\text{mean}}$ (not sampled $z$) for test images:

```python
z_mean, _, _ = vae.encoder.predict(x_test[:5000], verbose=0)

plt.figure(figsize=(8, 8))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test[:5000], cmap="tab10", s=4, alpha=0.6)
plt.xlabel("$z_{\text{mean},1}$")
plt.ylabel("$z_{\text{mean},2}$")
plt.title("VAE latent means colored by class")
plt.show()
```

**Expected improvements vs plain autoencoder:**

| Property | Plain AE | Trained VAE |
|----------|----------|-------------|
| Centering | Arbitrary offset | Clustered near origin |
| Holes at edges | Large | Reduced |
| Random decode quality | Mixed | Mostly recognizable |
| Class balance in plot | Uneven volumes | More uniform (p-value plot, Fig 3-14) |

---

## Generative Sampling from $\mathcal{N}(0,I)$

```python
grid_n = 20
z_sample = np.random.normal(size=(grid_n * grid_n, 2))
generated = vae.decoder.predict(z_sample, verbose=0)

fig, axes = plt.subplots(grid_n, grid_n, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated[i].squeeze(), cmap="gray")
    ax.axis("off")
plt.suptitle("VAE samples from standard normal prior")
plt.tight_layout()
plt.show()
```

Blue dots in Foster Figure 3-13 — sampled $z$ points with decoded images — should mostly land on recognizable clothing.

---

## p-Value Transform Diagnostic

Foster transforms latent coordinates to p-values to check uniformity of class occupancy (Figure 3-14). Intuition: if KL regularization works, no single clothing type monopolizes latent volume.

```python
from scipy import stats

# Per-dimension normality check on z_mean
for dim in range(2):
    stat, p = stats.normaltest(z_mean[:, dim])
    print(f"Dim {dim}: normality test p-value = {p:.4f}")
```

For 200-D CelebA ([Section 3.7](./section-07-celeba-faces.md)), inspect per-dimension histograms — any dimension far from $\mathcal{N}(0,1)$ suggests insufficient KL pressure (lower $\beta$ on reconstruction or raise KL weight).

---

## Saving and Reloading

```python
vae.encoder.save("fashion_vae_encoder.keras")
vae.decoder.save("fashion_vae_decoder.keras")

# Reload for downstream sections
encoder = tf.keras.models.load_model(
    "fashion_vae_encoder.keras",
    custom_objects={"Sampling": Sampling},
)
```

Custom objects (`Sampling`, `VAE`) must be registered on load.

---

## Hyperparameter Experiments

Foster encourages experimentation. Suggested ablations:

| Experiment | Expected effect |
|------------|-----------------|
| $\beta = 50$ | More holes, sharper recon |
| $\beta = 2000$ | Blurrier, tighter Gaussian |
| Latent dim = 8 | Better recon, harder to plot |
| 20 epochs | Diminishing KL gains |

Document **both** sample grids and KL/recon curves for each — qualitative and quantitative.

---

## Bridge to CelebA

Fashion-MNIST at 2D teaches mechanics. Faces require:

- RGB input (3 channels)
- 200-D latent space
- Batch normalization + LeakyReLU
- $\beta = 2000$

Same `VAE` class, deeper encoder-decoder — [Section 3.7](./section-07-celeba-faces.md).

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Plotting sampled $z$ instead of $\mu$ for structure | Noisy scatter | Use `z_mean` for maps |
| Too few epochs | KL still high | Train until KL plateaus |
| Same $\beta$ for faces | Wrong balance | Retune per dataset |
| No `validation_data` | Overfit blind spot | Always hold out test set |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **ELBO training** | Optimizing recon + KL jointly |
| **z_mean plot** | Deterministic encoding map for visualization |
| **Prior sampling** | $z \sim \mathcal{N}(0,I)$ at generation time |
| **Validation ELBO** | Held-out total loss proxy |
| **KL regulatory effect** | Pulling encodings toward origin |

---

## Reflection Questions

1. Why does Foster plot $z_{\text{mean}}$ rather than stochastic $z$ for latent scatter plots?
2. What would validation reconstruction loss rising while KL falls indicate?
3. How does Figure 3-13 differ from Figure 3-7 in terms of sampling viability?
4. Which hyperparameter would you adjust first if random samples still look like noise after 5 epochs?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 3 — Training & Analysis. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Foster's codebase: `notebooks/03_vae/02_vae_fashion/vae_fashion.ipynb`
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.

---

**Previous:** [Section 3.5 — VAE Loss & Reparameterization](./section-05-vae-loss-and-reparameterization.md)  
**Next:** [Section 3.7 — CelebA Faces](./section-07-celeba-faces.md)



