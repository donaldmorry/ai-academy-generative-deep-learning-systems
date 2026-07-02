# Section 3.3: Visualizing Latent Space

> **Source inheritance:** Foster, Ch. 3 — "Visualizing the Latent Space" & "Generating New Images"  
> **Enhanced with:** Label-colored scatter plots, uniform sampling pitfalls, and grid-overlay diagnostics  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

A 2D [latent space](../../GLOSSARY.md#latent-variable) is not just a compression trick — it is a **map** you can draw. Foster embeds 5,000 Fashion-MNIST test images through the encoder and plots each $(z_1, z_2)$ as a point. Unlabeled, you see a black cloud. Color by clothing type and structure **explodes** into view: trousers cluster bottom-right, ankle boots center-left, bags occupy a broad orange region.

Visualization turns an abstract bottleneck into evidence about what the network learned — and exposes why naive sampling from that space fails as generation.

---

## Embedding the Test Set

```python
import numpy as np
import matplotlib.pyplot as plt

example_images = x_test[:5000]
example_labels = y_test[:5000]  # labels NOT used in training — only for plotting

embeddings = encoder.predict(example_images, verbose=0)

plt.figure(figsize=(8, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c="black", alpha=0.5, s=3)
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.title("Fashion-MNIST latent space (unlabeled)")
plt.gca().set_aspect("equal")
plt.show()
```

Each point is $z = f_\phi(x)$ — the encoder's compression of one clothing image.

> **Readable form:** Every dot is a garment squeezed into two numbers.

---

## Fashion-MNIST Labels as a Diagnostic

Foster uses the 10 class labels purely for **post-hoc** analysis:

| ID | Label | Typical latent region |
|----|-------|----------------------|
| 0 | T-shirt/top | Mixed, overlaps shirts |
| 1 | Trouser | Compact cluster, bottom-right |
| 2 | Pullover | Near coats/shirts |
| 3 | Dress | Distinct from trousers |
| 4 | Coat | Near pullovers |
| 5 | Sandal | Footwear group |
| 6 | Shirt | Overlaps tops |
| 7 | Sneaker | Footwear |
| 8 | Bag | Large spread |
| 9 | Ankle boot | Tight red cluster |

```python
FASHION_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

plt.figure(figsize=(10, 10))
scatter = plt.scatter(
    embeddings[:, 0], embeddings[:, 1],
    c=example_labels, cmap="tab10", alpha=0.6, s=4,
)
plt.colorbar(scatter, ticks=range(10), label="Class ID")
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.title("Latent space colored by clothing type")
plt.show()
```

The model never saw labels during `autoencoder.fit(x_train, x_train)`. Clustering by appearance is **emergent** — proof the encoder captures semantically meaningful factors.

---

## Three Observations That Break Naive Sampling

Foster identifies structural problems visible only after coloring and grid-overlay (Figures 3-7 through 3-9):

**1. Unequal class occupancy.** Bags span a much larger latent region than ankle boots. Uniform sampling over a bounding box oversamples bags.

**2. Undefined global distribution.** Embeddings are not centered at $(0,0)$, not symmetric, and extend to $y > 8$ in Foster's plot. There is no obvious "default" sampling box.

**3. Holes and discontinuities.** Large empty regions exist — especially at domain edges. The autoencoder never trained the decoder on those coordinates.

Mathematically, plain autoencoders learn an encoding map $f_\phi: \mathcal{X} \to \mathbb{R}^d$ but do **not** learn a generative density $p(z)$. Sampling requires you to guess a distribution over $z$ — usually wrongly.

$$
p(x) \neq \int p(x|z)\, p(z)\, dz \quad \text{unless } p(z) \text{ is well-defined}
$$
> **Readable form:** The decoder knows how to paint from a coordinate, but nobody told it which coordinates are likely.

---

## Generating from Uniform Samples

Foster samples uniformly within the min-max bounds of training embeddings:

```python
mins, maxs = np.min(embeddings, axis=0), np.max(embeddings, axis=0)
sample = np.random.uniform(mins, maxs, size=(18, 2))
reconstructions = decoder.predict(sample, verbose=0)

fig, axes = plt.subplots(3, 6, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(reconstructions[i].squeeze(), cmap="gray")
    ax.axis("off")
    ax.set_title(f"[{sample[i,0]:.1f}, {sample[i,1]:.1f}]", fontsize=7)
plt.suptitle("Uniform samples in embedding bounding box")
plt.tight_layout()
plt.show()
```

Results are **mixed**: some plausible garments, many incoherent blobs. Figure 3-8 maps each blue sample dot to its decoded image — quality correlates with proximity to dense training regions.

---

## Grid Overlay: Seeing the Holes

Foster's Figure 3-9 decodes a regular grid of latent points and overlays training embeddings. White regions decode to nonsense because no training image was encoded nearby.

```python
n = 20
grid_x = np.linspace(mins[0], maxs[0], n)
grid_y = np.linspace(mins[1], maxs[1], n)

fig, axes = plt.subplots(n, n, figsize=(10, 10))
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z = np.array([[xi, yi]])
        img = decoder.predict(z, verbose=0)[0].squeeze()
        axes[i, j].imshow(img, cmap="gray")
        axes[i, j].axis("off")
plt.suptitle("Decoded grid — note holes at edges")
plt.tight_layout()
plt.show()
```

**Continuity failure:** Even if $(-1, -1)$ decodes to a good sandal, $(-1.1, -1.1)$ may not. The loss only constrains reconstructions at **encoded** points, not across neighborhoods.

---

## Interpolation Preview

Linear interpolation between two encoded points sometimes works locally:

```python
z_a = embeddings[0:1]
z_b = embeddings[100:1]
alphas = np.linspace(0, 1, 8)

fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i, alpha in enumerate(alphas):
    z_interp = (1 - alpha) * z_a + alpha * z_b
    img = decoder.predict(z_interp, verbose=0)[0].squeeze()
    axes[i].imshow(img, cmap="gray")
    axes[i].axis("off")
plt.suptitle("Linear interpolation in latent space")
plt.show()
```

Within a dense cluster, transitions can be smooth. Crossing holes between clusters produces ghostly hybrids — foreshadowing the smoother morphing a [VAE](../../GLOSSARY.md#variational-autoencoder) enables on faces ([Section 3.8](./section-08-latent-space-arithmetic.md)).

---

## Why 2D Is Special

Foster emphasizes that in 2D, discontinuities are subtle because classes are forced together. In 200 dimensions (CelebA, [Section 3.7](./section-07-celeba-faces.md)), gaps become enormous without regularization. Visualization is a **debugging tool** at low $d$; at high $d$, you monitor marginal histograms of each latent dimension instead.

| Dimensionality | Visualization strategy |
|----------------|------------------------|
| 2 | Scatter plots, grid decode |
| 10–50 | Pairwise plots, t-SNE of $z$ |
| 200+ | Per-dimension histograms vs $\mathcal{N}(0,1)$ |

---

## Path to Variational Autoencoders

The three sampling failures motivate the VAE upgrade in [Section 3.4](./section-04-variational-autoencoders.md):

| Plain autoencoder problem | VAE mechanism |
|---------------------------|---------------|
| No $p(z)$ | Prior $p(z) = \mathcal{N}(0, I)$ |
| Uneven occupancy | KL divergence spreads encodings |
| Discontinuities | Stochastic encoder smooths neighborhoods |

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Sampling from $[-1,1]^2$ by habit | Mostly garbage | Use training min/max or switch to VAE |
| Ignoring class imbalance in latent | Wrong item types generated | Inspect colored scatter first |
| Expecting smooth global interpolation | Broken cross-cluster paths | Stay within one cluster or use VAE |
| Using labels in training | Not an autoencoder anymore | Labels for visualization only |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Embedding** | Latent vector $z$ produced by the encoder |
| **Latent space** | Low-dimensional coordinate system for data |
| **Emergent clustering** | Class separation without supervised labels |
| **Hole** | Latent region with no training encodings |
| **Uniform sampling** | Drawing $z$ from a box — naive for plain autoencoders |
| **Interpolation** | $\;z = (1-\alpha) z_A + \alpha z_B$ between encodings |

---

## Reflection Questions

1. Why is it valid to use Fashion-MNIST labels for plotting but not for training the autoencoder?
2. Which class would you expect uniform sampling to over-represent, and why?
3. What does a large white region in a decoded grid imply about the training objective?
4. How would Foster's wardrobe analogy explain holes at the edges of the latent map?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 3 — Visualizing the Latent Space. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Foster's codebase: `notebooks/03_vae/01_autoencoder/autoencoder.ipynb`
- van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *JMLR*.
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.

---

**Previous:** [Section 3.2 — Reconstructing Images](./section-02-reconstructing-images.md)  
**Next:** [Section 3.4 — Variational Autoencoders](./section-04-variational-autoencoders.md)



