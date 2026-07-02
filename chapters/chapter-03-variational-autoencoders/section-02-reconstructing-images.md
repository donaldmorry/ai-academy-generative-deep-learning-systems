# Section 3.2: Reconstructing Images

> **Source inheritance:** Foster, Ch. 3 — "Reconstructing Images"  
> **Enhanced with:** Held-out evaluation, BCE vs RMSE trade-offs, and capacity limits of a 2D bottleneck  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Training an autoencoder is only half the story — you must verify that the network actually **remembers** what it learned. Foster's first diagnostic after `fit()` is reconstruction: pass held-out test images through the encoder-decoder pipeline and compare $\hat{x}$ to $x$. If trousers look like trousers but logos vanish, you have learned something real about the **information bottleneck**, not just that Keras ran without errors.

Reconstruction quality is the direct readout of how much signal survives compression into the [latent space](../../GLOSSARY.md#latent-variable). A 2D bottleneck forces brutal trade-offs: garment type and silhouette survive; fine texture does not.

---

## The Reconstruction Pipeline

After training the Fashion-MNIST autoencoder from [Section 3.1](./section-01-autoencoder-architecture.md), evaluation is straightforward:

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

# Assume encoder, decoder, autoencoder already trained (Section 3.1)
(x_train, _), (x_test, _) = datasets.fashion_mnist.load_data()

def preprocess(imgs):
    imgs = imgs.astype("float32") / 255.0
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    return np.expand_dims(imgs, -1)

x_test = preprocess(x_test)
example_images = x_test[:5000]
predictions = autoencoder.predict(example_images, verbose=0)
```

Foster's Figure 3-6 layout is instructive: top row = originals, middle row = 2D embedding coordinates, bottom row = reconstructions. The middle row reminds you that reconstruction is a **three-stage** story — compress, inspect the code, decompress.

| Stage | Tensor shape | What you learn |
|-------|--------------|----------------|
| Input $x$ | $(32, 32, 1)$ | Ground truth |
| Embedding $z$ | $(2,)$ | Information retained |
| Reconstruction $\hat{x}$ | $(32, 32, 1)$ | Decoder fidelity |

> **Readable form:** Same image in, same image out — but only after squeezing through two numbers.

---

## Quantifying Reconstruction Loss

The training objective is pixel-wise distance between $x$ and $\hat{x}$. Foster discusses two common choices.

**Binary cross-entropy** (pixels treated as independent Bernoullis, outputs in $[0,1]$ via sigmoid):

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^{n}\left[ x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i) \right]
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

**Root mean squared error**:

$$
\mathcal{L}_{\text{RMSE}} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{x}_i)^2}
$$
> **Readable form:** BCE penalizes confident wrong pixels asymmetrically; RMSE penalizes all errors symmetrically around the mean.

Foster's key insight on loss asymmetry:

| True pixel | BCE behavior | RMSE behavior |
|------------|--------------|---------------|
| High (0.7) | Penalizes 0.8 more than 0.6 | Equal distance from 0.7 |
| Low (0.3) | Penalizes 0.2 more than 0.4 | Equal distance from 0.3 |

BCE tends to push uncertain pixels toward 0.5 → **blurrier** reconstructions. RMSE preserves sharper edges but can look **pixelated**. Neither is universally correct — Foster recommends experimenting.

```python
import tensorflow as tf

def reconstruction_rmse(x_true, x_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(x_true - x_pred)))

def per_image_bce(x_true, x_pred, eps=1e-7):
    bce = -(x_true * tf.math.log(x_pred + eps)
            + (1 - x_true) * tf.math.log(1 - x_pred + eps))
    return tf.reduce_mean(bce, axis=(1, 2, 3))

sample_bce = per_image_bce(example_images[:8], predictions[:8])
print("Mean BCE (first 8):", float(tf.reduce_mean(sample_bce)))
```

---

## Visual Inspection: What Gets Lost?

Foster notes that reconstructions are recognizable but imperfect. Logos, stitching, and fine patterns disappear because a 2D code cannot store every pixel detail.

```python
def show_reconstructions(originals, recons, n=8):
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(originals[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recons[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")
    plt.tight_layout()
    plt.show()

show_reconstructions(example_images, predictions)
```

**What typically survives a 2D bottleneck:**

- Garment category (trouser vs boot vs bag)
- Overall silhouette and orientation
- Coarse shading

**What typically fails:**

- Brand logos and printed text
- Fine fabric texture
- Thin straps or laces

This is not a bug — it is the **rate-distortion** trade-off. The encoder keeps whatever minimizes average reconstruction error under capacity constraints.

---

## Capacity vs Regularization

The bottleneck dimension controls how much information flows through $z$:

$$
I(x; z) \leq \dim(z) \cdot \log(\text{resolution of each dimension})
$$
> **Readable form:** A 2D latent can only encode so much; raising dimensions helps reconstruction but hurts visualization and generative sampling (see [Section 3.3](./section-03-visualizing-latent-space.md)).

| Latent dim | Reconstruction | Generative sampling |
|------------|----------------|---------------------|
| 2 | Blurry, lossy | Easy to plot, hard to sample well |
| 32–128 | Sharper | Gaps and discontinuities worsen |
| 200+ (faces) | Good coarse features | Needs VAE regularization |

Foster's sidebar on **denoising autoencoders** previews another use of reconstruction: train on noisy inputs, target clean outputs. The encoder learns noise is not worth storing. That principle scales to [diffusion models](../../GLOSSARY.md#diffusion-model) in Chapter 08, which denoise iteratively across many timesteps.

---

## Train vs Validation Reconstruction

Because targets equal inputs (`x_train` → `x_train`), validation loss is the honest generalization metric:

```python
val_loss = autoencoder.evaluate(x_test, x_test, verbose=0)
print(f"Validation BCE: {val_loss:.4f}")
```

Watch for:

| Pattern | Interpretation |
|---------|----------------|
| Train ↓, val ↓ together | Healthy learning |
| Train ↓, val flat or ↑ | Overfitting (rare with 2D bottleneck) |
| Both high after many epochs | Under-capacity or learning rate issue |

With only two latent dimensions, overfitting is uncommon — the bottleneck itself is the strongest regularizer.

---

## Encoder-Only Reconstruction Check

You can decompose the pipeline to localize errors:

```python
z_batch = encoder.predict(example_images[:8], verbose=0)
recons_from_z = decoder.predict(z_batch, verbose=0)

# Should match full autoencoder output (up to floating point)
np.allclose(recons_from_z, autoencoder.predict(example_images[:8], verbose=0), atol=1e-5)
```

If encoder and decoder reconstructions diverge, the composed `Model` wiring is wrong — a common Functional API mistake.

---

## Interpreting Blur as a Generative Signal

Blurry reconstructions foreshadow a deeper generative problem. A plain autoencoder minimizes reconstruction on **training encodings** but does not require the latent space to be **well-behaved** for random $z$. Holes, discontinuities, and uneven class occupancy (Foster Figures 3-7 through 3-9) mean decoding arbitrary points often yields garbage.

That is why [Section 3.4](./section-04-variational-autoencoders.md) upgrades to a [variational autoencoder](../../GLOSSARY.md#variational-autoencoder): reconstruction loss alone is insufficient for generation.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Evaluating on training set only | Overconfident quality | Always use `x_test` |
| Sigmoid + MSE mismatch | Slow or unstable training | Pair sigmoid with BCE, linear with MSE |
| Forgetting padding | Shape mismatch | Preprocess identically to training |
| Judging only loss scalar | Misses structural errors | Always plot side-by-side grids |
| Expecting logo fidelity at 2D | Disappointment | Increase latent dim or accept blur |

---

## Connection to Other Sections

| Topic | Section |
|-------|--------|
| Architecture that produces $\hat{x}$ | [3.1 Autoencoder Architecture](./section-01-autoencoder-architecture.md) |
| Why blur ≠ good generation | [3.3 Visualizing Latent Space](./section-03-visualizing-latent-space.md) |
| Probabilistic fix (ELBO) | [3.4 Variational Autoencoders](./section-04-variational-autoencoders.md) |
| Higher-dim faces | [3.7 CelebA Faces](./section-07-celeba-faces.md) |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Reconstruction** | Decoder output $\hat{x}$ approximating input $x$ |
| **Reconstruction loss** | Pixel-wise penalty driving encoder-decoder training |
| **Binary cross-entropy** | Asymmetric loss suited to sigmoid pixel outputs |
| **RMSE** | Symmetric L2 pixel error; often sharper edges |
| **Bottleneck** | Low-dimensional layer limiting information throughput |
| **Rate-distortion trade-off** | Less capacity → more compression → more blur |

---

## Reflection Questions

1. Why does Foster show embedding coordinates between originals and reconstructions in Figure 3-6?
2. For a pixel with true value 0.8, would BCE or RMSE penalize a prediction of 0.9 more heavily relative to 0.7?
3. If validation reconstruction loss is much higher than training loss with a 512-D latent, what might be happening?
4. How does blur in reconstructions relate to poor samples from random latent points?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). O'Reilly. Ch. 3 — Reconstructing Images. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Foster's codebase: `notebooks/03_vae/01_autoencoder/autoencoder.ipynb`
- Hinton, G., & Salakhutdinov, R. (2006). Reducing the dimensionality of data with neural networks. *Science*.
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)

---

**Previous:** [Section 3.1 — Autoencoder Architecture](./section-01-autoencoder-architecture.md)  
**Next:** [Section 3.3 — Visualizing Latent Space](./section-03-visualizing-latent-space.md)



