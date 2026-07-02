# Section 3.8: Latent Space Arithmetic

> **Source inheritance:** Foster, Ch. 3 — "Latent Space Arithmetic" & "Morphing Between Faces"  
> **Enhanced with:** Attribute vectors, morphing interpolation, and semantic directions in 200-D space  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

A [VAE](../../GLOSSARY.md#variational-autoencoder) latent space is not only for sampling — it is a **vector space where directions mean things**. Foster computes a "smiling vector" as the difference between mean encodings of smiling vs non-smiling CelebA faces, then adds that vector to a neutral face's code. The decoded result smiles. Subtract a "young" vector, add "eyeglasses" — same identity, edited attributes. This is latent arithmetic: linear algebra with a visual API.

---

## Feature Vectors from Labels

CelebA provides binary attributes. For attribute $a$:

$$
\mathbf{v}_a = \mathbb{E}[z \mid a=1] - \mathbb{E}[z \mid a=0]
$$
> **Readable form:** Average location of "smiling" faces minus average location of "not smiling" equals the smile direction.

```python
import numpy as np
import pandas as pd

# metadata from CelebA list_attr_celeba.txt after download
attrs = pd.read_csv("celeba-dataset/list_attr_celeba.csv")
# Encode images (subset for speed) — z_mean from trained encoder
# z_mean_batch shape (N, 200), attr_matrix shape (N, 40)

def feature_vector(z_means, attr_values, attr_idx):
  smiling = attr_values[:, attr_idx] == 1
  return z_means[smiling].mean(axis=0) - z_means[~smiling].mean(axis=0)

smile_vec = feature_vector(z_means, attr_matrix, attr_idx=31)  # Smiling column
```

Foster's Figure 3-19 varies $\alpha$ in:

$$
z_{\text{new}} = z + \alpha \cdot \mathbf{v}_{\text{feature}}
$$
> **Readable form:** Move a latent code in a chosen feature direction, with alpha controlling how strong the edit is.

| $\alpha$ | Effect |
|----------|--------|
| 0 | Original encoding |
| +2 | Stronger feature |
| −2 | Opposite feature |

```python
def edit_face(z, feature_vec, alpha):
    z_new = z + alpha * feature_vec
    return decoder.predict(z_new[np.newaxis], verbose=0)[0]

z_neutral = encoder.predict(face_img[np.newaxis], verbose=0)[0]  # z_mean
for alpha in [-2, -1, 0, 1, 2]:
    edited = edit_face(z_neutral, smile_vec, alpha)
    # plot edited
```

Attributes Foster demonstrates: Smiling, Black_Hair, Eyeglasses, Young, Male, Blond_Hair.

---

## Why Arithmetic Works in VAE Space

Three VAE properties enable this:

1. **Continuity** — stochastic encoder smooths neighborhoods ([Section 3.4](./section-04-variational-autoencoders.md))
2. **Approximate linearity** — supervised attributes correlate with directions in unsupervised $z$
3. **High dimensionality** — 200-D has room for many semi-independent factors

Plain 2D autoencoders rarely support clean single-attribute edits — factors entangle.

---

## Morphing Between Two Identities

Given encodings $z_A$ and $z_B$ for two faces, interpolate:

$$
z(\alpha) = (1 - \alpha)\, z_A + \alpha\, z_B, \quad \alpha \in [0, 1]
$$
> **Readable form:** Walk a straight line in latent space; decode snapshots along the way.

```python
def morph_faces(img_a, img_b, steps=10):
    z_a = encoder.predict(img_a[np.newaxis], verbose=0)[0]  # z_mean
    z_b = encoder.predict(img_b[np.newaxis], verbose=0)[0]
    alphas = np.linspace(0, 1, steps)
    frames = []
    for alpha in alphas:
        z_interp = (1 - alpha) * z_a + alpha * z_b
        frames.append(decoder.predict(z_interp[np.newaxis], verbose=0)[0])
    return frames
```

Figure 3-20: smooth transitions across glasses, hair color, gender presentation — multiple factors change **simultaneously** yet fluidly. This is evidence the VAE learned a genuinely continuous face manifold.

---

## Visualization Grid

```python
import matplotlib.pyplot as plt

features = {
    "Smiling": smile_vec,
    "Eyeglasses": glasses_vec,
    "Young": young_vec,
}
alphas = [-2, -1, 0, 1, 2]

fig, axes = plt.subplots(len(features), len(alphas), figsize=(12, 8))
for i, (name, vec) in enumerate(features.items()):
    for j, alpha in enumerate(alphas):
        img = edit_face(z_base, vec, alpha)
        axes[i, j].imshow(np.clip(img, 0, 1))
        axes[i, j].axis("off")
        if j == 0:
            axes[i, j].set_ylabel(name)
        if i == 0:
            axes[i, j].set_title(f"$\\alpha$={alpha}")
plt.suptitle("Latent space arithmetic on CelebA")
plt.tight_layout()
plt.show()
```

---

## Limitations and Caveats

| Limitation | Explanation |
|------------|-------------|
| **Entanglement** | "Young" may correlate with hair style — not pure factors |
| **Linear assumption** | Large $\alpha$ leaves valid manifold |
| **Attribute noise** | CelebA labels are imperfect |
| **Blur accumulation** | Each decode softens; many steps degrade |

$\beta$-VAE and disentanglement research pursue **more axis-aligned** latents — beyond Foster's scope but motivated by the same arithmetic intuition.

---

## Comparison to Other Generative Models

| Model | Arithmetic |
|-------|------------|
| VAE | Natural — encode, add vector, decode |
| GAN | Possible via optimization in $z$ or attribute classifiers |
| Diffusion | Editing via guidance / inpainting (Chapter 08) |
| Flows | Latent $u$ arithmetic if base is Gaussian (Chapter 06) |

VAE arithmetic remains the **pedagogical gold standard** because it is one matrix addition.

---

## Chapter 03 Synthesis

| Section | Capability |
|--------|------------|
| 3.1–3.2 | Build and evaluate autoencoder reconstructions |
| 3.3 | Diagnose plain-AE sampling failure |
| 3.4–3.6 | VAE theory + ELBO training |
| 3.7 | Scale to RGB faces |
| 3.8 | Manipulate and morph in latent space |

Next: [Chapter 04 — GANs](../chapter-04-generative-adversarial-networks/section-01-gan-introduction.md) — adversarial training for sharper images at the cost of stability.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Using sampled $z$ not $\mu$ for arithmetic | Noisy edits | Use `z_mean` for editing |
| Wrong attribute sign (CelebA uses −1/1) | Inverted direction | Map to boolean consistently |
| $\alpha$ too large | Face breaks | Sweep $\alpha \in [-1, 1]$ first |
| Small encoding subset | Noisy feature vectors | Use 10K+ encodings per class |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Feature vector** | Difference of mean latents between attribute groups |
| **Latent arithmetic** | $z_{\text{new}} = z + \alpha \mathbf{v}$ |
| **Morphing** | Linear interpolation $z(\alpha) = (1-\alpha)z_A + \alpha z_B$ |
| **Semantic direction** | Latent axis correlating with human-interpretable attribute |
| **Entanglement** | Multiple factors changing along one direction |

---

## Reflection Questions

1. Why use mean encodings rather than individual $z$ samples when computing feature vectors?
2. Can morphing be interpreted as arithmetic with $\mathbf{v} = z_B - z_A$ and varying $\alpha$?
3. What would entangled factors look like in a morphing sequence?
4. How does latent arithmetic connect to the ELBO's continuity pressure from KL?

---


## Foster Notebook Reference

Re-run the chapter notebook in [GDL_code](https://github.com/davidADSP/GDL_code) and compare your tensor shapes, loss curves, and saved sample grids to Foster's figures. Document one hyperparameter you changed and how outputs shifted — this habit transfers directly to Part III architectures (Transformers, Stable Diffusion, MuseGAN).

| Checkpoint | Action |
|------------|--------|
| After `model.summary()` | Verify spatial dims match hand calculation |
| Mid-training | Save sample grid or diagnostic plot |
| After training | Compare to Foster figure captions in the PDF |

---

## Extension Reading

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.) — full chapter walkthrough
- Goodfellow, Bengio & Courville (2016). *Deep Learning* — generative models part
- Original papers cited in Foster's chapter references


## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 3 — Latent Space Arithmetic. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Liu, Z. et al. (2015). CelebA dataset.
- Foster's codebase: `notebooks/03_vae/03_faces/vae_faces.ipynb`
- White, T. (2016). Sampling Generative Networks (latent vector arithmetic in GANs).

---

**Previous:** [Section 3.7 — CelebA Faces](./section-07-celeba-faces.md)  
**Next:** [Section 4.1 — GAN Introduction](../chapter-04-generative-adversarial-networks/section-01-gan-introduction.md)


