# Section 6.7: GLOW

> **Source inheritance:** Foster, Ch. 6 — "GLOW"  
> **Enhanced with:** Invertible 1×1 convolutions, affine coupling on images, multi-scale architecture, and CelebA faces  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

**GLOW** (Kingma & Dhariwal, 2018) scales [RealNVP](./section-03-realnvp-architecture.md) coupling layers to high-resolution images. Three upgrades over vanilla RealNVP:

1. **Actnorm** — per-channel affine normalization (like batch norm but invertible)
2. **Invertible 1×1 convolution** — mixes channels with tractable Jacobian
3. **Multi-scale architecture** — squeeze spatial dims into channels, factor out latent at each scale

Foster applies GLOW to $32 \times 32$ CelebA faces — same exact-likelihood principle as [two moons](./section-04-two-moons-dataset.md), now with Conv2D coupling and hundreds of thousands of dimensions.

> **Readable form:** RealNVP's coupling trick, but on image tensors with channel shuffles and pyramid latents.

---

## From Dense Coupling to Conv2D Coupling

On images $x \in \mathbb{R}^{H \times W \times C}$, split by **checkerboard mask** or **channel split** instead of halving a 2-vector:

| 2D RealNVP | GLOW |
|------------|------|
| Mask `[1,0]` / `[0,1]` | Checkerboard or channel mask |
| Dense $s, t$ nets | Conv2D $s, t$ nets |
| Single scale | Multi-scale squeezes |

Affine update (same equations as RealNVP):

$$
z_{1:d} = x_{1:d}, \quad z_{d+1:D} = x_{d+1:D} \odot \exp(s) + t
$$
> **Readable form:** Leave one part of the vector unchanged while scaling and shifting the other part with coupling outputs.

Jacobian log-det remains $\sum s_j$ over transformed positions.

---

## Invertible 1×1 Convolution

Channel mixing is critical — without it, channel $c$ only talks to itself through coupling splits. GLOW inserts a **learned permutation** via $1 \times 1$ conv $W$:

$$
z_{c,h,w} = \sum_{c'} W_{c,c'} \, x_{c',h,w}
$$
> **Readable form:** Each spatial location: multiply channel vector by a shared invertible matrix.

$\log |\det J| = H \cdot W \cdot \log |\det W|$ — same at every pixel. $W$ is initialized orthogonal and trained; periodic LU reparameterization keeps inversion stable.

---

## Multi-Scale Flow

After $K$ coupling blocks at full resolution, **squeeze** halves $H, W$ and quadruples channels (space-to-depth). Factor out $4 \times 4 \times C$ latent slice as $z_i$; continue flow on the remainder at lower resolution. Repeat until $4 \times 4$ core.

Benefits:

- Captures **global** structure in early factored latents
- Reduces compute in deeper stages
- Hierarchical $p(x)$ factorization

```python
# Conceptual multi-scale loop (Foster follows Kingma et al.)
def multiscale_flow(x, n_levels=3):
  zs = []
  for level in range(n_levels):
    for _ in range(n_coupling_per_level):
      x, ld = coupling_block(x)
    z_i, x = factor_out(x)  # split channels; keep half as latent
    zs.append((z_i, ld))
  zs.append((x, final_ld))
  return zs
```

---

## Actnorm Layer

**Activation normalization** — per-channel affine with data-dependent init on first batch:

$$
y = s \odot x + b
$$
> **Readable form:** ActNorm applies a learned per-channel scale and bias to normalize activations with an invertible transform.

Invertible; log-det = $\sum \log |s_c| \times H \times W$. Replaces batch norm (which is not invertible). Initialized so each channel has zero mean, unit variance on the first minibatch.

---

## Training Objective (Unchanged)

Still maximize exact log-likelihood:

$$
\log p_X(x) = \log p_Z(f(x)) + \sum_{\text{blocks}} \log |\det J_{\text{block}}|
$$
> **Readable form:** The total combines the indexed terms, so each relevant example, state, feature, or dimension contributes once.

Sum log-dets from actnorm, 1×1 conv, and every coupling layer across scales. Foster uses TensorFlow Probability / custom Keras for CelebA — training is **heavy** compared to moons but same NLL loop as [Section 6.5](./section-05-training-realnvp.md).

---

## CelebA Results (Qualitative)

GLOW generates sharp $32 \times 32$ faces with:

- Diverse identities (good mode coverage vs GAN collapse)
- Slightly softer than contemporary GANs of 2018
- **Exact** density for anomaly scoring (e.g., detect out-of-distribution faces)

Interpolation: $z_t = (1-t) z_1 + t z_2$ in factored latent space, inverse flow → morphing faces. Latent arithmetic analogous to [VAE Section 3.8](../chapter-03-variational-autoencoders/section-08-latent-space-arithmetic.md) but with deterministic codes.

---

## GLOW vs RealNVP Summary

| Feature | RealNVP (moons) | GLOW (faces) |
|---------|-----------------|--------------|
| Domain | $\mathbb{R}^2$ | $32 \times 32 \times 3$ |
| Coupling net | Dense | Conv2D |
| Channel mix | N/A | 1×1 invertible conv |
| Multi-scale | No | Yes |
| Training cost | Seconds | Hours / GPU |

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Non-invertible $W$ | NaN inverse | Orthogonal init; LU param |
| Skipping actnorm init | Unstable first steps | Data-dependent init |
| Full-res only | Blurry global structure | Use multi-scale |
| Ignoring memory | OOM on 64×64 | Reduce batch or levels |

---

## Connection to Other Chapters

| Concept | Link |
|---------|------|
| RealNVP coupling | [Section 6.3](./section-03-realnvp-architecture.md) |
| Conv2D | [Chapter 02](../chapter-02-deep-learning/section-06-convolutional-layers.md) |
| CelebA / faces | [Section 3.7](../chapter-03-variational-autoencoders/section-07-celeba-faces.md) |
| FFJORD continuous | [Section 6.8](./section-08-ffjord-and-normalizing-flows-summary.md) |

---

## Actnorm and 1×1 Conv in Code (Conceptual)

GLOW stacks repeat this pattern at each resolution:

```
[Actnorm] → [Invertible1x1Conv] → [AffineCoupling] × N → [Squeeze] → next level
```

**Actnorm** forward: $y = s \odot x + b$ with log-det $HW \sum \log |s_c|$.  
**1×1 conv** forward: channel mix with log-det $HW \log |\det W|$.

Foster references GLOW on CelebA qualitatively — if extending the lab, start from the official Kingma & Dhariwal TensorFlow/PyTorch reference and map each block to the RealNVP coupling math you already implemented on moons.

---

## Memory and Compute at 32×32×3

| Component | Relative cost |
|-----------|---------------|
| Coupling Conv2D stacks | High |
| 1×1 conv per pixel | Moderate |
| Multi-scale factor-out | Reduces cost at low res |
| Exact NLL per image | Requires all log-dets |

For coursework, two-moons RealNVP proves invertibility; GLOW section is architectural literacy for reading papers and Stable Diffusion's predecessors in generative modeling history.

---

## CelebA Qualitative Checklist (GLOW)

When reviewing GLOW face samples from literature or demos:

- Identity diversity across batch
- Hair/background variation (mode coverage)
- Sharpness vs StyleGAN contemporaries
- Latent interpolation smoothness

Exact NLL on held-out faces should beat single Gaussian — flows' density claim is testable on embeddings if not on pixels alone.

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **GLOW** | Generative flow with invertible 1×1 convs and multi-scale layout |
| **Actnorm** | Invertible per-channel affine normalization |
| **Squeeze** | Reshape that trades spatial size for channels |
| **Factor-out** | Split part of tensor as latent at each scale |
| **Invertible 1×1 conv** | Learned channel permutation with tractable det |

---

## Reflection Questions

1. Why does GLOW need 1×1 conv if coupling already transforms half the channels?
2. How does multi-scale factorization help with compute and representation?
3. What advantage does exact $p(x)$ give for face anomaly detection?
4. Why is actnorm used instead of batch normalization?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 6 — GLOW. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Kingma, D. P. & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1x1 Convolutions. [https://arxiv.org/abs/1807.03039](https://arxiv.org/abs/1807.03039)

---

**Previous:** [Section 6.6 — Analysis of RealNVP](./section-06-analysis-of-realnvp.md)  
**Next:** [Section 6.8 — FFJORD & Summary](./section-08-ffjord-and-normalizing-flows-summary.md)
