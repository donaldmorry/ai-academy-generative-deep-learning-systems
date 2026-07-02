# Section 6.1: Normalizing Flows Introduction

> **Source inheritance:** Foster, Ch. 6 — "Normalizing Flows"  
> **Enhanced with:** Bijective maps, base distributions, F.L.O.W. machine story, vs VAE/GAN  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

**Normalizing flows** learn an **invertible** mapping $f: \mathcal{X} \to \mathcal{Z}$ from complex data to a simple base distribution — usually $\mathcal{N}(0, I)$. Unlike VAE encoders that approximate posteriors, flow layers are **exact bijections** with tractable Jacobian determinants, so you can compute **exact** $\log p(x)$ via the change-of-variables formula.

Foster's F.L.O.W. machine: digitize paintings to random numbers (forward $f$), recover originals (inverse $f^{-1}$), then ring a bell to sample Gaussian noise and decode **new** paintings (generate).

---

## The Three Requirements

Foster states every flow layer must be:

1. **Invertible** — given $z$, recover $x$ exactly (no information loss)
2. **Expressive** — stack layers to map complex $\mathcal{X}$ to Gaussian $\mathcal{Z}$
3. **Tractable Jacobian** — $\det \partial z / \partial x$ computable efficiently

| Model | Invertible? | Exact $\log p(x)$? |
|-------|-------------|-------------------|
| VAE | No (stochastic) | ELBO only |
| GAN | N/A | Implicit |
| **Flow** | **Yes** | **Yes** |
| Autoregressive | Yes (triangular Jacobian) | Yes, slow |

---

## Forward and Inverse

$$
z = f(x), \quad x = f^{-1}(z) = g(z)
$$
> **Readable form:** Forward squashes data to Gaussian space; inverse expands samples back to data space.

**Generation:** $z \sim \mathcal{N}(0,I)$, return $x = g(z)$.

**Density estimation:** compute $\log p_X(x)$ using change of variables ([Section 6.2](./section-02-change-of-variables.md)).

---

## Base Distribution $p_Z(z)$

Choose $p_Z$ simple — standard Gaussian:

$$
p_Z(z) = \mathcal{N}(z; 0, I)
$$
> **Readable form:** The latent variable follows a standard normal distribution before the flow transforms it.

Sampling is trivial; $\log p_Z(z)$ has closed form. The flow **transforms** this simple density into complex data density $p_X(x)$.

```python
import tensorflow_probability as tfp
tfd = tfp.distributions
base = tfd.MultivariateNormalDiag(loc=[0., 0.])
z_sample = base.sample(1000)
log_pz = base.log_prob(z_sample)
```

---

## Composition of Flows

Stack invertible layers $f_1, f_2, \ldots, f_K$:

$$
z = f_K \circ \cdots \circ f_1(x)
$$
> **Readable form:** Map data to latent space by applying each invertible flow transform in sequence.

Inverse: apply $f_k^{-1}$ in reverse order. Total log-determinant **sums** across layers — key to scalable training.

---

## Architectural Constraints

Invertibility limits architecture vs free-form CNNs:

| Allowed | Restricted |
|---------|------------|
| Coupling layers (RealNVP) | Arbitrary downsampling |
| 1×1 conv (GLOW) | Non-invertible pooling |
| Affine transforms | Changing dimension |

Foster builds **RealNVP** on 2D two moons ([Sections 6.3–6.5](./section-03-realnvp-architecture.md)) before surveying GLOW and FFJORD.

---

## F.L.O.W. Machine Analogy

| Machine part | Flow concept |
|--------------|--------------|
| Digitize painting | Forward $f(x)$ |
| Recover painting | Inverse $g(z)$ |
| Bell → random numbers | Sample $z \sim \mathcal{N}(0,I)$ |
| New painting | $x = g(z)$ |

Unlike GANs, no adversary — pure **maximum likelihood** on $\log p_X(x)$.

---

## Comparison to Chapter 03–05

| | VAE | GAN | Flow |
|---|-----|-----|------|
| Training | ELBO | Minimax | MLE |
| Latent | Approximate $q(z|x)$ | Noise $z$ | Exact $z=f(x)$ |
| Sample speed | Fast | Fast | Fast (one inverse pass) |
| Density | Lower bound | None | **Exact** |

---

## Preview: RealNVP on Two Moons

Foster's pedagogical 2D example ([Section 6.4](./section-04-two-moons-dataset.md)): learn bijection mapping two crescents to Gaussian blob; sample $z$, inverse to crescent points. Visual proof flows work before scaling to images (GLOW).

Notebook: `notebooks/06_normflow/01_realnvp/realnvp.ipynb`

---

## Sampling from a Trained Flow

At inference: $z \sim \mathcal{N}(0,I)$, apply inverse coupling layers in reverse order, obtain $x = g(z)$. Foster visualizes this on two moons ([Section 6.4](./section-04-two-moons-dataset.md)) — crescent points from Gaussian draws.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Non-invertible layer | Wrong density | Use coupling / invertible 1×1 conv only |
| Ignoring log-det | Biased MLE | Add $\log |\det J|$ in loss |
| Wrong base dim | Shape mismatch | Preserve dimension through flow |
| Confusing with VAE | Stochastic encoder | Flows are deterministic bijections |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Normalizing flow** | Invertible map + base distribution for exact density |
| **Bijection** | One-to-one differentiable map with differentiable inverse |
| **Base distribution** | Simple $p_Z$, usually standard Gaussian |
| **Change of variables** | Formula relating $p_X$ and $p_Z$ via Jacobian |
| **Coupling layer** | RealNVP block updating half coordinates from other half |

---

## Reflection Questions

1. Why must flow layers preserve dimensionality?
2. How does sampling from a flow differ from sampling a VAE prior?
3. What three properties does Foster require of each flow transformation?
4. Why is exact likelihood an advantage over GANs?

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



## Lab Integration Notes

When completing the chapter lab, tie this section's implementation checklist to your deliverable:

1. **Reproduce** Foster's primary figure for this topic (save PNG).
2. **Ablate** one hyperparameter (learning rate, latent dim, noise schedule, critic steps, etc.).
3. **Compare** to one other Part II model family on the same metric (likelihood, sample sharpness, or training stability).
4. **Write** three sentences explaining failure modes you observed and the fix you applied.

These steps mirror how generative modeling is evaluated in research and production — not only final image quality, but reproducibility and diagnosed trade-offs.

## Tensor Shape Debugging Template

```python
import tensorflow as tf

def trace_shapes(model, sample_input):
    x = sample_input
    for layer in model.layers:
        try:
            x = layer(x)
            print(f"{layer.name:30s} {x.shape}")
        except Exception as e:
            print(f"{layer.name:30s} ERROR: {e}")
            break
```

Use after every architecture change before committing to long training runs.


## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 6 — Normalizing Flows. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Dinh, L. et al. (2014). NICE: Non-linear Independent Components Estimation.
- Foster's codebase: `notebooks/06_normflow/01_realnvp/realnvp.ipynb`

---

**Previous:** [Section 5.8 — Mixture Distributions](../chapter-05-autoregressive-models/section-08-mixture-distributions-for-pixelcnn.md)  
**Next:** [Section 6.2 — Change of Variables](./section-02-change-of-variables.md)


