# Section 6.8: FFJORD & Normalizing Flows Summary

> **Source inheritance:** Foster, Ch. 6 — "FFJORD" / chapter wrap-up  
> **Enhanced with:** Continuous-time flows, Neural ODE perspective, chapter comparison, and when to use flows  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Discrete flows ([RealNVP](./section-03-realnvp-architecture.md), [GLOW](./section-07-glow.md)) compose finitely many coupling layers. **FFJORD** (Grathwohl et al., 2019) treats the flow as a **continuous transformation** defined by an ordinary differential equation (ODE):

$$
\frac{dz(t)}{dt} = f_\theta(z(t), t), \quad z(0) = x
$$
> **Readable form:** Latent code is where you end up after flowing data along a learned vector field from time 0 to time 1.

The terminal state $z(1)$ is the latent code. Likelihood still uses change of variables, but the log-determinant becomes an **integral** along the ODE — estimated with Hutchinson's trace estimator. FFJORD is a capstone idea: flows need not be discrete layer stacks.

---

## Discrete vs Continuous Flows

| | RealNVP / GLOW | FFJORD |
|---|----------------|--------|
| Transform | Finite layers $f_1 \circ \cdots \circ f_L$ | ODE from $t=0$ to $t=1$ |
| Jacobian | Product of per-layer log-dets | $\int \text{tr}(\partial f / \partial z)\, dt$ |
| Architecture | Coupling + masks | Any Lipschitz neural net $f_\theta$ |
| Invertibility | By construction | Numerical ODE solve backward |
| Foster coverage | Full implementation | Conceptual summary |

RealNVP and GLOW are **discrete-time** normalizing flows. FFJORD is **continuous-time** — the limiting case as layer count → ∞ with infinitesimal steps.

---

## FFJORD Likelihood (Sketch)

Instantaneous change of log-density along an ODE (Chen et al., Neural ODEs):

$$
\frac{d \log p(z(t))}{dt} = -\text{tr}\left(\frac{\partial f_\theta}{\partial z(t)}\right)
$$
> **Readable form:** This derivative measures local sensitivity: a larger magnitude means the output changes more for a small input change.

Integrate from $t=0$ (data) to $t=1$ (base Gaussian) to get $\log p(x)$. The trace is approximated stochastically — avoids materializing full Jacobian in high dimensions.

**Trade-offs:**

- **Pros:** Flexible $f_\theta$ without hand-designed coupling masks
- **Cons:** ODE solvers are slow; numerical error in inversion; training variance from trace estimator

Foster points to FFJORD as research frontier rather than a second full notebook — understand the idea, implement RealNVP first.

---

## Chapter 06 Recap: The Flow Family

Jacob's F.L.O.W. machine ([Section 6.1](./section-01-normalizing-flows-introduction.md)) distilled:

1. **Bijection** $f: \mathcal{X} \to \mathcal{Z}$ with inverse $f^{-1}$
2. **Base distribution** $p_Z$ — usually $\mathcal{N}(0,I)$
3. **Change of variables** — $\log p_X(x) = \log p_Z(f(x)) + \log |\det J|$

| Section | Topic |
|--------|-------|
| 6.1 | Motivation vs VAE/GAN |
| 6.2 | Change-of-variables math |
| 6.3 | RealNVP coupling layers |
| 6.4 | Two moons dataset |
| 6.5 | NLL training |
| 6.6 | Analysis & diagnostics |
| 6.7 | GLOW on images |
| 6.8 | FFJORD & summary |

---

## Flows in the Generative Taxonomy

From [Chapter 01](../chapter-01-generative-modeling/section-07-generative-model-taxonomy.md):

| | Likelihood | Sampling speed | Image SOTA (2020+) |
|---|------------|----------------|---------------------|
| **Flow** | Exact | Fast (1 pass) | Good, not top |
| VAE | Bound | Fast | Moderate |
| GAN | None | Fast | Was top |
| AR (PixelCNN) | Exact | Slow | Niche |
| Diffusion | VLB | Slow (many steps) | Top |

**Use flows when:**

- You need **exact** or tractable $\log p(x)$ (anomaly detection, science)
- Fast sampling matters
- Invertibility is required (certain editing pipelines)

**Prefer diffusion/GAN when:**

- Peak perceptual quality on large images is the only metric
- Training budget favors established DDPM/ latent diffusion stacks

---

## Connections Forward and Back

| Direction | Link |
|-----------|------|
| **Back** | VAE encoder-decoder analogy ([Chapter 03](../chapter-03-variational-autoencoders/README.md)) |
| **Sideways** | Autoregressive exact likelihood ([Chapter 05](../chapter-05-autoregressive-models/README.md)) |
| **Forward** | Energy-based models ([Chapter 07](../chapter-07-energy-based-models/section-01-energy-based-models-introduction.md)) — also use scores, different normalization |
| **Forward** | Diffusion ([Chapter 08](../chapter-08-diffusion-models/section-01-diffusion-models-introduction.md)) — iterative invertibility |

Score-based models bridge EBMs and diffusion — Song et al. unified perspectives that Foster previews in Chapter 07–08.

---

## Practical TensorFlow Ecosystem

| Tool | Role |
|------|------|
| Custom Keras `train_step` | RealNVP on moons |
| `tensorflow_probability` | Distributions, some flow layers |
| `tf.keras.layers.Normalization` | Data prep |
| Neural ODE libs (optional) | FFJORD research code |

For coursework: complete RealNVP moons + read GLOW/FFJORD sections. Production image flows often live in PyTorch (nflows, Glow reference) — math transfers.

---

## Key Equations Reference

**Coupling (RealNVP):**

$$
z_{d+1:D} = x_{d+1:D} \odot e^{s(x_{1:d})} + t(x_{1:d}), \quad \log|\det J| = \sum_j s_j
$$
> **Readable form:** The total combines the indexed terms, so each relevant example, state, feature, or dimension contributes once.

**Change of variables:**

$$
p_X(x) = p_Z(f(x)) \left| \det \frac{\partial f}{\partial x} \right|
$$
> **Readable form:** This derivative measures local sensitivity: a larger magnitude means the output changes more for a small input change.

**FFJORD instantaneous:**

$$
\frac{d\log p}{dt} = -\text{tr}\left(\frac{\partial f}{\partial z}\right)
$$
> **Readable form:** This derivative measures local sensitivity: a larger magnitude means the output changes more for a small input change.

---

## Reflection Questions

1. What problem does FFJORD solve that discrete coupling layers cannot?
2. Why did diffusion models overtake GLOW on image benchmarks despite flows' exact likelihood?
3. Name one application where exact $\log p(x)$ is non-negotiable.
4. How is Jacob's bell related to sampling $z \sim \mathcal{N}(0,I)$?

---

## Decision Guide: Which Flow Extension?

| Your goal | Start with |
|-----------|------------|
| Pass exam on Jacobian | RealNVP moons ([Sections 6.3–6.6](./section-03-realnvp-architecture.md)) |
| Implement exact NLL on images | GLOW reference code |
| Research continuous depth | FFJORD / Neural ODE papers |
| Production images today | Latent diffusion (Chapter 08), not flows |

FFJORD's Hutchinson trace estimator trades compute per step for architectural freedom — know the idea, implement RealNVP first.

---

## Chapter 06 Equation Sheet

Copy these into your lab notes:

$$
z_{d+1:D} = x_{d+1:D} \odot e^{s(x_{1:d})} + t(x_{1:d}), \quad \log|J| = \sum_j s_j
$$
> **Readable form:** The total combines the indexed terms, so each relevant example, state, feature, or dimension contributes once.

$$
\log p_X(x) = \log \mathcal{N}(f(x); 0, I) + \log \left| \det \frac{\partial f}{\partial x} \right|
$$
> **Readable form:** This derivative measures local sensitivity: a larger magnitude means the output changes more for a small input change.

$$
\frac{d \log p}{dt} = -\text{tr}\left(\frac{\partial f_\theta}{\partial z}\right) \quad \text{(FFJORD)}
$$
> **Readable form:** This derivative measures local sensitivity: a larger magnitude means the output changes more for a small input change.

---

## Exam-Style Rapid Fire

1. **Why triangular Jacobian?** — Coupling masks make $\det J$ a product of diagonal scales.
2. **Why invertible?** — Exact sampling requires $z \to x$ reverse map.
3. **FFJORD vs RealNVP?** — Continuous ODE vs discrete layers; Hutchinson vs closed-form log-det.
4. **When prefer flow over diffusion?** — Need exact $\log p(x)$ or one-shot sample with likelihood.

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **FFJORD** | Free-form continuous dynamics for flows via ODE |
| **Neural ODE** | Network defines derivative $dz/dt = f_\theta(z,t)$ |
| **Discrete-time flow** | Finite composition of invertible layers |
| **Trace estimator** | Hutchinson trick for FFJORD log-det integral |
| **Bijection** | One-to-one onto map with differentiable inverse |

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 6 — FFJORD, summary. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Grathwohl, W. et al. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. [https://arxiv.org/abs/1810.01367](https://arxiv.org/abs/1810.01367)
- Chen, T. Q. et al. (2018). Neural Ordinary Differential Equations. [https://arxiv.org/abs/1806.07366](https://arxiv.org/abs/1806.07366)
- Kingma, D. P. & Dhariwal, P. (2018). Glow. [https://arxiv.org/abs/1807.03039](https://arxiv.org/abs/1807.03039)

---

**Previous:** [Section 6.7 — GLOW](./section-07-glow.md)  
**Next:** [Lab 06](./section-lab-06-realnvp-on-two-moons.md)
