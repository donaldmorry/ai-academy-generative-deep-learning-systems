# Section 7.7: Other EBMs & Connections

> **Source inheritance:** Foster, Ch. 7 — wrap-up / connections to score-based and diffusion models  
> **Enhanced with:** NCSN, score matching, Boltzmann machines, diffusion bridge, and chapter summary  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Foster's MNIST EBM is one point in a large family of **energy-based** and **score-based** methods. Modern image generation often bypasses explicit $E(x)$ and learns the **score** $\nabla_x \log p(x)$ directly — then connects to **denoising diffusion** ([Chapter 08](../chapter-08-diffusion-models/section-01-diffusion-models-introduction.md)). This section maps the landscape: classical Boltzmann machines → contrastive divergence EBMs → noise conditional score networks (NCSN) → DDPM.

> **Readable form:** Energy models teach you to shape landscapes; score and diffusion models teach you to follow the slope downhill through noise.

---

## Classical Energy Models

| Model | Era | Idea |
|-------|-----|------|
| **Hopfield network** | 1982 | Associative memory as energy minima |
| **Boltzmann machine** | 1980s | Binary units, $E = -\sum w_{ij} s_i s_j$ |
| **Restricted BM (RBM)** | 2006+ | Bipartite layers; CD training |
| **Deep Boltzmann** | 2009 | Stack of RBMs |

Deep learning revived CD for **continuous** deep EBMs (Du & Mordatch, 2019) — Foster's notebook is in this line.

---

## Score Matching Alternative to CD

For $p(x) \propto e^{-E(x)}$:

$$
\nabla_x \log p(x) = -\nabla_x E(x) - \nabla_x \log Z
$$
> **Readable form:** This derivative measures local sensitivity: a larger magnitude means the output changes more for a small input change.

The $\nabla_x \log Z$ term vanishes — score does not depend on the partition function! **Score matching** (Hyvärinen, 2005) trains a network $s_\theta(x) \approx \nabla_x \log p(x)$ without MCMC negatives.

**NCSN** (Song & Ermon, 2019): add noise at multiple scales so the score is well-defined even in low-density regions. Sample via annealed Langevin — related to Foster's Langevin section but on **scores**, not raw energies.

---

## Bridge to Diffusion Models

Ho et al. (2020) **DDPM** showed:

- Forward diffusion adds Gaussian noise (fixed, not learned)
- Reverse process trains a network to predict noise $\epsilon$ — equivalent to score matching at each noise level
- Connection: diffusion ELBO ↔ weighted denoising score matching

```
EBM (Foster Ch.7)     →  energy E(x), CD + Langevin
Score-based (NCSN)    →  ∇log p(x), noise-conditioned
Diffusion (DDPM)      →  predict ε in x_t = √ᾱ x_0 + √(1-ᾱ) ε
Stable Diffusion      →  latent diffusion + score in VAE space
```

You trained the "hard way" with CD; production systems use diffusion — but the **intuition** (navigate data manifold downhill) is shared.

---

## Products of Experts and Joint Energies

Hinton's **product of experts** combines energies:

$$
p(x) \propto \prod_k e^{-E_k(x)} \quad \Leftrightarrow \quad E_{\text{total}}(x) = \sum_k E_k(x)
$$
> **Readable form:** The total combines the indexed terms, so each relevant example, state, feature, or dimension contributes once.

Useful for multimodal conditioning — each expert enforces one constraint (e.g., style, class label). **Conditional EBMs** extend Foster's scalar $E(x)$ to $E(x|y)$ for guided generation.

---

## EBM vs Score vs Diffusion (Summary)

| | Deep EBM (Foster) | NCSN | DDPM |
|---|-------------------|------|------|
| Learned object | $E_\theta(x)$ | $s_\theta(x)$ | $\epsilon_\theta(x_t, t)$ |
| Training | Contrastive divergence | Denoising score match | MSE on noise |
| Sampling | Langevin on $x$ | Annealed Langevin | Reverse diffusion |
| $Z$ needed? | No (CD) | No | No (VLB) |
| Foster chapter | 7 | Cited in 8 | 8 |

---

## When to Study EBMs Today

**Still relevant for:**

- Research on unnormalized models
- Understanding Langevin MCMC
- Anomaly detection via energy thresholds
- Theoretical links to diffusion/score literature

**Often replaced by:**

- Diffusion models for image synthesis
- Flows when exact likelihood required
- GANs for fastest sharp sampling (some domains)

---

## Chapter 07 Recap

| Section | Topic |
|--------|-------|
| 7.1 | Boltzmann / Diane Mixx story |
| 7.2 | MNIST + swish energy CNN |
| 7.3 | Langevin dynamics |
| 7.4 | Contrastive divergence + buffer |
| 7.5 | Custom Keras training |
| 7.6 | Histograms & sample analysis |
| 7.7 | Connections (this section) |

**Next chapter:** [Diffusion models](../chapter-08-diffusion-models/section-01-diffusion-models-introduction.md) — where Foster's story moves from energy landscapes to iterative denoising (DiffuseTV).

---

## Forward Peek: DDIM and Faster Sampling

Song et al. (2020) **DDIM** accelerates diffusion sampling — non-Markovian reverse process with fewer steps. Foster mentions in Chapter 8 analysis. EBMs don't have an equivalent shortcut without learned samplers — another reason diffusion won engineering mindshare.

---

## Timeline: From Boltzmann to Stable Diffusion

```
1980s Hopfield/RBM  →  2002 CD (Hinton)  →  2019 Deep EBM (Du)
        ↓                      ↓                    ↓
   energy E(x)            train without Z      Langevin samples
        ↓
2019 NCSN (Song)  →  2020 DDPM (Ho)  →  2022 Latent Diffusion
   score ∇log p         predict ε           VAE + U-Net + text
```

Reading Chapter 07 then Chapter 08 follows this arc deliberately — you feel the pain of CD + Langevin, then appreciate why predicting noise at fixed corruption levels scales better.

---

## Hybrid Systems in Production (2024+)

Modern pipelines rarely use pure single-family models:

| System | Hybrid components |
|--------|---------------------|
| Stable Diffusion | VAE + DDPM U-Net + CLIP + guidance |
| DALL·E 3 | Autoregressive prior + diffusion decoder |
| Some GAN refiners | GAN discriminator loss on diffusion outputs |

Part II teaches each ingredient separately so you can read architecture diagrams in papers and know which Foster chapter corresponds to which box.

---

## Suggested Further Reading Order

1. Finish Foster Ch. 8 lab (flowers)
2. Ho et al. DDPM paper — 10 pages, matches course math
3. Song & Ermon NCSN — connects to Section 7.7
4. Rombach et al. Stable Diffusion — see latent compression payoff from Chapter 03

---

## Exercise: Map Papers to Sections

| Paper | Closest Foster section |
|-------|----------------------|
| Hinton CD 2002 | [7.4 Contrastive Divergence](./section-04-contrastive-divergence.md) |
| Du & Mordatch 2019 | [7.5 Training EBM](./section-05-training-the-ebm.md) |
| Song NCSN 2019 | [7.7 Other EBMs](./section-07-other-ebms-and-connections.md) |
| Ho DDPM 2020 | [8.4 Reverse Process](../chapter-08-diffusion-models/section-04-reverse-diffusion-process.md) |

Completing this table from memory is a strong pre-exam review for Part II generative models.

---

## Transition to Chapter 08

After Lab 07, skim [Section 8.1](../chapter-08-diffusion-models/section-01-diffusion-models-introduction.md) DiffuseTV story — notice how iterative denoising replaces Langevin as the default sampler in modern systems. The energy landscape intuition remains: descent toward high-density regions, now parameterized as noise prediction at each blur level.

---

## Glossary Cross-Links

Review [GLOSSARY.md](../../GLOSSARY.md) entries for **energy-based model**, **contrastive divergence**, and **score function** before the Chapter 07 exam — Foster uses these terms interchangeably with informal names in the Diane Mixx narrative.

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Score function** | $\nabla_x \log p(x)$ |
| **Score matching** | Train $s_\theta \approx \nabla \log p$ without $Z$ |
| **NCSN** | Noise Conditional Score Network |
| **Product of experts** | Sum of energies from multiple models |
| **Annealed Langevin** | Gradually reduce noise scale while sampling |

---

## Reflection Questions

1. Why does score matching avoid the partition function but CD still avoids it differently?
2. How is predicting diffusion noise $\epsilon$ related to the score?
3. What did Hopfield networks store as energy minima?
4. Why might you still implement a toy EBM before studying DDPM?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 7 — connections; Ch. 8 preview. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Song, Y. & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. [https://arxiv.org/abs/1905.12672](https://arxiv.org/abs/1905.12672)
- Ho, J. et al. (2020). Denoising Diffusion Probabilistic Models. [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- Du, Y. & Mordatch, I. (2019). Implicit Generation and Modeling with Energy-Based Models.

---

**Previous:** [Section 7.6 — Analysis of the EBM](./section-06-analysis-of-the-ebm.md)  
**Next:** [Lab 07](./section-lab-07-ebm-on-mnist-with-langevin-sampling.md)
