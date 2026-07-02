# Section 8.8: Analysis & Connections

> **Source inheritance:** Foster, Ch. 8 — analysis, DDIM, Stable Diffusion connections, Part II wrap-up  
> **Enhanced with:** Chapter comparison, latent diffusion, score-based unification, and production landscape  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Foster closes Chapter 08 by evaluating the **flower DDPM**, comparing it to every other Part II family, and pointing to what came next: **DDIM** fast sampling, **latent diffusion** (Stable Diffusion), **classifier-free guidance**, and the theoretical bridge between diffusion, score matching, and [EBMs](../chapter-07-energy-based-models/section-07-other-ebms-and-connections.md).

You built a miniature Stable Diffusion core — iterative denoising U-Net on images. Production systems add a VAE latent space, text encoders, and distilled samplers.

> **Readable form:** Your flower model is the denoising heart inside every modern image generator.

---

## Evaluating Flower DDPM Outputs

**Quantitative (lightweight):**

- Track `noise_loss` on validation batches — should correlate with sample quality
- FID vs flower training set (optional; needs many samples + inception)

**Qualitative checklist:**

| Criterion | Pass indicator |
|-----------|----------------|
| Diversity | Different colors, species, layouts |
| Coherence | Recognizable flower structure |
| Sharpness | Petal edges, not soup |
| Artifacts | No repeating grid glitches |

Compare grids at epochs 25, 50, 100 — diffusion often **suddenly** snaps into structure mid-training (unlike GAN gradual sharpening).

---

## Part II Generative Taxonomy (Complete)

| Chapter | Model | Likelihood | Sampling | Foster dataset |
|--------|-------|------------|----------|----------------|
| 03 | VAE | ELBO | 1-step | Fashion-MNIST, CelebA |
| 04 | GAN | Implicit | 1-step | CelebA |
| 05 | PixelCNN | Exact | Sequential | Fashion-MNIST |
| 06 | RealNVP/GLOW | Exact | 1 inverse | Two moons, faces |
| 07 | EBM | Unnormalized | Langevin | MNIST |
| 08 | **DDPM** | VLB | **T steps** | **Flowers** |

**Diffusion wins** on image quality vs training stability trade-off in 2020–2024. **Flows win** on exact density. **GANs win** on raw speed. **AR wins** on exact likelihood for discrete data.

---

## Score-Based Unification

Three views of the same object:

$$
\nabla_x \log p(x) \quad \longleftrightarrow \quad -\epsilon / \sigma \quad \longleftrightarrow \quad -\nabla_x E(x)
$$
> **Readable form:** This derivative measures local sensitivity: a larger magnitude means the output changes more for a small input change.

| Framework | Object learned |
|-----------|----------------|
| EBM | Energy $E(x)$ |
| NCSN | Score $\nabla \log p$ |
| DDPM | Noise $\epsilon$ at each noise level |

Song et al. unified diffusion and score-based models — Foster's Chapter 07→08 arc is intentional pedagogy.

---

## DDIM: Faster Sampling

**Problem:** 1000 U-Net evaluations per image.  
**DDIM:** deterministic sampler skipping substeps — 50 steps often sufficient.

Key idea: non-Markovian reverse process sharing same marginals as DDPM. Not in Foster's base notebook but standard in deployment:

```python
# Conceptual — eta=0 deterministic DDIM step
# x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1-alpha_{t-1}) * pred_eps
```

Also explore **consistency models** (2023+) distilling multi-step into 1–4 steps — active research frontier.

---

## Latent Diffusion (Stable Diffusion)

Train diffusion in **VAE latent space** $z = \text{Enc}(x)$, not pixel space:

| Stage | Component |
|-------|-----------|
| 1 | VAE encoder/decoder (frozen) |
| 2 | DDPM U-Net on 64×64 latents |
| 3 | Text encoder (CLIP) for conditioning |
| 4 | Guidance at sample time |

Your 64×64 RGB DDPM is **pixel-space diffusion** — Stable Diffusion is the same loop on 8× compressed latents at 512×512 effective resolution. [Chapter 03 VAE](../chapter-03-variational-autoencoders/section-04-variational-autoencoders.md) + Chapter 08 = production recipe.

---

## Classifier-Free Guidance

Train U-Net with random null condition (10% dropout). Sample with:

$$
\epsilon_\theta^{\text{guided}} = \epsilon_\theta(x_t, t, \emptyset) + w\, (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))
$$
> **Readable form:** Start from the unconditional noise prediction and add a scaled push toward the conditional prediction to increase prompt adherence.

$w \approx 7.5$ typical for text-to-image. Explains prompt adherence in DALL·E / Stable Diffusion — not in flower DDPM (unconditional) but the natural extension.

---

## Connections to Part III (Preview)

Foster Part III covers **Transformers** (autoregressive at scale), **MuseGAN**, **Transformers for video** — all build on Part II intuitions:

- Autoregressive factorization → GPT, image tokens
- Diffusion → Sora-class video, audio (Stable Audio)
- Adversarial + perceptual losses → hybrid systems

Your flower DDPM is the same mathematical object inside larger pipelines.

---

## When to Choose What (2026 Practitioner Guide)

| Need | Recommendation |
|------|----------------|
| Text-to-image SOTA | Latent diffusion + guidance |
| Exact likelihood / science | Normalizing flows |
| Fastest prototype | GAN or small VAE |
| Discrete sequences | Autoregressive / Transformer |
| OOD detection | Flow NLL or EBM energy |
| Course homework | Match Foster's chapter for each family |

---

## Chapter 08 Section Recap

| Section | Topic |
|--------|-------|
| 8.1 | DiffuseTV / DDPM intro |
| 8.2 | Forward process $q$ |
| 8.3 | Reparameterization & schedules |
| 8.4 | Reverse process & $\epsilon$-loss |
| 8.5 | U-Net architecture |
| 8.6 | Training + EMA |
| 8.7 | Sampling loop |
| 8.8 | Analysis & connections (this section) |

---

## Reflection Questions

1. How is your flower DDPM related to Stable Diffusion's core loop?
2. Why did diffusion overtake GANs on ImageNet FID around 2020?
3. What would you add to Foster's model for text-conditioned flowers?
4. Which Part II model would you pick for exact anomaly scores on tabular data?

---

## Lab Deliverables Checklist (Chapter 08)

| Deliverable | Description |
|-------------|-------------|
| Schedule plot | Linear vs offset cosine signal/noise curves |
| Training curve | `noise_loss` vs epoch |
| Sample grid | 4×4 flowers at epoch 50+ |
| Ablation | One change: LR, EMA decay, or steps at sample |
| Written compare | 3 sentences vs VAE or GAN on same flowers metric |

Tie each item to a Foster figure caption from Ch. 8 for full marks on reproducibility.

---

## Part II Capstone Reflection

Chapters 03–08 each taught one generative family with the same evaluation habits: reproduce a Foster figure, ablate one hyperparameter, compare families on a shared metric. Diffusion is the capstone — most moving parts (schedule, U-Net, EMA, sampler) but the most familiar in 2026 tooling. Carry these habits into Part III Transformers and MuseGAN labs.

---

## Congratulations

You have completed Foster Part II's generative model survey: VAE, GAN, autoregressive, flow, EBM, and diffusion. Each family trades likelihood, speed, and quality differently — no single winner, only problem-dependent choices.

---

## Open Problems (Research Hooks)

- **Few-step generation** — consistency models, distillation
- **Controllable layout** — ControlNet, cross-attention U-Nets
- **Video** — temporal attention + 3D U-Net variants
- **Audio** — mel-spectrogram diffusion (same math, 2D U-Net)

Your flower DDPM is the baseline checkpoint on this roadmap.

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Latent diffusion** | DDPM in VAE-compressed space |
| **DDIM** | Fast non-Markovian sampler |
| **Classifier-free guidance** | Conditional sampling without separate classifier |
| **FID** | Fréchet Inception Distance — sample quality metric |
| **Score unification** | Equivalence of $\epsilon$, score, and energy gradients |

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 8 — full chapter. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Ho, J. et al. (2020). DDPM. [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- Rombach, R. et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion). [https://arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)
- Song, J. et al. (2020). DDIM. [https://arxiv.org/abs/2010.02502](https://arxiv.org/abs/2010.02502)
- Song, Y. et al. (2021). Score-Based Generative Modeling through Stochastic Differential Equations.

---

**Previous:** [Section 8.7 — Sampling from DDPM](./section-07-sampling-from-ddpm.md)  
**Next:** [Lab 08](./section-lab-08-ddpm-on-flowers-with-u-net.md)
