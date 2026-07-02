# Section 4.8: Analysis & Comparison

> **Source inheritance:** Foster, Ch. 4 — "Analysis" sections & Chapter Summary  
> **Enhanced with:** DCGAN vs WGAN-GP vs CGAN vs VAE trade-offs, failure case studies, evaluation  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Chapter 04 built three [GAN](../../GLOSSARY.md#generative-adversarial-network-gan) variants on two datasets — DCGAN bricks, WGAN-GP faces, CGAN conditioned faces. This capstone section synthesizes **what worked, what failed, and how GANs compare to VAEs** from [Chapter 03](../chapter-03-variational-autoencoders/README.md). Foster's chapter summary is clear: adversarial training trades optimization pain for **sharp, realistic samples** — when it works.

---

## DCGAN on Bricks: What We Learned

**Successes (Foster Figures 4-7, 4-8):**

- Generator learns 3D structure — shadows, studs, rectangular forms
- Samples are **novel** — L1 nearest-neighbor to training set shows variation
- No hand-crafted features — pure pixel learning

**Failures (Figures 4-9, 4-10):**

| Failure | Cause | Fix attempted |
|---------|-------|---------------|
| Noise forever | D too strong | Label noise, dropout |
| Identical outputs | Mode collapse | Stronger D, larger batch |
| Misleading loss | Uninformative G loss | Visual monitoring |

```python
def nearest_neighbor_l1(generated, training_set, k=1):
    """Foster memorization check — generated vs closest training image."""
    gen_flat = generated.reshape(-1, np.prod(generated.shape[1:]))
    train_flat = training_set.reshape(-1, np.prod(training_set.shape[1:]))
    distances = np.mean(np.abs(gen_flat[:, None] - train_flat[None, :]), axis=-1)
    return np.min(distances, axis=1)
```

---

## WGAN-GP on CelebA: Stability Wins

Foster Figure 4-14 vs 4-15:

| Metric | DCGAN (typical) | WGAN-GP |
|--------|-----------------|---------|
| Loss curves | Oscillatory | Smooth convergence |
| Mode collapse | Common | Rare in Foster's run |
| Sample sharpness | Good | **Better hair/background** |
| Training complexity | Simpler code | GP + critic loop |

**Qualitative VAE comparison** (Chapter 3 vs 4):

| | VAE faces | WGAN-GP faces |
|---|-----------|---------------|
| Sharpness | Softer | Sharper edges |
| Training | Stable ELBO | Needs tuning but GP helps |
| Latent edit | Arithmetic | CGAN labels |
| Likelihood | ELBO bound | None |

---

## CGAN: Controllable Generation

Figure 4-17 analysis:

- Fixed $z$, swap blond / not-blond label → isolated attribute change
- Proves GAN latent + label space captures **factorized** semantics
- Contrast with VAE arithmetic ([Section 3.8](../chapter-03-variational-autoencoders/section-08-latent-space-arithmetic.md)) — both enable editing, different mechanisms

```python
# Side-by-side chapter comparison at inference
z = np.random.normal(size=(1, 32))
vae_face = vae_decoder(z_vae)           # Chapter 03 — sample prior
wgan_face = wgan_generator(z)           # Chapter 04 — sample noise
cgan_blond = cgan_generator([z, [0,1]]) # Chapter 04 — conditioned
```

---

## Generative Model Comparison Table

| Model | Density | Sample speed | Sharpness | Stability | Control |
|-------|---------|--------------|-----------|-----------|---------|
| VAE | ELBO | 1 forward pass | Moderate | High | Latent arithmetic |
| DCGAN | Implicit | 1 forward pass | High | Low | Limited |
| WGAN-GP | Implicit | 1 forward pass | High | Medium | Limited |
| CGAN | Implicit | 1 forward pass | High | Medium | **Label** |
| PixelCNN (Mod 05) | Exact | 1024+ steps | Moderate | High | Slow |
| DDPM (Mod 08) | Variational bound | 1000 steps | Very high | High | Text/img cond |

---

## Evaluation Beyond Eyeballing

Foster uses qualitative grids primarily. Modern practice adds:

| Metric | What it measures |
|--------|------------------|
| **FID** (Fréchet Inception Distance) | Feature distribution distance |
| **IS** (Inception Score) | Classifier confidence + diversity |
| L1 memorization | Overfitting to training set |

```python
# FID requires pretrained Inception — conceptual usage
# pip install tensorflow-gan or use pytorch-fid externally
# Lower FID = closer to real image statistics
```

For course labs, Foster's approach suffices: **grids + loss curves + failure documentation**.

---

## Failure Case Study Template

Document one training run (Lab 04 deliverable):

```markdown
## Failure: Mode collapse at epoch 40
- **Symptoms:** Identical bricks in 4x4 grid; d_loss < 0.1
- **Diagnosis:** Discriminator too weak after LR change
- **Fix:** Increased D filters; reduced G steps; batch 128→256
- **Outcome:** Diversity restored by epoch 60
```

---

## When to Choose GAN vs VAE vs Diffusion

| Choose **VAE** when | Choose **GAN** when | Choose **diffusion** when |
|--------------------|--------------------|---------------------------|
| Need latent arithmetic | Need sharp images fast | Need SOTA quality |
| Stability critical | Have GPU time to tune | Can afford slow sampling |
| Probabilistic framework matters | Adversarial OK | Training stability paramount |

Foster's pedagogical order: VAE first (principled), GAN second (adversarial), diffusion last (state of art).

---

## Chapter 04 Artifact Checklist

Before [Chapter 05](../chapter-05-autoregressive-models/section-01-autoregressive-framework.md):

- [ ] DCGAN brick samples at multiple epochs
- [ ] WGAN-GP face grid (25 epochs)
- [ ] CGAN blond vs not-blond comparison (fixed $z$)
- [ ] One documented failure + fix
- [ ] Written comparison vs VAE (sharpness, stability, control)

---

## Historical Context

Foster notes GAN extensions drive Chapter 10 (ProGAN, StyleGAN, BigGAN). Core Chapter 04 ideas persist:

- Adversarial two-player framework
- Convolutional generator/discriminator
- Wasserstein + GP stabilization
- Conditional generation

Diffusion models (Chapter 08) largely supplanted GANs for image SOTA, but GANs remain relevant for speed-sensitive applications and video.

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **FID** | Distribution distance in Inception feature space |
| **Memorization** | Generator copying training examples |
| **Implicit density** | Sample without $P(x)$ |
| **Mode collapse** | Loss of output diversity |
| **Conditional control** | Label-guided generation (CGAN) |

---

## Reflection Questions

1. Why does Foster use L1 nearest-neighbor instead of pixel-perfect equality for memorization?
2. Name two advantages VAEs retain over WGAN-GP despite softer images.
3. What evidence from CGAN Figure 4-17 shows factorized representations?
4. When would you still pick a GAN over a diffusion model in 2026?

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

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 4 — Summary & Analysis. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Heusel, M. et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID).
- Goodfellow, I. et al. (2014). Generative Adversarial Nets.

---

**Previous:** [Section 4.7 — Conditional GAN](./section-07-conditional-gan.md)  
**Next:** [Section 5.1 — Autoregressive Framework](../chapter-05-autoregressive-models/section-01-autoregressive-framework.md)
