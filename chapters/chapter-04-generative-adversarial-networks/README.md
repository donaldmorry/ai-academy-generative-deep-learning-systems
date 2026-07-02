# Chapter 04: Generative Adversarial Networks

> **Source:** *Generative Deep Learning (2nd ed.) — David Foster*, Chapter 4
> **Part:** Part II — Methods
> **Estimated time:** 12–15 hours
> **Prerequisites:** Course 4, Chapters 01–03 — generative framework, Keras CNNs, and VAE contrast; Course 3, Chapter 8 — optimization and gradient-based training

---

## Chapter Overview

Generative adversarial networks pit a generator against a discriminator in a minimax game. This chapter implements DCGAN on the Bricks dataset, studies GAN training tips and failure modes (mode collapse, vanishing gradients), then advances to Wasserstein GAN with gradient penalty (WGAN-GP) for stable training. You will finish with a conditional GAN (CGAN) that generates class-specific images. These adversarial foundations prepare you for advanced GANs in Chapter 10 and contrast with the explicit-likelihood methods in neighboring chapters.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain the GAN minimax objective and the roles of generator and discriminator
2. Implement a DCGAN with Conv2DTranspose generator and strided-conv discriminator in Keras
3. Diagnose GAN training pathologies: mode collapse, oscillation, and vanishing gradients
4. Apply practical GAN training tips: learning rates, label smoothing, and architecture choices
5. Implement WGAN-GP with Wasserstein loss and gradient penalty for Lipschitz constraint
6. Build a conditional GAN that accepts class labels to control generated image attributes
7. Evaluate GAN sample quality qualitatively and with simple metrics (e.g., FID overview)

---

## Sections

| # | Section | Topics |
|---|--------|--------|
| 4.1 | [GAN Introduction](./section-01-gan-introduction.md) | Minimax game; implicit generative models; historical impact |
| 4.2 | [DCGAN Architecture](./section-02-dcgan-architecture.md) | Bricks dataset; generator; discriminator; design guidelines |
| 4.3 | [Training the DCGAN](./section-03-training-the-dcgan.md) | Alternating updates; loss tracking; sample visualization |
| 4.4 | [GAN Tips and Tricks](./section-04-gan-tips-and-tricks.md) | Mode collapse; LR tuning; leaky ReLU; spectral norm preview |
| 4.5 | [Wasserstein GAN (WGAN-GP)](./section-05-wasserstein-gan-wgan-gp.md) | Wasserstein distance; Lipschitz constraint; critic |
| 4.6 | [Gradient Penalty](./section-06-gradient-penalty.md) | GP loss implementation; training stability improvements |
| 4.7 | [Conditional GAN](./section-07-conditional-gan.md) | Label conditioning; embedding; controlled generation |
| 4.8 | [Analysis & Comparison](./section-08-analysis-and-comparison.md) | Sample grids; failure case study; vs VAE trade-offs |

---

## Lab / Project

See also: [Lab 04](./section-lab-04-dcgan-wgan-gp-and-conditional-gan.md)

**Lab 04: DCGAN, WGAN-GP, and Conditional GAN**

1. Train a DCGAN on the Bricks dataset; log generator/discriminator losses and sample every N epochs.
2. Document one training failure (e.g., mode collapse) and the fix you applied.
3. Implement WGAN-GP; compare training stability to vanilla DCGAN.
4. Train a CGAN on MNIST with class-conditioned generation (generate specific digits).
5. *Deliverable:* Sample grids from all three models and a training stability comparison write-up.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Adversarial training concept | Game theory, Nash equilibria (Course 2, Ch 17) |
| Conv2DTranspose & CNNs | Image synthesis pipelines (Course 1, Chapter 10) |
| Implicit generative models | Contrast with MLE, VAEs (Course 3, Ch 20) |

---

## Prerequisites

- Course 4, Chapters 01–03 — generative framework, Keras CNNs, and VAE contrast
- Course 3, Chapter 8 — optimization and gradient-based training

---

## Self-Assessment

1. What are the generator and discriminator trying to optimize in a standard GAN?
2. List three DCGAN architecture guidelines and why each matters.
3. What is mode collapse, and what training changes can mitigate it?
4. How does WGAN-GP differ from the original GAN loss, and why is it more stable?
5. How does a conditional GAN incorporate class label information?
6. When would you choose a GAN over a VAE for image generation?

---

**Previous:** [Chapter 03 — Variational Autoencoders](../chapter-03-variational-autoencoders/README.md)
**Next:** [Chapter 05 — Autoregressive Models](../chapter-05-autoregressive-models/README.md)
