# Chapter 10: Advanced GANs

> **Source:** *Generative Deep Learning (2nd ed.) — David Foster*, Chapter 10
> **Part:** Part III — Applications
> **Estimated time:** 11–13 hours
> **Prerequisites:** Course 4, Chapters 04 and 09 — GAN training and transformer/attention concepts; Course 3, Chapter 9 — advanced CNN architectures

---

## Chapter Overview

This chapter surveys state-of-the-art GAN architectures that pushed photorealistic image synthesis before diffusion models dominated. You will study ProGAN's progressive training, StyleGAN's mapping and synthesis networks, StyleGAN2 improvements (weight demodulation, path length regularization), and survey SAGAN, BigGAN, VQ-GAN, and ViT-VQGAN. These architectures introduce techniques—self-attention, vector quantization, progressive growing—that appear across modern generative systems including multimodal models.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain progressive growing in ProGAN and its impact on high-resolution training stability
2. Describe StyleGAN's mapping network, style-based generator, and synthesis pipeline
3. Identify StyleGAN2 improvements: weight demodulation, path length regularization, no progressive growing
4. Understand self-attention in SAGAN for capturing long-range image dependencies
5. Explain BigGAN's class-conditional batch normalization and scaling strategies
6. Describe VQ-GAN and ViT-VQGAN: vector quantization bridging GANs and transformers
7. Evaluate when advanced GANs remain relevant vs diffusion-based alternatives

---

## Sections

| # | Section | Topics |
|---|--------|--------|
| 10.1 | [Advanced GANs Overview](./section-01-advanced-gans-overview.md) | Evolution from DCGAN to StyleGAN2; landscape survey |
| 10.2 | [ProGAN](./section-02-progan.md) | Progressive training; growing resolution; fade-in layers |
| 10.3 | [StyleGAN Architecture](./section-03-stylegan-architecture.md) | Mapping network; AdaIN; style vectors; synthesis network |
| 10.4 | [StyleGAN2](./section-04-stylegan2.md) | Weight demodulation; path length regularization; architecture cleanup |
| 10.5 | [Self-Attention GAN (SAGAN)](./section-05-self-attention-gan-sagan.md) | Spectral normalization; attention maps; stability |
| 10.6 | [BigGAN](./section-06-biggan.md) | Class conditioning; large batch training; truncation trick |
| 10.7 | [VQ-GAN](./section-07-vq-gan.md) | Vector quantization; codebook; autoregressive prior on tokens |
| 10.8 | [ViT-VQGAN & Summary](./section-08-vit-vqgan-and-summary.md) | Vision transformer encoder; two-stage generation; comparison |

---

## Lab / Project

See also: [Lab 10](./section-lab-10-stylegan2-exploration-and-vq-gan-concepts.md)

**Lab 10: StyleGAN2 Exploration & VQ-GAN Concepts**

1. Run a pretrained StyleGAN2 inference notebook; explore latent space interpolation.
2. Implement a minimal style-based modulation layer (AdaIN) in Keras.
3. Sketch the VQ-GAN two-stage pipeline: encoder → quantizer → decoder → prior.
4. Compare StyleGAN2 face samples with your Chapter 03 VAE and Chapter 08 diffusion outputs.
5. *Deliverable:* Interpolation gallery, AdaIN code snippet, and GAN vs diffusion comparison.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Self-attention mechanisms | Transformers (Course 4, Chapter 09; Course 3, Ch 12) |
| Adversarial training | DCGAN, WGAN-GP foundations (Course 4, Chapter 04) |
| Vector quantization | Discrete latent codes in multimodal models (Course 4, Chapter 13) |

---

## Prerequisites

- Course 4, Chapters 04 and 09 — GAN training and transformer/attention concepts
- Course 3, Chapter 9 — advanced CNN architectures

---

## Self-Assessment

1. How does ProGAN's progressive training improve high-resolution GAN stability?
2. What is the role of the mapping network in StyleGAN?
3. What problem does StyleGAN2's weight demodulation solve?
4. How does self-attention in SAGAN differ from transformer self-attention?
5. What is the two-stage pipeline in VQ-GAN?
6. When would you choose a modern diffusion model over StyleGAN2?

---

**Previous:** [Chapter 09 — Transformers](../chapter-09-transformers/README.md)
**Next:** [Chapter 11 — Music Generation](../chapter-11-music-generation/README.md)
