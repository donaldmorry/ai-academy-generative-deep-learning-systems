# Chapter 08: Diffusion Models

> **Source:** *Generative Deep Learning (2nd ed.) — David Foster*, Chapter 8
> **Part:** Part II — Methods
> **Estimated time:** 13–15 hours
> **Prerequisites:** Course 4, Chapters 03–07 — VAEs, GANs, and energy-based model foundations; Course 3, Chapter 9 — U-Net-style architectures and deep CNN design

---

## Chapter Overview

Denoising diffusion models have become the dominant approach for high-quality image synthesis. This chapter implements denoising diffusion probabilistic models (DDPMs) on the Flowers dataset: forward noising process, reverse denoising with a U-Net, diffusion schedules, and iterative sampling. You will use the reparameterization trick, train a noise-prediction network, and generate images through the full reverse process. This chapter directly enables understanding Stable Diffusion and Imagen in Chapter 13.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Describe the forward diffusion process and how it gradually adds Gaussian noise
2. Derive the reverse diffusion process and the role of learned noise prediction
3. Design diffusion schedules (linear, cosine) and understand their impact on generation
4. Build a U-Net denoising model with time-step conditioning in TensorFlow/Keras
5. Train a DDPM on the Flowers dataset using the reparameterization trick
6. Implement the full iterative sampling loop to generate images from pure noise
7. Analyze sample quality and connect DDPMs to score-based and EBM perspectives

---

## Sections

| # | Section | Topics |
|---|--------|--------|
| 8.1 | [Diffusion Introduction](./section-01-diffusion-models-introduction.md) | DDPM overview; motivation; comparison with GANs/VAEs |
| 8.2 | [Forward Diffusion Process](./section-02-forward-diffusion-process.md) | Markov chain; noise schedule; closed-form q(xₜ|x₀) |
| 8.3 | [Reparameterization & Schedules](./section-03-reparameterization-and-diffusion-schedules.md) | Sampling xₜ; linear vs cosine β schedules |
| 8.4 | [Reverse Diffusion Process](./section-04-reverse-diffusion-process.md) | Learned p(xₜ₋₁|xₜ); noise prediction objective |
| 8.5 | [U-Net Denoising Model](./section-05-u-net-denoising-model.md) | Encoder-decoder; skip connections; time embedding |
| 8.6 | [Training the Diffusion Model](./section-06-training-the-diffusion-model.md) | Flowers dataset; loss function; Keras training loop |
| 8.7 | [Sampling from DDPM](./section-07-sampling-from-ddpm.md) | Iterative denoising; number of steps; sample visualization |
| 8.8 | [Analysis & Connections](./section-08-analysis-and-connections.md) | Sample quality; link to score matching; Stable Diffusion preview |

---

## Lab / Project

See also: [Lab 08](./section-lab-08-ddpm-on-flowers-with-u-net.md)

**Lab 08: DDPM on Flowers with U-Net**

1. Implement the forward diffusion process with a cosine noise schedule.
2. Build a time-conditioned U-Net denoiser in Keras; embed diffusion timestep.
3. Train on the Flowers dataset; monitor noise-prediction loss.
4. Run the full reverse sampling loop (1000 steps or DDIM-accelerated variant).
5. *Deliverable:* Generated flower images, training curves, and schedule comparison notes.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| U-Net encoder-decoder | Semantic segmentation patterns (Course 1, Chapter 10) |
| Score matching & EBMs | Energy-based perspective (Course 4, Chapter 07) |
| Modern image synthesis | Stable Diffusion architecture (Course 4, Chapter 13) |

---

## Prerequisites

- Course 4, Chapters 03–07 — VAEs, GANs, and energy-based model foundations
- Course 3, Chapter 9 — U-Net-style architectures and deep CNN design

---

## Self-Assessment

1. What happens during the forward diffusion process?
2. What does the denoising network learn to predict, and why?
3. Why is a U-Net well-suited as the denoising architecture?
4. How does the reparameterization trick simplify training at arbitrary timesteps?
5. What is the trade-off between more diffusion steps and sample quality?
6. How do diffusion models relate to the score-based view of EBMs?
7. What role does time-step embedding play in the U-Net?

---

**Previous:** [Chapter 07 — Energy-Based Models](../chapter-07-energy-based-models/README.md)
**Next:** [Chapter 09 — Transformers](../chapter-09-transformers/README.md)
