# Chapter 07: Energy-Based Models

> **Source:** *Generative Deep Learning (2nd ed.) — David Foster*, Chapter 7
> **Part:** Part II — Methods
> **Estimated time:** 9–11 hours
> **Prerequisites:** Course 4, Chapters 01–02 — probabilistic framework and Keras CNNs; Course 2, Chapter 14 — probabilistic reasoning and sampling basics

---

## Chapter Overview

Energy-based models (EBMs) assign low energy to observed data and high energy elsewhere, defining distributions via Boltzmann distributions without normalized densities. You will implement an energy function on MNIST, sample with Langevin dynamics, and train with contrastive divergence. EBMs connect historically to Boltzmann machines and modern score-based methods that underpin diffusion models in Chapter 08. Understanding EBMs clarifies why sampling and training can be challenging when the partition function is intractable.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Define energy-based models and relate energy to unnormalized probability via Boltzmann distribution
2. Implement a neural network energy function E(x) for MNIST in Keras
3. Sample from EBMs using Langevin dynamics with gradient-based MCMC steps
4. Train EBMs with contrastive divergence: positive data vs negative samples
5. Analyze EBM training stability and common failure modes (divergent MCMC chains)
6. Connect EBMs to score matching and their role in modern generative modeling
7. Compare EBM sampling cost with GAN and flow model generation

---

## Sections

| # | Section | Topics |
|---|--------|--------|
| 7.1 | [EBM Introduction](./section-01-energy-based-models-introduction.md) | Energy functions; Boltzmann distribution; partition function |
| 7.2 | [MNIST Energy Function](./section-02-mnist-energy-function.md) | Network architecture; scalar energy output; Keras build |
| 7.3 | [Langevin Dynamics Sampling](./section-03-langevin-dynamics-sampling.md) | Gradient-based MCMC; step size; noise injection |
| 7.4 | [Contrastive Divergence](./section-04-contrastive-divergence.md) | Positive phase; negative phase; CD-k training |
| 7.5 | [Training the EBM](./section-05-training-the-ebm.md) | Loss formulation; hyperparameters; monitoring convergence |
| 7.6 | [Analysis of the EBM](./section-06-analysis-of-the-ebm.md) | Generated digits; energy landscape visualization |
| 7.7 | [Other EBMs & Connections](./section-07-other-ebms-and-connections.md) | Boltzmann machines; score matching link to diffusion |

---

## Lab / Project

See also: [Lab 07](./section-lab-07-ebm-on-mnist-with-langevin-sampling.md)

**Lab 07: EBM on MNIST with Langevin Sampling**

1. Build a CNN-based energy function for MNIST in Keras.
2. Implement Langevin dynamics sampling; tune step size and number of steps.
3. Train with contrastive divergence; log energy of real vs generated samples.
4. Generate 64 MNIST-like digits; discuss sample diversity and artifacts.
5. *Deliverable:* Generated samples, energy histograms, and MCMC hyperparameter notes.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Boltzmann distributions | Graphical models (Course 2, Ch 13–14) |
| MCMC & sampling | Inference methods (Course 2, Ch 14) |
| Score-based generative models | Diffusion model foundations (Course 3, Ch 20) |

---

## Prerequisites

- Course 4, Chapters 01–02 — probabilistic framework and Keras CNNs
- Course 2, Chapter 14 — probabilistic reasoning and sampling basics

---

## Self-Assessment

1. How does an EBM define a probability distribution without computing the partition function?
2. What is Langevin dynamics, and how is it used to sample from an EBM?
3. Explain contrastive divergence training in terms of positive and negative samples.
4. Why is EBM training often less stable than VAE or flow training?
5. How do EBMs connect to the score-based perspective used in diffusion models?
6. What hyperparameters most affect Langevin sampling quality?

---

**Previous:** [Chapter 06 — Normalizing Flow Models](../chapter-06-normalizing-flow-models/README.md)
**Next:** [Chapter 08 — Diffusion Models](../chapter-08-diffusion-models/README.md)
