# Chapter 06: Normalizing Flow Models

> **Source:** *Generative Deep Learning (2nd ed.) — David Foster*, Chapter 6
> **Part:** Part II — Methods
> **Estimated time:** 10–12 hours
> **Prerequisites:** Course 4, Chapters 01–03 — probability, MLE, and latent variable models; Course 3, Chapter 4 — linear algebra and Jacobian intuition

---

## Chapter Overview

Normalizing flows transform a simple base distribution through invertible mappings to model complex data distributions with exact likelihood computation. This chapter covers the change-of-variables formula, Jacobian determinants, and the RealNVP architecture with coupling layers trained on the Two Moons dataset. You will also survey GLOW and FFJORD as representative advanced flow models. Flows offer bijective mappings and tractable densities but architectural constraints differ fundamentally from VAEs and GANs.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain the change-of-variables formula and its role in computing exact log-likelihoods
2. Compute Jacobian determinants for affine coupling transformations
3. Implement RealNVP coupling layers with alternating masks in TensorFlow/Keras
4. Train a RealNVP model on Two Moons and visualize learned density and samples
5. Compare flow-based models with VAEs and autoregressive models on likelihood tractability
6. Describe GLOW (invertible 1×1 convolutions) and FFJORD (continuous normalizing flows)
7. Identify architectural constraints (invertibility, dimension preservation) of flow models

---

## Sections

| # | Section | Topics |
|---|--------|--------|
| 6.1 | [Normalizing Flows Introduction](./section-01-normalizing-flows-introduction.md) | Bijective maps; base distribution; exact density |
| 6.2 | [Change of Variables](./section-02-change-of-variables.md) | Jacobian determinant; log-det-Jacobian; composition of flows |
| 6.3 | [RealNVP Architecture](./section-03-realnvp-architecture.md) | Affine coupling layers; alternating masks; scale-and-translate |
| 6.4 | [Two Moons Dataset](./section-04-two-moons-dataset.md) | 2D density estimation; visualization; sampling |
| 6.5 | [Training RealNVP](./section-05-training-realnvp.md) | Negative log-likelihood loss; Keras implementation; monitoring |
| 6.6 | [Analysis of RealNVP](./section-06-analysis-of-realnvp.md) | Learned densities; sample quality; failure modes |
| 6.7 | [GLOW](./section-07-glow.md) | Invertible 1×1 conv; ActNorm; multi-scale architecture overview |
| 6.8 | [FFJORD & Summary](./section-08-ffjord-and-normalizing-flows-summary.md) | Neural ODE flows; continuous transformations; model comparison |

---

## Lab / Project

See also: [Lab 06](./section-lab-06-realnvp-on-two-moons.md)

**Lab 06: RealNVP on Two Moons**

1. Implement affine coupling layers with alternating checkerboard masks in Keras.
2. Train RealNVP on the Two Moons dataset; plot learned log-density contours.
3. Sample 1000 points from the trained flow; compare to ground-truth distribution.
4. Write a comparison table: flows vs VAEs vs autoregressive on likelihood, speed, flexibility.
5. *Deliverable:* Density contour plots, samples, and model comparison table.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Jacobian & change of variables | Calculus for ML (Course 3, Ch 4) |
| Exact likelihood models | MLE and probabilistic modeling (Course 3, Ch 3, 20) |
| Invertible architectures | Bijective constraints vs standard CNNs (Course 3, Ch 9) |

---

## Prerequisites

- Course 4, Chapters 01–03 — probability, MLE, and latent variable models
- Course 3, Chapter 4 — linear algebra and Jacobian intuition

---

## Self-Assessment

1. State the change-of-variables formula for a bijective transformation f.
2. Why must normalizing flow layers be invertible with tractable Jacobian determinants?
3. How do RealNVP coupling layers achieve efficient Jacobian computation?
4. What is the training loss for a flow model, and how does it relate to MLE?
5. Compare normalizing flows with VAEs on likelihood tractability and sample quality.
6. What innovation does GLOW add over basic coupling-layer flows?

---

**Previous:** [Chapter 05 — Autoregressive Models](../chapter-05-autoregressive-models/README.md)
**Next:** [Chapter 07 — Energy-Based Models](../chapter-07-energy-based-models/README.md)
