# Chapter 01: Generative Modeling

> **Source:** *Generative Deep Learning (2nd ed.) — David Foster*, Chapter 1
> **Part:** Part I — Introduction to Generative Deep Learning
> **Estimated time:** 8–10 hours
> **Prerequisites:** Course 1, Chapters 01–09 — ML fundamentals and introductory deep learning with Keras; Course 3, Chapter 3 — basic probability and information theory

---

## Chapter Overview

This chapter establishes the conceptual foundation for the entire course. You will distinguish generative from discriminative modeling, walk through Foster's first generative model (a simple coin-flip example), and build intuition for representation learning and the probabilistic framework underlying all generative techniques. Core probability theory—joint distributions, Bayes' rule, maximum likelihood—and a taxonomy of generative model families prepare you for the methods in Part II. You will also clone Foster's codebase, configure Docker, and verify GPU access for TensorFlow/Keras work in later chapters.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Distinguish generative modeling from discriminative modeling and explain when each applies
2. Describe the generative modeling framework: learn p(x) or p(x|z) to sample new data
3. Explain representation learning and why latent variables enable creative synthesis
4. Apply core probability concepts—joint, marginal, conditional distributions and Bayes' rule
5. Classify generative models by architecture: VAEs, GANs, flows, autoregressive, diffusion
6. Set up Foster's generative-deep-learning repository with Docker and GPU support
7. Run the book's Hello World generative example and interpret its output

---

## Sections

| # | Section | Topics |
|---|--------|--------|
| 1.1 | What Is Generative Modeling? | Creative AI; sampling; density estimation; use cases |
| 1.2 | Generative vs Discriminative | p(x) vs p(y|x); classification vs synthesis; decision boundaries |
| 1.3 | The Rise of Generative AI | Historical milestones; modern applications; industry landscape |
| 1.4 | Our First Generative Model | Coin-flip model; likelihood; sampling; framework walkthrough |
| 1.5 | Representation Learning | Latent spaces; feature hierarchies; unsupervised structure discovery |
| 1.6 | Core Probability Theory | Joint/marginal/conditional; Bayes; MLE; KL divergence preview |
| 1.7 | Generative Model Taxonomy | Explicit vs implicit; tractable likelihood; model family map |
| 1.8 | Codebase Setup | Git clone; Docker; TensorFlow GPU; project structure tour |

---

## Lab / Project

**Lab 01: Generative Modeling Foundations & Environment Setup**

1. Clone Foster's `generative-deep-learning` repository and build the Docker image.
2. Verify TensorFlow detects your GPU (`tf.config.list_physical_devices('GPU')`).
3. Implement and run the Chapter 1 coin-flip generative model; plot sampled sequences.
4. Write a one-page taxonomy diagram mapping each Part II chapter to its model family.
5. *Deliverable:* Working environment screenshot, coin-flip notebook, and taxonomy sketch.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Probability & Bayes' rule | Formal inference, graphical models (Course 2, Ch 12–14) |
| Representation learning | Autoencoders, manifold learning (Course 3, Ch 14–15) |
| TensorFlow/Keras setup | Neural network training loop (Course 1, Chapters 08–09) |

---

## Prerequisites

- Course 1, Chapters 01–09 — ML fundamentals and introductory deep learning with Keras
- Course 3, Chapter 3 — basic probability and information theory

---

## Self-Assessment

1. What is the fundamental difference between modeling p(x) and modeling p(y|x)?
2. Why do generative models require a notion of latent representation?
3. Name three generative model families covered in this course and one key property of each.
4. What does maximum likelihood estimation optimize, and why is it natural for generative models?
5. What steps are required to run Foster's codebase in Docker with GPU acceleration?
6. Give an example application where a generative model is preferable to a discriminative one.

---

**Next:** [Chapter 02 — Deep Learning](../chapter-02-deep-learning/README.md)
