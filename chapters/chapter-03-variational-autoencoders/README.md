# Chapter 03: Variational Autoencoders

> **Source:** *Generative Deep Learning (2nd ed.) — David Foster*, Chapter 3
> **Part:** Part II — Methods
> **Estimated time:** 12–14 hours
> **Prerequisites:** Course 4, Chapters 01–02 — generative framework and Keras CNN training; Course 3, Chapter 14 — autoencoders and representation learning

---

## Chapter Overview

Variational autoencoders (VAEs) are your first full generative deep learning system. You will build a standard autoencoder on Fashion-MNIST, visualize its latent space, then upgrade to a VAE with the reparameterization trick and ELBO loss. Training on CelebA unlocks latent space arithmetic—smiling vectors, gender directions—and face morphing between identities. VAEs trade sample sharpness for stable training and a principled probabilistic objective, making them the foundation for world models in Chapter 12 and a key contrast point against GANs and diffusion models.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Implement encoder-decoder autoencoder architectures in Keras for image reconstruction
2. Visualize and interpret low-dimensional latent spaces from trained autoencoders
3. Derive the VAE objective: reconstruction loss plus KL divergence to a prior
4. Apply the reparameterization trick for differentiable sampling from q(z|x)
5. Train a VAE on CelebA and generate novel face images by sampling the latent prior
6. Perform latent space arithmetic and morphing between face embeddings
7. Compare VAE sample quality and training stability against GAN approaches

---

## Sections

| # | Section | Topics |
|---|--------|--------|
| 3.1 | [Autoencoder Architecture](./section-01-autoencoder-architecture.md) | Encoder; decoder; bottleneck; Fashion-MNIST reconstruction |
| 3.2 | [Reconstructing Images](./section-02-reconstructing-images.md) | Loss functions; blurry outputs; capacity vs regularization |
| 3.3 | [Visualizing Latent Space](./section-03-visualizing-latent-space.md) | 2D embeddings; clustering; interpolation preview |
| 3.4 | [Variational Autoencoders](./section-04-variational-autoencoders.md) | Probabilistic encoder; prior p(z); ELBO derivation |
| 3.5 | [VAE Loss & Reparameterization](./section-05-vae-loss-and-reparameterization.md) | KL term; sampling z = μ + σε; Keras implementation |
| 3.6 | [Training the VAE](./section-06-training-the-vae.md) | Fashion-MNIST VAE; hyperparameters; monitoring ELBO |
| 3.7 | [CelebA Faces](./section-07-celeba-faces.md) | Dataset prep; deeper architecture; generating new faces |
| 3.8 | [Latent Space Arithmetic](./section-08-latent-space-arithmetic.md) | Vector arithmetic; morphing; semantic directions |

---

## Lab / Project

See also: [Lab 03](./section-lab-03-vae-on-celeba-with-latent-arithmetic.md)

**Lab 03: VAE on CelebA with Latent Arithmetic**

1. Train an autoencoder on Fashion-MNIST; visualize 2D latent space with t-SNE or direct projection.
2. Convert to a VAE; implement the combined reconstruction + KL loss in Keras.
3. Train on CelebA (64×64); sample 16 faces from N(0, I).
4. Perform latent arithmetic (e.g., neutral → smiling) and morph between two identities.
5. *Deliverable:* Grid of generated faces, arithmetic examples, and ELBO training curves.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Autoencoders & latent variables | Manifold learning, PCA (Course 1, Chapter 06; Course 3, Ch 14) |
| KL divergence & variational inference | Information theory, EM algorithm (Course 3, Ch 3, 19) |
| Convolutional encoders/decoders | CNN feature hierarchies (Course 1, Chapter 10) |

---

## Prerequisites

- Course 4, Chapters 01–02 — generative framework and Keras CNN training
- Course 3, Chapter 14 — autoencoders and representation learning

---

## Self-Assessment

1. What is the role of the bottleneck in an autoencoder?
2. Write the VAE loss as reconstruction plus KL divergence and explain each term.
3. Why is the reparameterization trick necessary for backpropagation through sampling?
4. Why are VAE-generated images often blurrier than GAN outputs?
5. How do you generate a new image from a trained VAE at inference time?
6. Describe one latent arithmetic experiment and what it reveals about the embedding.

---

**Previous:** [Chapter 02 — Deep Learning](../chapter-02-deep-learning/README.md)
**Next:** [Chapter 04 — Generative Adversarial Networks](../chapter-04-generative-adversarial-networks/README.md)
