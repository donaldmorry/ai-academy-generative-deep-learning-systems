# Chapter 05: Autoregressive Models

> **Source:** *Generative Deep Learning (2nd ed.) — David Foster*, Chapter 5
> **Part:** Part II — Methods
> **Estimated time:** 11–13 hours
> **Prerequisites:** Course 4, Chapters 01–02 — probability foundations and Keras training; Course 3, Chapter 10 — sequence modeling and recurrent networks

---

## Chapter Overview

Autoregressive models factorize the joint distribution as a product of conditionals, generating data one element at a time. You will train an LSTM on a recipe text corpus—covering tokenization, embeddings, and sequence generation—then extend to image generation with PixelCNN and masked convolutions. RNN extensions (stacked layers, GRU, bidirectional cells) bridge to the transformer architectures in Chapter 09. Autoregressive models offer tractable likelihoods but sequential sampling can be slow—a trade-off you will compare against flows and diffusion.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Factorize joint distributions into ordered conditionals for tractable likelihood estimation
2. Preprocess text data: tokenization, vocabulary building, and padded sequence batches
3. Build and train an LSTM text generator with Embedding and LSTM layers in Keras
4. Sample novel text sequences via temperature-controlled autoregressive decoding
5. Implement PixelCNN with masked convolutions for tractable image density estimation
6. Use residual blocks and mixture distributions in PixelCNN for improved image quality
7. Compare autoregressive generation speed and quality against parallel methods (GANs, diffusion)

---

## Sections

| # | Section | Topics |
|---|--------|--------|
| 5.1 | [Autoregressive Framework](./section-01-autoregressive-framework.md) | Chain rule factorization; ordering choices; likelihood |
| 5.2 | [LSTM for Text](./section-02-lstm-for-text.md) | Recipes dataset; tokenization; training set creation |
| 5.3 | [Embeddings & LSTM Layers](./section-03-embeddings-and-lstm-layers.md) | Embedding layer; LSTM cell internals; Keras implementation |
| 5.4 | [Training & Sampling Text](./section-04-training-and-sampling-text.md) | Teacher forcing; temperature; nucleus sampling preview |
| 5.5 | [RNN Extensions](./section-05-rnn-extensions.md) | Stacked RNNs; GRU; bidirectional cells; when to use each |
| 5.6 | [PixelCNN](./section-06-pixelcnn.md) | Masked convolutions; causal ordering over pixels; architecture |
| 5.7 | [Residual Blocks & Training](./section-07-residual-blocks-and-training-the-pixelcnn.md) | Gated ResNet blocks; training PixelCNN on images |
| 5.8 | [Mixture Distributions](./section-08-mixture-distributions-for-pixelcnn.md) | Discretized logistic mixture; improved density estimation |

---

## Lab / Project

See also: [Lab 05](./section-lab-05-lstm-recipe-generator-and-pixelcnn.md)

**Lab 05: LSTM Recipe Generator & PixelCNN**

1. Train an LSTM on the Recipes dataset; generate 5 novel recipes at temperatures 0.5, 1.0, 1.5.
2. Implement masked convolutions for a PixelCNN on a small image dataset (e.g., MNIST).
3. Generate images pixel-by-pixel; visualize partial generation progress.
4. Compare sample quality and generation time vs your Chapter 04 GAN.
5. *Deliverable:* Generated recipes, PixelCNN samples, and speed/quality comparison table.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| RNNs & LSTMs | Sequence modeling (Course 3, Ch 10; Course 1, Chapter 13) |
| Text tokenization & embeddings | NLP pipelines (Course 1, Chapter 13) |
| Tractable likelihood | Maximum likelihood framework (Course 3, Ch 3) |

---

## Prerequisites

- Course 4, Chapters 01–02 — probability foundations and Keras training
- Course 3, Chapter 10 — sequence modeling and recurrent networks

---

## Self-Assessment

1. How does an autoregressive model factorize p(x₁, …, xₙ)?
2. What is teacher forcing, and what exposure bias problem does it create?
3. How do masked convolutions enforce causal ordering in PixelCNN?
4. What effect does temperature have on autoregressive text sampling?
5. Why is autoregressive image generation slower than GAN sampling?
6. What advantage do autoregressive models have over GANs regarding likelihood?

---

**Previous:** [Chapter 04 — Generative Adversarial Networks](../chapter-04-generative-adversarial-networks/README.md)
**Next:** [Chapter 06 — Normalizing Flow Models](../chapter-06-normalizing-flow-models/README.md)
