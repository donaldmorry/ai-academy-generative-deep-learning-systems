# Chapter 02: Deep Learning

> **Source:** *Generative Deep Learning (2nd ed.) — David Foster*, Chapter 2
> **Part:** Part I — Introduction to Generative Deep Learning
> **Estimated time:** 10–12 hours
> **Prerequisites:** Course 1, Chapters 08–10 — deep learning foundations and CNN introduction; Course 3, Chapters 6–9 — neural networks, CNNs, and optimization

---

## Chapter Overview

Before building generative architectures, this chapter refreshes and extends the deep learning toolkit used throughout the book. You will prepare image data for TensorFlow/Keras, build and train a multilayer perceptron (MLP) on tabular data, then construct a convolutional neural network (CNN) with batch normalization and dropout. These components—conv layers, training loops, evaluation metrics—reappear in every generative model from DCGANs to diffusion U-Nets. Foster's MNIST and image-classification examples mirror the patterns you will reuse when training encoders, decoders, and discriminators.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Prepare and normalize image datasets for Keras training pipelines
2. Build, compile, and train an MLP using the Keras Sequential and Functional APIs
3. Explain how CNNs learn hierarchical spatial features via convolution and pooling
4. Implement batch normalization and dropout for stable, regularized training
5. Evaluate classification performance and interpret training/validation curves
6. Connect CNN building blocks to upcoming generator and discriminator architectures
7. Profile GPU memory usage during training to plan for generative workloads

---

## Sections

| # | Section | Topics |
|---|--------|--------|
| 2.1 | [Data for Deep Learning](./section-01-data-for-deep-learning.md) | Normalization; train/val/test splits; tf.data pipelines |
| 2.2 | [What Is a Neural Network?](./section-02-what-is-a-neural-network.md) | Layers; activations; forward pass; high-level features |
| 2.3 | [TensorFlow and Keras](./section-03-tensorflow-and-keras.md) | tf.keras APIs; Model subclassing preview; eager execution |
| 2.4 | [Multilayer Perceptron](./section-04-multilayer-perceptron.md) | Data prep; model build; compile; train; evaluate |
| 2.5 | [MLP Training Deep Dive](./section-05-mlp-training-deep-dive.md) | Loss functions; optimizers; callbacks; overfitting signs |
| 2.6 | [Convolutional Layers](./section-06-convolutional-layers.md) | Filters; stride; padding; feature maps; spatial hierarchy |
| 2.7 | [Batch Normalization & Dropout](./section-07-batch-normalization-and-dropout.md) | Internal covariate shift; regularization; inference mode |
| 2.8 | [Building and Training a CNN](./section-08-building-and-training-a-cnn.md) | Architecture design; training loop; evaluation on images |

---

## Lab / Project

See also: [Lab 02](./section-lab-02-mlp-and-cnn-training-with-keras.md)

**Lab 02: MLP and CNN Training with Keras**

1. Train an MLP on a tabular dataset from the book (or UCI fallback) using Keras.
2. Build a CNN for MNIST or Fashion-MNIST with Conv2D, BatchNorm, Dropout, and Dense head.
3. Plot training/validation accuracy and loss; diagnose overfitting.
4. Export the trained CNN weights—you will reuse Conv2D patterns in Chapter 04 (DCGAN).
5. *Deliverable:* Notebook with both models, learning curves, and a short architecture comparison.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| CNNs & conv layers | Image classification, ResNets (Course 1, Chapter 10; Course 3, Ch 9) |
| Training loops & optimizers | SGD, Adam, regularization (Course 3, Ch 8) |
| Keras Functional API | Applied DL workflows (Course 1, Chapters 08–09) |

---

## Prerequisites

- Course 1, Chapters 08–10 — deep learning foundations and CNN introduction
- Course 3, Chapters 6–9 — neural networks, CNNs, and optimization

---

## Self-Assessment

1. Why are CNNs preferred over fully connected networks for image data?
2. What problem does batch normalization address during training?
3. Walk through the Keras steps: build → compile → fit → evaluate.
4. How does dropout regularize a network, and what changes at inference time?
5. What output activation and loss would you use for 10-class image classification?
6. Which Chapter 2 components will reappear in a DCGAN discriminator?

---

**Previous:** [Chapter 01 — Generative Modeling](../chapter-01-generative-modeling/README.md)
**Next:** [Chapter 03 — Variational Autoencoders](../chapter-03-variational-autoencoders/README.md)
