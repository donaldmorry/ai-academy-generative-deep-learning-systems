# Chapter 12: World Models

> **Source:** *Generative Deep Learning (2nd ed.) — David Foster*, Chapter 12
> **Part:** Part III — Applications
> **Estimated time:** 12–14 hours
> **Prerequisites:** Course 4, Chapter 03 — VAE architecture and latent space training; Course 2, Chapter 17 — reinforcement learning fundamentals

---

## Chapter Overview

World models combine generative modeling with reinforcement learning: a VAE compresses high-dimensional observations, an MDN-RNN predicts latent dynamics, and a controller learns policies—often trained in the 'dream' of the generative model. Using the CarRacing environment, you will collect rollouts, train each component in Keras, and optimize the controller with CMA-ES. This chapter shows how generative models enable sample-efficient RL by learning compact environment simulations.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain the world model framework: VAE + MDN-RNN + controller for RL
2. Collect and preprocess rollout data from the CarRacing Gym environment
3. Train a VAE to compress high-dimensional observations into a compact latent state
4. Implement an MDN-RNN that predicts latent dynamics with mixture density outputs
5. Train a controller policy optimized via CMA-ES in the learned latent space
6. Perform in-dream training: train the controller entirely inside the generative simulation
7. Evaluate the agent's performance in the real environment vs training in the dream

---

## Sections

| # | Section | Topics |
|---|--------|--------|
| 12.1 | [World Models Introduction](./section-01-world-models-introduction.md) | RL + generative models; motivation; Ha & Schmidhuber |
| 12.2 | [Reinforcement Learning Basics](./section-02-reinforcement-learning-basics.md) | Policy; reward; CarRacing environment setup |
| 12.3 | [World Model Architecture](./section-03-world-model-architecture.md) | VAE + MDN-RNN + controller; data flow diagram |
| 12.4 | [Collecting Rollout Data](./section-04-collecting-rollout-data.md) | Random policy rollouts; preprocessing; dataset structure |
| 12.5 | [Training the VAE](./section-05-training-the-vae.md) | Encoder-decoder on frames; latent space exploration |
| 12.6 | [Training the MDN-RNN](./section-06-training-the-mdn-rnn.md) | Mixture density network; temporal prediction; sampling |
| 12.7 | [Training the Controller](./section-07-training-the-controller.md) | CMA-ES optimization; parallel evaluation; reward shaping |
| 12.8 | [In-Dream Training](./section-08-in-dream-training.md) | Training inside generative simulation; transfer to real env |

---

## Lab / Project

See also: [Lab 12](./section-lab-12-world-model-on-carracing.md)

**Lab 12: World Model on CarRacing**

1. Collect 10,000 random rollouts from CarRacing-v0; preprocess frames.
2. Train the VAE component; visualize latent space of road observations.
3. Train the MDN-RNN on latent sequences; sample predicted trajectories.
4. Optimize a controller with CMA-ES in the dream environment.
5. Evaluate the controller in the real CarRacing environment; report episode rewards.
6. *Deliverable:* Training curves, dream vs real rollout videos, and final reward score.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| VAE latent representations | Chapter 03 VAE training (Course 4, Chapter 03) |
| Reinforcement learning | MDP agents, policies (Course 2, Ch 17) |
| Sequence prediction | RNN/MDN dynamics (Course 3, Ch 10) |

---

## Prerequisites

- Course 4, Chapter 03 — VAE architecture and latent space training
- Course 2, Chapter 17 — reinforcement learning fundamentals

---

## Self-Assessment

1. What are the three components of a world model, and what does each do?
2. Why is a VAE useful for compressing CarRacing observations?
3. What does the MDN-RNN predict, and why use a mixture density output?
4. What is CMA-ES, and why is it suited for controller optimization here?
5. What is in-dream training, and what advantage does it provide?
6. How do you evaluate whether the world model accurately simulates the environment?

---

**Previous:** [Chapter 11 — Music Generation](../chapter-11-music-generation/README.md)
**Next:** [Chapter 13 — Multimodal Models](../chapter-13-multimodal-models/README.md)
