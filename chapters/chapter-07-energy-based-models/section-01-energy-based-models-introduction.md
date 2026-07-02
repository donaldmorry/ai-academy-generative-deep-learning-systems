# Section 7.1: Energy-Based Models Introduction

> **Source inheritance:** Foster, Ch. 7 — "Energy-Based Models"  
> **Enhanced with:** Boltzmann distributions, Diane Mixx story, unnormalized densities, and contrastive divergence preview  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

An **energy-based model (EBM)** assigns a scalar **energy** $E_\theta(x)$ to each input — low energy means "plausible," high energy means "unlikely." Probabilities follow a Boltzmann distribution:

$$
p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z_\theta}, \quad Z_\theta = \int e^{-E_\theta(x)}\, dx
$$
> **Readable form:** Probability is high when energy is low, but we must divide by partition function Z to normalize.

The partition function $Z_\theta$ is **intractable** in high dimensions — you cannot integrate over all 32×32 MNIST images. EBMs sidestep this: train with **contrastive divergence** ([Section 7.4](./section-04-contrastive-divergence.md)), sample with **Langevin dynamics** ([Section 7.3](./section-03-langevin-dynamics-sampling.md)). No exact $p(x)$ needed — only energy **comparisons**.

Foster's Diane Mixx story: a running club where low times (low energy) are probable; the scoreboard doesn't need global normalization to rank runners.

---

## Diane Mixx and the Long-au-Vin Club

| Story element | EBM role |
|---------------|----------|
| Runner finishing time | Energy $E(x)$ — lower is better |
| Club record board | Relative ranking of plausibility |
| New runner | Sample $x$ with low energy |
| Training | Push real times down, fake times up |

Unlike [normalizing flows](../chapter-06-normalizing-flow-models/section-01-normalizing-flows-introduction.md), EBMs do **not** force $p(x)$ to integrate to 1 after every transform. Unlike [GANs](../chapter-04-generative-adversarial-networks/section-01-gan-introduction.md), there is no adversary — only an energy landscape shaped by contrastive gradients.

---

## Boltzmann Distribution Intuition

For MNIST digit $x$:

- Real digit images → $E_\theta(x) \approx -5$ (low)
- Random noise → $E_\theta(x) \approx +10$ (high)
- Blurry almost-digit → $E_\theta(x) \approx 0$ (middle)

After training, $\exp(-E(x))$ is **unnormalized** — only **ratios** $p(x_1)/p(x_2) = \exp(-(E(x_1)-E(x_2)))$ are meaningful without computing $Z$.

---

## Two Core Challenges

**Challenge 1 — Sampling:** $E_\theta$ maps image → scalar. How do you find images with **low** energy? Answer: gradient descent on $x$ (Langevin dynamics), not on $\theta$.

**Challenge 2 — Training:** MLE needs $\log p_\theta(x) = -E_\theta(x) - \log Z_\theta$. The $\log Z_\theta$ term is intractable. Answer: contrastive divergence approximates the gradient without $Z$:

$$
\nabla_\theta \mathcal{L} \approx \mathbb{E}_{x \sim \text{data}}[\nabla_\theta E_\theta(x)] - \mathbb{E}_{x \sim \text{model}}[\nabla_\theta E_\theta(x)]
$$
> **Readable form:** Raise energy on fakes, lower on reals — push the distributions apart.

---

## EBM vs Other Generative Families

| | EBM | Flow | VAE | GAN |
|---|-----|------|-----|-----|
| Normalized $p(x)$ | No (implicit via $Z$) | Yes | ELBO | No |
| Sampling | Iterative (Langevin) | 1 inverse pass | 1 decoder pass | 1 generator pass |
| Training | Contrastive divergence | NLL | ELBO | Adversarial |
| Network output | Scalar energy | Bijection | Mean/logvar | Image or logit |

EBMs connect to **score-based** and **diffusion** models ([Chapter 08](../chapter-08-diffusion-models/section-01-diffusion-models-introduction.md)): denoising score matching learns $\nabla_x \log p(x)$ directly, avoiding $Z$ another way.

---

## Historical Context

| Era | Milestone |
|-----|-----------|
| 1980s | Hopfield nets, Boltzmann machines |
| 2002 | Hinton — contrastive divergence |
| 2019 | Du & Mordatch — deep EBMs on images |
| 2019–2020 | Song & Ermon — score matching → NCSN |
| 2020 | DDPM — diffusion unifies with scores |

Modern diffusion training is spiritually descended from energy/score modeling, even when Foster's EBM notebook is the hands-on entry point.

---

## Chapter 07 Roadmap

| Section | Topic |
|--------|-------|
| 7.1 | Introduction (this section) |
| 7.2 | MNIST energy network architecture |
| 7.3 | Langevin sampling |
| 7.4 | Contrastive divergence |
| 7.5 | Full training loop |
| 7.6 | Analysis |
| 7.7 | Other EBMs & connections |

Notebook: `notebooks/07_ebm/01_ebm/ebm.ipynb` (adapted from Phillip Lippe's tutorial).

---

## Minimal Energy Forward Pass Preview

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_energy_fn():
  inp = layers.Input(shape=(32, 32, 1))
  x = layers.Conv2D(16, 5, strides=2, padding="same")(inp)
  x = layers.Flatten()(x)
  energy = layers.Dense(1)(x)  # scalar — no softmax
  return models.Model(inp, energy)
```

Output is a **single number**, not a class distribution. Training details in [Sections 7.4–7.5](./section-04-contrastive-divergence.md).

---

## Energy Landscape Intuition (2D Sketch)

Imagine a terrain over two pixel dimensions (conceptually):

```
High E  ▲     random noise lives on mountaintops
        │    ╱╲
        │   ╱  ╲___ digit manifold valley
Low E   │__╱         real MNIST digits sit here
        └──────────────────► pixel space
```

Langevin sampling rolls balls downhill from random peaks toward valleys. Contrastive divergence shovels earth **out** of valleys (raise fake energy) and **into** them (lower real energy) without measuring the total volume of the terrain ($Z$).

---

## When EBMs Still Matter

Despite diffusion dominance for image synthesis, energy/score thinking remains active in:

- **Anomaly detection** — flag inputs with $E(x) > \tau$
- **Adversarial robustness research** — energy gaps and calibration
- **Compositional generation** — products of experts sum energies
- **Theoretical ML** — connecting MCMC, thermodynamics, and deep nets

Foster's MNIST EBM is a minimal sandbox for these ideas before you study DDPM's denoising objective in Chapter 08.

---

## Notebook and Repository Pointers

| Resource | Path |
|----------|------|
| Foster EBM notebook | `notebooks/07_ebm/01_ebm/ebm.ipynb` |
| Lippe tutorial (adapted) | Cited in Foster Ch. 7 |
| Du & Mordatch reference | Deep EBM on harder datasets |

Run the notebook after reading Sections 7.2–7.5 — the `Buffer` and `train_step` code maps directly to Foster's Examples 7-4 through 7-6.

---

## Pre-Flight Questions

Answer before opening the notebook:

1. What is $Z_\theta$ and why do we ignore it during CD?
2. Which tensor gets `tape.watch` in Langevin — weights or pixels?
3. What does a widening `fake - real` gap indicate?
4. How is swish different from ReLU for $\nabla_x E$?

If any answer is shaky, re-read [Sections 7.2–7.4](./section-02-mnist-energy-function.md) before training.

---

## Chapter 07 at a Glance

| Section | One-line summary |
|--------|------------------|
| 7.1 | Boltzmann + Diane Mixx story |
| 7.2 | Swish CNN → scalar energy |
| 7.3 | Langevin on pixels |
| 7.4 | CD loss + replay buffer |
| 7.5 | Custom Keras `EBM` class |
| 7.6 | Histograms & sample grids |
| 7.7 | Scores, NCSN, diffusion bridge |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Energy-based model** | Generative model via scalar energy $E_\theta(x)$ |
| **Boltzmann distribution** | $p(x) \propto e^{-E(x)}$ |
| **Partition function $Z$** | Normalizing constant — intractable in high-D |
| **Contrastive divergence** | Training without computing $Z$ |
| **Langevin dynamics** | Gradient-based sampling in input space |

---

## Reflection Questions

1. Why is the partition function intractable for 32×32 images?
2. How does an EBM differ from a GAN discriminator output?
3. What does "unnormalized" mean for day-to-day training?
4. How might score-based models avoid both $Z$ and explicit adversaries?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 7 — Introduction. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Du, Y. & Mordatch, I. (2019). Implicit Generation and Modeling with Energy-Based Models. [https://arxiv.org/abs/1903.08594](https://arxiv.org/abs/1903.08594)
- Hinton, G. (2002). Training Products of Experts by Minimizing Contrastive Divergence.

---

**Previous:** [Chapter 07 Overview](./README.md)  
**Next:** [Section 7.2 — MNIST Energy Function](./section-02-mnist-energy-function.md)
