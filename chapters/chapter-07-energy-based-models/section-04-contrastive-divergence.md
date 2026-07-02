# Section 7.4: Contrastive Divergence

> **Source inheritance:** Foster, Ch. 7 — "Training with Contrastive Divergence"  
> **Enhanced with:** CD gradient derivation, replay buffer, positive/negative phases, and loss without partition function  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

**Contrastive divergence (CD)**, introduced by Hinton (2002), trains [EBMs](./section-01-energy-based-models-introduction.md) without computing the partition function $Z_\theta$. The MLE gradient for Boltzmann $p_\theta(x) \propto e^{-E_\theta(x)}$ simplifies to:

$$
\nabla_\theta \mathcal{L} = \mathbb{E}_{x \sim p_{\text{data}}}[\nabla_\theta E_\theta(x)] - \mathbb{E}_{x \sim p_\theta}[\nabla_\theta E_\theta(x)]
$$
> **Readable form:** Lower energy on real images; raise energy on model samples — maximize the gap.

We approximate model samples with short [Langevin chains](./section-03-langevin-dynamics-sampling.md) from a **replay buffer** of past fakes. One training step: push down real energies, pull up fake energies.

---

## Why MLE Needs CD

Naive NLL:

$$
\mathcal{L} = -\mathbb{E}_{x \sim \text{data}} \left[ \log p_\theta(x) \right] = \mathbb{E}\left[ E_\theta(x) + \log Z_\theta \right]
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

$\log Z_\theta = \log \int e^{-E_\theta(x)} dx$ is intractable. Taking $\nabla_\theta$ kills $Z$ in the CD derivation (telescoping under mild conditions) — you never evaluate $Z$, only **energy differences** between data and samples.

---

## Contrastive Loss in Practice

Foster's per-batch loss:

$$
\mathcal{L}_{\text{CD}} = \mathbb{E}_{x^- \sim \text{fakes}}[E_\theta(x^-)] - \mathbb{E}_{x^+ \sim \text{data}}[E_\theta(x^+)]
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

Plus **regularization** penalizing huge energies:

$$
\mathcal{L}_{\text{reg}} = \alpha \mathbb{E}[E_\theta(x^+)^2 + E_\theta(x^-)^2]
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

```python
cdiv_loss = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out)
reg_loss = alpha * tf.reduce_mean(real_out ** 2 + fake_out ** 2)
loss = cdiv_loss + reg_loss
```

**Positive phase:** real MNIST batch (slight input noise for augmentation). **Negative phase:** Langevin samples from buffer.

---

## Replay Buffer

Pure random noise starts every Langevin chain wastes compute. Foster's `Buffer` class:

1. Initialize 128 uniform random images
2. Each step: 5% fresh noise + 95% random picks from buffer history
3. Run Langevin 60 steps → new fakes
4. Append to buffer, cap at 8192 images

```python
import numpy as np
import random

class Buffer:
  def __init__(self, model):
    self.model = model
    self.examples = [
      tf.random.uniform(shape=(1, 32, 32, 1)) * 2 - 1
      for _ in range(128)
    ]

  def sample_new_exmps(self, steps, step_size, noise):
    n_new = np.random.binomial(128, 0.05)
    rand_imgs = tf.random.uniform((n_new, 32, 32, 1)) * 2 - 1
    old_imgs = tf.concat(
      random.choices(self.examples, k=128 - n_new), axis=0
    )
    inp_imgs = tf.concat([rand_imgs, old_imgs], axis=0)
    inp_imgs = generate_samples(
      self.model, inp_imgs, steps=steps, step_size=step_size, noise=noise
    )
    self.examples = tf.split(inp_imgs, 128, axis=0) + self.examples
    self.examples = self.examples[:8192]
    return inp_imgs
```

Buffer provides **persistent negative particles** — closer to true $p_\theta$ than i.i.d. noise each batch.

---

## One CD Step Visualized

Foster's Figure 7-5:

```
Real images  ──→  E(x) LOW   ──→  gradient pushes energy DOWN on reals
Fake images  ──→  E(x) HIGH  ──→  gradient pushes energy UP on fakes
```

No normalization step — energies can drift in absolute scale; only **relative** separation matters. Regularization $\alpha$ prevents runaway magnitudes.

---

## CD vs GAN Training

| | Contrastive Divergence | GAN |
|---|------------------------|-----|
| Real signal | Lower $E(x)$ | $D(x) \to 1$ |
| Fake signal | Raise $E(G)$ or Langevin samples | $D(G(z)) \to 0$ |
| Generator | Implicit via Langevin | Explicit network $G$ |
| Mode collapse | Less common | Common |
| Sampling speed | Slow (Langevin) | Fast |

EBMs trade sampling speed for a single unified energy landscape — no separate generator network.

---

## CD-k Variants

**CD-0:** Start negatives from data + noise (fast, biased).  
**CD-1:** One Gibbs/Langevin step from data (Hinton's original).  
**Foster's deep CD:** 60 Langevin steps + buffer — closer to persistent contrastive divergence in modern deep EBMs (Du & Mordatch).

More steps → better negative samples → sharper training signal → slower epochs.

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Reversed loss sign | Reals get high energy | `mean(fake) - mean(real)` |
| Empty buffer | Weak negatives | Initialize 128+ examples |
| $\alpha = 0$ | Energy explosion | Use $\alpha \approx 0.1$ |
| No input noise on reals | Overfitting | Add tiny Gaussian to reals |

---

## Connection to Other Sections

| Concept | Link |
|---------|------|
| Langevin | [Section 7.3](./section-03-langevin-dynamics-sampling.md) |
| Full train loop | [Section 7.5](./section-05-training-the-ebm.md) |
| Energy net | [Section 7.2](./section-02-mnist-energy-function.md) |
| Score-based / diffusion | [Section 7.7](./section-07-other-ebms-and-connections.md) |

---

## CD Gradient Intuition Step-by-Step

One minibatch:

1. **Real forward** — compute $E_\theta(x^+)$ for MNIST digits
2. **Buffer Langevin** — 60 steps from mixed init → $x^-$
3. **Fake forward** — compute $E_\theta(x^-)$
4. **Loss** — $\bar{E}(x^-) - \bar{E}(x^+) + \alpha(\mathbb{E}[E^2])$
5. **Backward** — update $\theta$ to widen real-vs-fake gap

If step 4 decreases, reals are getting **relatively** more probable under unnormalized $\exp(-E)$ even though $Z$ is unknown.

---

## Relation to Noise Contrastive Estimation (NCE)

Contrastive divergence is spiritually related to **NCE** — classify data vs noise samples. GAN discriminators are another contrastive family. EBMs use **continuous** Langevin negatives rather than fixed noise distribution — harder to tune but more expressive generator implicit in the energy landscape.

---

## Buffer Hyperparameters

| Setting | Foster value | Effect if changed |
|---------|--------------|-------------------|
| Buffer size | 8192 | Smaller → less diverse negatives |
| Refresh rate | 5% new noise | Higher → more exploration |
| Langevin steps | 60 | More → slower, sharper negatives |
| Batch | 128 | Must match train batch for concat |

Document your buffer settings in lab reports — CD is sensitive to negative quality.

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Contrastive divergence** | CD — training EBMs without $Z$ |
| **Positive phase** | Gradient terms on data points |
| **Negative phase** | Gradient terms on model samples |
| **Replay buffer** | Cache of past Langevin samples |
| **Persistent CD** | Reusing negative particles across steps |

---

## Reflection Questions

1. Why does CD avoid computing $\log Z_\theta$?
2. What role does the replay buffer play in negative sample quality?
3. How is `cdiv_loss` analogous to the GAN discriminator objective?
4. Why add regularization on squared energies?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 7 — Contrastive Divergence. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Hinton, G. (2002). Training Products of Experts by Minimizing Contrastive Divergence.
- Du, Y. & Mordatch, I. (2019). Implicit Generation and Modeling with Energy-Based Models.

---

**Previous:** [Section 7.3 — Langevin Dynamics Sampling](./section-03-langevin-dynamics-sampling.md)  
**Next:** [Section 7.5 — Training the EBM](./section-05-training-the-ebm.md)
