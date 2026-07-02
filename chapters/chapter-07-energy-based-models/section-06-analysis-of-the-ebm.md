# Section 7.6: Analysis of the EBM

> **Source inheritance:** Foster, Ch. 7 — chapter analysis sections  
> **Enhanced with:** Energy histograms, Langevin sample grids, training curves, and failure diagnosis  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

After [training the EBM](./section-05-training-the-ebm.md), analysis asks: did the energy landscape separate data from noise? Foster evaluates (1) **energy histograms** — reals vs fakes vs uniform noise, (2) **Langevin sample grids** — do long chains produce digit-like images?, (3) **metric curves** — `real`, `fake`, `cdiv` over epochs.

Unlike GANs (no scalar loss semantics) or flows (NLL), EBMs give interpretable energies — if training worked, a threshold on $E(x)$ classifies MNIST vs noise reasonably well.

> **Readable form:** Plot energies — reals should cluster low, garbage high, Langevin samples in between then improving.

---

## Energy Distribution Plots

```python
import numpy as np
import matplotlib.pyplot as plt

def energy_histogram(ebm, real_batch, n_noise=512):
  noise = tf.random.uniform((n_noise, 32, 32, 1)) * 2 - 1
  x0 = tf.random.uniform((n_noise, 32, 32, 1)) * 2 - 1
  langevin, _ = generate_samples(
    ebm.model, x0, steps=200, step_size=10, noise=0.005
  )

  e_real = ebm.model(real_batch).numpy().flatten()
  e_noise = ebm.model(noise).numpy().flatten()
  e_langevin = ebm.model(langevin).numpy().flatten()

  plt.hist(e_real, bins=40, alpha=0.6, label="real", density=True)
  plt.hist(e_noise, bins=40, alpha=0.6, label="noise", density=True)
  plt.hist(e_langevin, bins=40, alpha=0.6, label="Langevin", density=True)
  plt.legend()
  plt.xlabel("Energy $E(x)$")
  plt.title("Energy distributions")
```

**Success:** three separated peaks — real left (low), noise right (high), Langevin between or approaching real. **Failure:** overlapping histograms → more epochs or tune Langevin hyperparameters.

---

## Sample Quality Grid

```python
def show_langevin_grid(ebm, n=16, steps=200):
  x0 = tf.random.uniform((n, 32, 32, 1)) * 2 - 1
  samples, traj = generate_samples(
    ebm.model, x0, steps=steps, step_size=10, noise=0.005
  )
  fig, axes = plt.subplots(4, 4, figsize=(6, 6))
  for ax, img in zip(axes.flat, samples):
    ax.imshow(img.numpy().squeeze(), cmap="gray", vmin=-1, vmax=1)
    ax.axis("off")
  plt.savefig("ebm_samples.png")
```

Compare to [GAN MNIST samples](../chapter-04-generative-adversarial-networks/section-03-training-the-dcgan.md) and [VAE reconstructions](../chapter-03-variational-autoencoders/section-02-reconstructing-images.md). EBM samples may be **blurrier** or **diverse** — mode coverage is often better than vanilla GANs, sharpness worse than diffusion.

---

## Training Curve Analysis

```python
import pandas as pd

# From history.history after fit()
df = pd.DataFrame(history.history)
df[["real", "fake", "cdiv"]].plot()
plt.ylabel("Energy / CD loss")
plt.xlabel("Epoch")
```

| Pattern | Interpretation |
|---------|----------------|
| `fake - real` growing | Healthy contrast |
| `real` rising late | Catastrophic forgetting — lower LR |
| `fake` flat | Weak Langevin — more steps or step_size |
| `cdiv` negative | Bug — check loss sign |

---

## Anomaly Detection Use Case

EBMs excel at **scoring** inputs:

```python
def anomaly_score(ebm, x):
  return ebm.model(x).numpy()

scores = anomaly_score(ebm, x_test)
# High energy → likely not MNIST
threshold = np.percentile(scores, 95)
```

Unlike [flows](../chapter-06-normalizing-flow-models/section-06-analysis-of-realnvp.md) you don't get exact $\log p(x)$, but energy ranking often suffices for OOD detection. Calibrate threshold on validation set.

---

## Interpolation Experiment (Optional)

Langevin interpolation between two noise seeds is non-trivial (not linear latent space). For EBMs, **conditional** variants or joint energies $E(x,y)$ are needed for controlled interpolation — Foster notes limitations vs VAE latent arithmetic.

Try: same Langevin seed, vary step count — watch partial digits emerge.

---

## Known Limitations (Foster + Literature)

| Limitation | Detail |
|------------|--------|
| Slow sampling | 200+ Langevin steps per image |
| Slow training | Langevin inside every batch |
| Approximate CD | Short chains bias gradients |
| Stability | Sensitive to LR, $\alpha$, clip values |
| Sharpness | Below GAN/diffusion on MNIST |

Du & Mordatch (2019) showed deep EBMs **can** work on CIFAR with careful tuning — Foster's MNIST exercise is the pedagogical core.

---

## Comparison Table

| Model | MNIST sample quality | Exact density | Train speed |
|-------|---------------------|---------------|-------------|
| EBM | Moderate | No (unnormalized) | Slow |
| VAE | Blurry | ELBO | Fast |
| GAN | Sharp | No | Medium |
| RealNVP | N/A on digits in Foster | Yes | Medium |
| DDPM | Sharp (Chapter 08) | VLB | Slow |

---

## Common Failure Modes

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Gray blobs only | Undertrained | More epochs |
| Identical samples | Langevin stuck | Increase noise |
| All high energy | Inverted CD | Flip loss |
| Salt noise | step_size too big | Reduce to 5–10 |

---

## Extended Evaluation Protocol

For coursework or lab reports, document:

1. **Energy gap** — `mean(fake) - mean(real)` on test set at epoch 0, 25, 50
2. **Histogram figure** — three curves (real / noise / Langevin 200-step)
3. **Sample grid** — 4×4 Langevin from uniform init
4. **Failure note** — one hyperparameter you changed and what broke

Compare energy gap to GAN discriminator accuracy on the same MNIST split — both measure separability, but only EBM energies are calibrated for OOD ranking within training distribution.

---

## Long-Chain vs Short-Chain Langevin

| Chain length | Use case |
|--------------|----------|
| 60 steps | Training negatives (fast, biased) |
| 200 steps | Epoch-end visualization |
| 500+ steps | Best-looking standalone samples |

Do not expect 60-step training negatives to look like digits — they only need higher energy than reals.

---

## Side-by-Side with GAN Discriminator Scores

| Input type | EBM energy (trained) | GAN D(x) (trained) |
|------------|----------------------|---------------------|
| Real MNIST | Low | ≈ 1 |
| Uniform noise | High | ≈ 0 |
| Langevin 200-step | Medium → low | N/A without G |

EBM energies are uncalibrated absolutes; GAN outputs are bounded. Use **ranking** and **gap** for EBM evaluation, not threshold 0.5 logic.

---

## Exporting Metrics for Plots

```python
import json
with open("ebm_history.json", "w") as f:
  json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)
```

Persist `real`, `fake`, and `cdiv` each epoch — enables post-hoc analysis without retraining.

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Energy histogram** | Distribution of $E(x)$ by sample type |
| **OOD detection** | Flagging high-energy inputs |
| **Sample grid** | Visual qualitative evaluation |
| **CD gap** | $\mathbb{E}[E(fake)] - \mathbb{E}[E(real)]$ |
| **Long-chain sampling** | More Langevin steps at inference |

---

## Reflection Questions

1. What three histograms would you overlay to verify training?
2. How could energy scores detect corrupted MNIST digits?
3. Why might EBM samples be blurrier than GAN samples?
4. What metric from `history` best indicates ready-to-sample?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 7 — Analysis. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Du, Y. & Mordatch, I. (2019). Implicit Generation and Modeling with Energy-Based Models.

---

**Previous:** [Section 7.5 — Training the EBM](./section-05-training-the-ebm.md)  
**Next:** [Section 7.7 — Other EBMs & Connections](./section-07-other-ebms-and-connections.md)
