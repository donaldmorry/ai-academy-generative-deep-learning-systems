# Section 7.3: Langevin Dynamics Sampling

> **Source inheritance:** Foster, Ch. 7 — "Sampling Using Langevin Dynamics"  
> **Enhanced with:** Input-space gradient descent, stochastic noise, GradientTape, and MNIST generation  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

The [energy network](./section-02-mnist-energy-function.md) $E_\theta(x)$ scores images but does not **produce** them. **Langevin dynamics** generates samples by treating pixels as variables: start from random noise, take small steps in the direction that **decreases** energy, plus Gaussian noise to escape local minima.

This is gradient descent on **inputs** $x$, not weights $\theta$ — the reverse of standard backprop training. Foster calls it **stochastic gradient Langevin dynamics** when noise is injected each step.

> **Readable form:** Roll a ball downhill on the energy landscape over pixel space — with jitter so it doesn't get stuck.

---

## Training vs Langevin: Two Uses of Gradients

| | Standard training | Langevin sampling |
|---|-------------------|-------------------|
| Variables updated | $\theta$ (weights) | $x$ (pixels) |
| Gradient of | Loss w.r.t. $\theta$ | $E_\theta(x)$ w.r.t. $x$ |
| Goal | Fit model to data | Find low-energy images |
| Network weights | Changing | **Fixed** |

When training a CNN classifier, you compute $\partial \mathcal{L}/\partial W$. For Langevin, freeze $E_\theta$ and compute $\partial E_\theta / \partial x$ via `tf.GradientTape(watch(x))`.

---

## Langevin Update Equation

$$
x_k = x_{k-1} - \eta \nabla_x E_\theta(x_{k-1}) + \omega, \quad \omega \sim \mathcal{N}(0, \sigma^2 I)
$$
> **Readable form:** New image equals old image minus step size times energy gradient, plus random noise.

- $\eta$ — **step size** (too large: overshoot minima; too small: slow)
- $\sigma$ — **noise scale** (zero noise: plain gradient descent, risk of local traps)
- $x_0 \sim \text{Uniform}[-1, 1]$ — random starting images

Foster minimizes **negative** energy in code (`out_score = -model(inp_imgs)`) so gradient ascent on $-E$ equals descent on energy.

---

## Implementation in TensorFlow

```python
import tensorflow as tf

def generate_samples(model, inp_imgs, steps, step_size, noise):
  imgs_per_step = []
  for _ in range(steps):
    # Stochastic term
    inp_imgs += tf.random.normal(inp_imgs.shape, mean=0.0, stddev=noise)
    inp_imgs = tf.clip_by_value(inp_imgs, -1.0, 1.0)

    with tf.GradientTape() as tape:
      tape.watch(inp_imgs)
      out_score = -model(inp_imgs)  # maximize score = minimize energy

    grads = tape.gradient(out_score, inp_imgs)
    grads = tf.clip_by_value(grads, -0.03, 0.03)  # stability
    inp_imgs += step_size * grads  # gradient ASCENT on -E
    inp_imgs = tf.clip_by_value(inp_imgs, -1.0, 1.0)
    imgs_per_step.append(inp_imgs.numpy().copy())
  return inp_imgs, imgs_per_step
```

**Clipping** pixels to $[-1,1]$ and gradients to small range prevents explosions during early training when $E_\theta$ is poorly calibrated.

---

## Hyperparameters (Foster's Defaults)

| Parameter | Typical value | Role |
|-----------|---------------|------|
| `steps` | 60 (training fakes) | More steps → better fake quality, slower |
| `step_size` | 10 | Pixel update magnitude |
| `noise` | 0.005 | Exploration vs exploitation |
| Init | Uniform $[-1,1]$ | Covers full canvas |

For **visualization** of denoising, save `imgs_per_step` every 10 steps — watch static morph into digit-like blobs as energy falls.

---

## Visualizing the Energy Landscape (2D Intuition)

Foster's Figure 7-4 shows a 2D surface: $x$-axis is pixel space (conceptually 2D), height is $E(x)$. Langevin path zigzags downhill with noise. MNIST lives in 1024-D pixel space — same math, impossible to plot fully.

```python
import matplotlib.pyplot as plt

x0 = tf.random.uniform((8, 32, 32, 1)) * 2 - 1
x_final, trajectory = generate_samples(energy_model, x0, steps=60, step_size=10, noise=0.005)

fig, axes = plt.subplots(1, 6, figsize=(12, 2))
for ax, t in zip(axes, [0, 10, 20, 30, 40, 59]):
  axes[ax].imshow(trajectory[t][0].squeeze(), cmap="gray", vmin=-1, vmax=1)
  axes[ax].axis("off")
plt.savefig("langevin_trajectory.png")
```

Before contrastive training, Langevin may not converge to digits — the landscape is untrained. After [Section 7.5](./section-05-training-the-ebm.md), samples resemble MNIST.

---

## Imperfect Samples Are OK for Training

Exact sampling from $p_\theta(x) \propto e^{-E(x)}$ would require infinite Langevin steps. **Contrastive divergence** ([Section 7.4](./section-04-contrastive-divergence.md)) uses short chains (60 steps) — "negative samples" need only be **more energetic than reals**, not perfect equilibrium samples.

This approximation bias is the price of tractable EBM training.

---

## Langevin vs Other Samplers

| Sampler | Used in |
|---------|---------|
| Langevin on $x$ | EBMs (this section) |
| Inverse flow | [RealNVP](../chapter-06-normalizing-flow-models/section-03-realnvp-architecture.md) |
| Decoder forward | [VAE](../chapter-03-variational-autoencoders/section-04-variational-autoencoders.md) |
| DDPM reverse steps | [Chapter 08](../chapter-08-diffusion-models/section-07-sampling-from-ddpm.md) |
| Autoregressive loop | [PixelCNN](../chapter-05-autoregressive-models/section-06-pixelcnn.md) |

All navigate toward high-density regions — Langevin is the most direct "follow $\nabla_x \log p$" approach when $p$ is unnormalized.

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Forgot `tape.watch(x)` | `grads is None` | Watch input tensor |
| Descent on $+E$ not $-E$ | Noise increases | Flip sign in code |
| step_size too large | Salt-and-pepper chaos | Reduce to 1–10 |
| No gradient clip | NaN images | `clip_by_value` on grads |

---

## Tuning Langevin Hyperparameters

| If samples are... | Try... |
|-------------------|--------|
| Noisy salt-and-pepper | Lower `step_size` (e.g., 5) |
| Stuck in gray blobs | Increase `steps` (100→200) |
| All identical | Increase `noise` slightly |
| Diverging to ±1 clip wall | Lower `step_size`, tighten grad clip |

Log mean energy of Langevin outputs per epoch — should decrease early in training as the landscape becomes meaningful.

---

## Relationship to Optimization

Langevin on $x$ with fixed $\theta$ mirrors SGD on $\theta$ with fixed $x$:

| | Weight training | Langevin sampling |
|---|-----------------|-------------------|
| Variable | $\theta$ | $x$ |
| Objective | Minimize batch loss | Minimize $E_\theta(x)$ |
| Noise | Mini-batch stochasticity | Explicit Gaussian on $x$ |
| Output | Trained classifier/generator | Synthetic image |

Recognizing this symmetry helps when reading papers that alternate between "learning parameters" and "sampling inputs."

---

## Code Checklist Before Training CD

- [ ] `inp_imgs` requires gradient via `tape.watch`
- [ ] Energy model in inference mode for weights but input grads enabled
- [ ] Pixel clip `[-1, 1]` after each update
- [ ] Gradient clip on $\partial E / \partial x$ (Foster: ±0.03)
- [ ] Return final images to `Buffer` for contrastive loss

---

## Numerical Stability Notes

Foster clips **gradients on pixels** to $[-0.03, 0.03]$ and **pixels** to $[-1, 1]$ every step. Without clips, early-training energy landscapes can shoot inputs to saturation — flat regions where swish gradients vanish. If samples look like salt noise, reduce `step_size` before removing clips.

---

## Batch Size Note

Langevin in Foster runs on 128 images in parallel — each gets independent gradient steps. Smaller batches (32) train faster per epoch but noisier CD; larger batches need more GPU RAM for `GradientTape` over full image tensors.

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Langevin dynamics** | MCMC using gradient of log-density + noise |
| **Stochastic gradient Langevin** | Noisy discretization of Langevin SDE |
| **Input-space gradient** | $\nabla_x E(x)$ via backprop through frozen net |
| **Step size $\eta$** | Learning rate for pixel updates |
| **Negative phase** | Generating low-energy fakes for contrastive training |

---

## Reflection Questions

1. Why must network weights stay fixed during Langevin sampling?
2. What happens if noise $\sigma = 0$ forever?
3. How is Langevin related to $\nabla_x \log p(x)$ when $p \propto e^{-E}$?
4. Why are 60 steps enough for training but not for perfect generation?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 7 — Langevin Dynamics. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Welling, M. & Teh, Y. W. (2011). Bayesian Learning via Stochastic Gradient Langevin Dynamics.
- Foster's notebook: `notebooks/07_ebm/01_ebm/ebm.ipynb`

---

**Previous:** [Section 7.2 — MNIST Energy Function](./section-02-mnist-energy-function.md)  
**Next:** [Section 7.4 — Contrastive Divergence](./section-04-contrastive-divergence.md)



