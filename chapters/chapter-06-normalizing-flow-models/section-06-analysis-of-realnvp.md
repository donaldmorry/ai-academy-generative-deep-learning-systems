# Section 6.6: Analysis of RealNVP

> **Source inheritance:** Foster, Ch. 6 — "Analysis of the RealNVP Model"  
> **Enhanced with:** Latent visualization, sample quality, density heatmaps, and failure diagnosis  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Training [RealNVP](./section-05-training-realnvp.md) is only half the story — you must **verify** the flow learned a valid bijection. Foster's analysis checks three views: (1) data mapped to latent space looks Gaussian, (2) samples from $\mathcal{N}(0,I)$ land on both moon crescents, (3) log-likelihood improves on held-out points.

In 2D you can draw density heatmaps and watch the crescents **unbend** into a blob. This diagnostic discipline transfers directly to image flows ([GLOW](./section-07-glow.md)) where you rely on FID and visual grids instead.

> **Readable form:** Good flow = round latent cloud + moon-shaped samples + rising log p(x).

---

## Latent Space Visualization

Forward-map all training points:

```python
z_train, log_det = flow_model(train_data, inverse=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(train_data[:, 0], train_data[:, 1], s=4, alpha=0.4)
axes[0].set_title("Data space (moons)")
axes[1].scatter(z_train[:, 0], z_train[:, 1], s=4, alpha=0.4)
axes[1].set_title("Latent space (should be Gaussian)")
axes[1].set_aspect("equal")
```

**Success:** right panel shows a fuzzy disk centered at origin — no crescent structure left. **Failure:** latent points still curved → insufficient layers, early stopping, or LR too high.

Overlay theoretical Gaussian contours:

```python
from matplotlib import patches
circle = patches.Circle((0, 0), radius=2, fill=False, linestyle="--")
axes[1].add_patch(circle)
```

---

## Generated Sample Quality

```python
n = 2000
z = tf.random.normal((n, 2))
x_gen, _ = flow_model(z, inverse=True)

plt.figure(figsize=(5, 5))
plt.scatter(train_data[:, 0], train_data[:, 1], s=2, alpha=0.2, label="train")
plt.scatter(x_gen[:, 0], x_gen[:, 1], s=2, alpha=0.4, label="generated")
plt.legend()
plt.title("Train vs generated moons")
```

Checklist:

| Criterion | Pass |
|-----------|------|
| Both crescents covered | Yes |
| No mass in empty gap between moons | Mostly |
| Similar noise spread as training | Roughly |
| No collapsed cluster at one point | No |

Mode collapse here means all samples on one crescent — rare with flows (unlike GANs) but possible with broken masks.

---

## Density Heatmaps (2D)

Evaluate $\log p_X(x)$ on a grid:

```python
xs = np.linspace(-2, 2, 100)
ys = np.linspace(-2, 2, 100)
xx, yy = np.meshgrid(xs, ys)
grid = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype("float32")

log_px = compute_log_likelihood(flow_model, grid).numpy()
log_px = log_px.reshape(100, 100)

plt.contourf(xx, yy, np.exp(log_px), levels=50)
plt.scatter(train_data[:, 0], train_data[:, 1], s=1, c="white", alpha=0.3)
plt.title("Learned density $p_X(x)$")
```

High-density regions should trace the moon shapes. Low density in the gap between crescents confirms the model learned **bimodality**, not a single blob.

---

## Log-Likelihood Curves

Track NLL across epochs (from `history` or custom callback):

```python
# After training
val_nll = -np.mean(compute_log_likelihood(flow_model, val_data).numpy())
print(f"Validation NLL: {val_nll:.3f}")
```

Compare to a Gaussian baseline fit on moons (single-mode miss) — RealNVP should win decisively. If train NLL << val NLL, consider overfitting (unlikely on 3k 2D points with regularized coupling nets).

---

## Jacobian and Volume Interpretation

Remember from [Section 6.2](./section-02-change-of-variables.md): $\log |\det J|$ measures how much the flow **expands or compresses** local volume. On moon tips (high curvature), determinants spike — the map must stretch thin regions to fill Gaussian area.

```python
_, log_det = flow_model(train_data, inverse=False)
plt.hist(log_det.numpy(), bins=50)
plt.title("Log-det distribution on training data")
```

Extreme outliers in log-det often correlate with points at crescent tips — expected, not necessarily a bug.

---

## RealNVP Strengths and Limits (Foster's Take)

**Strengths:**

- Exact likelihood — use for anomaly detection, compression bounds
- Fast one-pass sampling
- Stable MLE training (no discriminator balancing)

**Limits on moons / images:**

- Architectural handcuffs (coupling, masking)
- Depth needed for sharp multimodal structure
- Image SOTA often held by diffusion/GANs despite flows' tractability

[Section 6.7](./section-07-glow.md) pushes RealNVP ideas to $32 \times 32$ faces; [Section 6.8](./section-08-ffjord-and-normalizing-flows-summary.md) surveys continuous-time extensions.

---

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Latent still crescent-shaped | Too few layers | Increase `n_layers` |
| Samples only on one moon | Mask never alternates | Check mask schedule |
| Density smeared between moons | Undertrained | More epochs |
| NaN in heatmap | $\exp(s)$ overflow | tanh on scales, clip grads |

---

## Connection to Other Chapters

| Concept | Link |
|---------|------|
| GAN mode collapse | [Chapter 04](../chapter-04-generative-adversarial-networks/section-04-gan-tips-and-tricks.md) |
| VAE latent plots | [Section 3.3](../chapter-03-variational-autoencoders/section-03-visualizing-latent-space.md) |
| Flow taxonomy | [Section 1.7](../chapter-01-generative-modeling/section-07-generative-model-taxonomy.md) |

---

## Round-Trip Bijection Test

Verify invertibility numerically on moons:

```python
z, log_det_f = flow_model(data, inverse=False)
x_recon, log_det_inv = flow_model(z, inverse=True)
recon_err = tf.reduce_mean((data - x_recon) ** 2)
print("MSE round-trip:", recon_err.numpy())
print("Log-det sum:", (log_det_f + log_det_inv).numpy()[:5])
```

Round-trip MSE should be ~1e-6 float32; log-det forward + inverse should approximately cancel in sum over layers (up to numerical tolerance).

---

## Interpolation in Latent Space

```python
z_a, _ = flow_model(data[0:1], inverse=False)
z_b, _ = flow_model(data[1:2], inverse=False)
for alpha in np.linspace(0, 1, 10):
  z_mix = (1 - alpha) * z_a + alpha * z_b
  x_mix, _ = flow_model(z_mix, inverse=True)
  # plot x_mix — smooth morph between moon points
```

Flows give **exact** latent arithmetic unlike VAE stochastic encoders — pedagogically satisfying on 2D plots.

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Latent visualization** | Plotting $z = f(x)$ to verify Gaussianization |
| **Density heatmap** | 2D contour of learned $p_X(x)$ |
| **Validation NLL** | Held-out negative log-likelihood |
| **Volume preservation** | Jacobian det = 1; RealNVP is *non*-volume-preserving |
| **Bijection check** | Round-trip $x \to z \to x'$ with small error |

---

## Reflection Questions

1. What three plots would you show to convince a colleague the flow works?
2. Why might log-det histograms be heavy-tailed even for a correct model?
3. How does flow "mode coverage" compare to GANs on two moons?
4. What metric would you add for 32×32 GLOW that replaces 2D heatmaps?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 6 — Analysis. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Foster's notebook: `notebooks/06_normflow/01_realnvp/realnvp.ipynb`

---

**Previous:** [Section 6.5 — Training RealNVP](./section-05-training-realnvp.md)  
**Next:** [Section 6.7 — GLOW](./section-07-glow.md)
