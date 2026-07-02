# Section 4.6: Gradient Penalty

> **Source inheritance:** Foster, Ch. 4 — "The Gradient Penalty Loss" & "Training the WGAN-GP"  
> **Enhanced with:** Interpolated samples, GradientTape on inputs, GP hyperparameters  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

The **gradient penalty (GP)** is how WGAN-GP enforces the 1-Lipschitz constraint without crippling the critic via weight clipping. Foster's implementation samples random **interpolates** between real and fake images, computes $\|\nabla_{\hat{x}} D(\hat{x})\|_2$, and penalizes deviation from 1. The critic learns to change slowly with respect to pixel inputs — exactly the condition Wasserstein GAN theory requires.

This section is the mathematical and code heart of [Section 4.5](./section-05-wasserstein-gan-wgan-gp.md).

---

## Why Gradient Penalty?

Weight clipping ($w \in [-0.01, 0.01]$) limits critic expressiveness — a "terrible" enforcement per the original WGAN authors. GP instead adds a soft loss term:

$$
\mathcal{L}_{\text{GP}} = \mathbb{E}_{\hat{x}}\left[\left(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1\right)^2\right]
$$
> **Readable form:** Penalize critic gradients that are too large or too small on interpolated points.

**Total critic loss:**

$$
\mathcal{L}_c = \underbrace{\mathbb{E}[D(G(z))] - \mathbb{E}[D(x)]}_{\text{Wasserstein}} + \lambda_{\text{GP}} \cdot \mathcal{L}_{\text{GP}}
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

Foster typically uses `gp_weight` $\lambda_{\text{GP}} = 10$.

---

## Interpolated Samples

GP is evaluated on points **between** real and fake (Figure 4-13):

$$
\hat{x} = \alpha \cdot x_{\text{real}} + (1 - \alpha) \cdot x_{\text{fake}}, \quad \alpha \sim \mathcal{U}(0, 1)
$$
> **Readable form:** Sample an interpolated point on the line between a real image and a generated image, then check the critic gradient there.

Pairwise per batch element — same index real/fake combined.

```python
def gradient_penalty(self, batch_size, real_images, fake_images):
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    # Foster uses uniform [0,1] in book; implementation may use normal — clip alpha
    alpha = tf.clip_by_value(alpha, 0.0, 1.0)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = self.critic(interpolated, training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean(tf.square(norm - 1.0))
    return gp
```

**Critical:** `tape.watch(interpolated)` — gradients w.r.t. **input pixels**, not weights. This is Langevin-style input-gradient logic ([Chapter 07](../chapter-07-energy-based-models/section-03-langevin-dynamics-sampling.md)) applied to GAN training.

---

## Step-by-Step GP Computation

| Step | Operation |
|------|-----------|
| 1 | Sample $\alpha$ per image |
| 2 | $\hat{x} = x_{\text{real}} + \alpha(x_{\text{fake}} - x_{\text{real}})$ |
| 3 | Forward $\hat{x}$ through critic |
| 4 | $\nabla_{\hat{x}} D(\hat{x})$ via GradientTape |
| 5 | L2 norm per sample over spatial + channel dims |
| 6 | Mean $(\|g\|_2 - 1)^2$ |

```python
# L2 norm over H, W, C for each batch item
norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
```

---

## Integration in `train_step`

Foster's WGAN-GP inner critic loop (3 iterations):

```python
c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(real_predictions)
c_gp = self.gradient_penalty(batch_size, real_images, fake_images)
c_loss = c_wass_loss + c_gp * self.gp_weight  # gp_weight often 10.0

c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
self.c_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_variables))
```

Track metrics separately:

```python
self.c_wass_loss_metric.update_state(c_wass_loss)
self.c_gp_metric.update_state(c_gp)
```

Figure 4-15: `c_gp` stays small and stable — Lipschitz constraint satisfied throughout training.

---

## No BatchNorm in Critic

Foster sidebar: **BatchNorm in critic breaks GP** because BN correlates samples within a batch, making per-sample gradient norms less meaningful. WGAN-GP critics use:

- Conv2D + LeakyReLU + Dropout (optional)
- **No** BatchNormalization in critic path

Generator may still use BatchNorm — asymmetry is intentional.

---

## Hyperparameter Tuning

| Parameter | Typical | Effect |
|-----------|---------|--------|
| `gp_weight` | 10 | Higher → stricter Lipschitz |
| `critic_steps` | 3–5 | More → better critic convergence |
| `learning_rate` | 1e-4 | Standard for WGAN-GP |
| Interpolation $\alpha$ dist | Uniform[0,1] | Coverage of line segments |

If `c_gp` explodes: lower `gp_weight` or reduce critic LR.

If samples are blurry: critic may be over-constrained — slightly lower `gp_weight`.

---

## CGAN Extension Preview

[Section 4.7](./section-07-conditional-gan.md) passes **label channels** into critic during GP — interpolated images use same label conditioning:

```python
def gradient_penalty(self, batch_size, real_images, fake_images, image_one_hot_labels):
    # ... same interpolation on images
    pred = self.critic([interpolated, image_one_hot_labels], training=True)
```

GP must evaluate critic with identical conditioning as real/fake forward passes.

---

## Comparison to Other Lipschitz Methods

| Method | Mechanism | Drawback |
|--------|-----------|----------|
| Weight clipping | Hard bounds on $w$ | Weak critic |
| Gradient penalty | Soft grad norm | Extra forward/backward per step |
| Spectral normalization | Normalize weight matrices | Different theory; also popular |

WGAN-GP remains Foster's teaching choice for stable face generation.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Forgot `tape.watch(interpolated)` | `grads` is None | Watch input tensor |
| BN in critic | Unstable `c_gp` | Remove critic BatchNorm |
| GP on real/fake only | Invalid constraint | Only on interpolates |
| Wrong norm axes | Scalar shape bug | Sum over [1,2,3] for NHWC |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Gradient penalty** | Loss pushing $\|\nabla_x D(x)\|_2 \to 1$ |
| **Interpolation** | $\hat{x}$ between real and fake samples |
| **gp_weight** | $\lambda$ scaling GP term in critic loss |
| **Lipschitz constraint** | Theoretical requirement for WGAN |
| **Input gradients** | Derivative of critic output w.r.t. pixels |

---

## Reflection Questions

1. Why interpolate between real and fake rather than evaluating GP on reals alone?
2. How does gradient penalty differ from weight clipping in mechanism?
3. Why does BatchNorm in the critic harm gradient penalty?
4. What does rising `c_gp` during training suggest about `gp_weight`?

---


## Foster Notebook Reference

Re-run the chapter notebook in [GDL_code](https://github.com/davidADSP/GDL_code) and compare your tensor shapes, loss curves, and saved sample grids to Foster's figures. Document one hyperparameter you changed and how outputs shifted — this habit transfers directly to Part III architectures (Transformers, Stable Diffusion, MuseGAN).

| Checkpoint | Action |
|------------|--------|
| After `model.summary()` | Verify spatial dims match hand calculation |
| Mid-training | Save sample grid or diagnostic plot |
| After training | Compare to Foster figure captions in the PDF |

---

## Extension Reading

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.) — full chapter walkthrough
- Goodfellow, Bengio & Courville (2016). *Deep Learning* — generative models part
- Original papers cited in Foster's chapter references


## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 4 — Gradient Penalty. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Gulrajani, I. et al. (2017). Improved Training of Wasserstein GANs. [https://arxiv.org/abs/1704.00028](https://arxiv.org/abs/1704.00028)
- Foster's codebase: `notebooks/04_gan/02_wgan_gp/wgan_gp.ipynb`

---

**Previous:** [Section 4.5 — Wasserstein GAN (WGAN-GP)](./section-05-wasserstein-gan-wgan-gp.md)  
**Next:** [Section 4.7 — Conditional GAN](./section-07-conditional-gan.md)


