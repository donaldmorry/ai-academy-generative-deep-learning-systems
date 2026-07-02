# Section 4.5: Wasserstein GAN (WGAN-GP)

> **Source inheritance:** Foster, Ch. 4 — "Wasserstein GAN with Gradient Penalty (WGAN-GP)"  
> **Enhanced with:** Wasserstein loss, critic vs discriminator, Lipschitz constraint, CelebA faces  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Vanilla [GANs](../../GLOSSARY.md#generative-adversarial-network-gan) use binary cross-entropy — a loss that **saturates** when the discriminator is confident, starving the generator. Arjovsky et al.'s **Wasserstein GAN** replaces BCE with the **Earth Mover's distance** approximated by a **critic** that outputs unbounded scores, not probabilities. Gulrajani et al. added the **gradient penalty (GP)** to enforce the 1-Lipschitz constraint without weight clipping.

Foster trains WGAN-GP on CelebA — sharper faces than the VAE in [Chapter 03](../chapter-03-variational-autoencoders/section-07-celeba-faces.md), with loss curves that actually mean something (Figure 4-15).

---

## From BCE to Wasserstein Loss

**Standard GAN discriminator** minimizes:

$$
\min_D \; -\mathbb{E}_{x}[\log D(x)] - \mathbb{E}_{z}[\log(1 - D(G(z)))]
$$
> **Readable form:** The original discriminator objective rewards high scores for real images and low scores for generated images.

**WGAN critic** minimizes:

$$
\min_D \; -\mathbb{E}_{x}[D(x)] + \mathbb{E}_{z}[D(G(z))]
$$
> **Readable form:** The Wasserstein critic minimizes fake scores minus real scores, encouraging real samples to rank higher than generated samples.

Equivalently: **maximize** $\mathbb{E}[D(x)] - \mathbb{E}[D(G(z))]$ — push real scores up, fake scores down.

| Property | BCE GAN | WGAN |
|----------|---------|------|
| Output activation | Sigmoid → $[0,1]$ | Linear → $(-\infty, \infty)$ |
| Labels | 0 and 1 | $-1$ and $+1$ |
| Name | Discriminator | **Critic** |
| Loss correlation with quality | Poor | **Meaningful** (per original paper) |

```python
# WGAN critic — NO sigmoid on final layer
critic_output = layers.Conv2D(1, 4, padding="valid", activation=None)(x)
```

---

## Generator Loss in WGAN

$$
\min_G \; -\mathbb{E}_{z}[D(G(z))]
$$
> **Readable form:** The generator improves by making the critic assign higher scores to generated samples.

Generator maximizes critic score on fakes — same direction as fooling a discriminator, different functional form:

```python
g_loss = -tf.reduce_mean(fake_predictions)
```

---

## The 1-Lipschitz Constraint

WGAN theory requires critic $D$ to be **1-Lipschitz**:

$$
|D(x_1) - D(x_2)| \leq \|x_1 - x_2\|
$$
> **Readable form:** Critic cannot change predictions faster than input pixels change.

Without this, Wasserstein distance estimation is invalid and training diverges.

**Original WGAN:** weight clipping to $[-0.01, 0.01]$ after each step — Foster quotes authors calling it "clearly terrible" (limits critic capacity).

**WGAN-GP:** soft penalty pushing gradient norm toward 1 — [Section 4.6](./section-06-gradient-penalty.md).

---

## Architectural Changes from DCGAN

Foster's WGAN-GP recap:

| Change | Reason |
|--------|--------|
| Remove final sigmoid | Unbounded critic output |
| Labels $\pm 1$ | Wasserstein formulation |
| No BatchNorm in critic | Correlates batch samples; breaks GP |
| Multiple critic steps per G step | Train critic toward convergence |
| Gradient penalty in critic loss | Enforce Lipschitz |

**Critic-to-generator update ratio:** typically **3–5** critic steps per generator step (Foster uses 3).

```python
for _ in range(3):  # critic inner loop
    # ... compute c_loss, update critic
# then one generator update
```

Paradoxically opposite to vanilla GAN where strong D is bad — WGAN **needs** a strong critic for meaningful gradients.

---

## Critic Architecture (CelebA)

Similar to DCGAN discriminator but:

- RGB input $(64, 64, 3)$
- Linear output scalar
- No BatchNorm (use LayerNorm or nothing — Foster omits BN in critic)

Generator mirrors DCGAN RGB transpose conv stack with `tanh` output and $[-1,1]$ preprocessing.

Notebook: `notebooks/04_gan/02_wgan_gp/wgan_gp.ipynb` (adapted from Aakash Kumar Nain's Keras tutorial).

---

## Training Loop Sketch

```python
class WGANGP(models.Model):
  def train_step(self, real_images):
      batch_size = tf.shape(real_images)[0]
      for _ in range(self.critic_steps):
          z = tf.random.normal((batch_size, self.latent_dim))
          with tf.GradientTape() as tape:
              fake_images = self.generator(z, training=True)
              fake_pred = self.critic(fake_images, training=True)
              real_pred = self.critic(real_images, training=True)
              c_wass = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
              c_gp = self.gradient_penalty(batch_size, real_images, fake_images)
              c_loss = c_wass + self.gp_weight * c_gp
          grads = tape.gradient(c_loss, self.critic.trainable_variables)
          self.c_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

      z = tf.random.normal((batch_size, self.latent_dim))
      with tf.GradientTape() as tape:
          fake_pred = self.critic(self.generator(z, training=True), training=True)
          g_loss = -tf.reduce_mean(fake_pred)
      g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
      self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
```

---

## Loss Curve Interpretation

Foster Figure 4-15: `c_loss`, `c_wass`, `c_gp`, and `g_loss` all **stabilize smoothly** over 25 epochs — unlike DCGAN's adversarial oscillation.

| Metric | Healthy trend |
|--------|---------------|
| `c_wass` | Stabilizes — critic separates distributions |
| `c_gp` | Small, steady — Lipschitz satisfied |
| `g_loss` | Correlates with improving samples |

---

## WGAN-GP vs VAE on Faces

Foster's comparison (Figure 4-14 vs VAE Figure 3-18):

| | VAE | WGAN-GP |
|---|-----|---------|
| Hair/background boundary | Soft | **Sharp** |
| Training stability | High | Moderate (better than DCGAN) |
| Training time | Shorter to decent results | Longer |
| Latent arithmetic | Natural | Less smooth |

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Sigmoid on critic | Invalid WGAN | Linear output only |
| BN in critic | Unstable GP | Remove BatchNorm |
| 1:1 critic/G updates | Weak critic | 3–5 critic steps |
| Weight clipping + GP | Conflicting constraints | Use GP **or** clipping, not both |
| Ignoring GP weight | Divergence | Tune `gp_weight` (often 10) |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Wasserstein distance** | Earth Mover's distance between distributions |
| **Critic** | WGAN discriminator with linear output |
| **1-Lipschitz** | Bounded gradient of critic w.r.t. input |
| **WGAN-GP** | WGAN + gradient penalty for Lipschitz |
| **Critic steps** | Inner loop training critic before G update |

---

## Reflection Questions

1. Why does WGAN use labels $+1/-1$ instead of $1/0$?
2. Why is a strong critic desirable in WGAN but dangerous in vanilla GAN?
3. What problem does weight clipping cause, motivating gradient penalty?
4. How do WGAN-GP loss curves differ from DCGAN curves in Foster Figure 4-6 vs 4-15?

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

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 4 — WGAN-GP. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Arjovsky, M. et al. (2017). Wasserstein GAN. [https://arxiv.org/abs/1701.07875](https://arxiv.org/abs/1701.07875)
- Gulrajani, I. et al. (2017). Improved Training of Wasserstein GANs. [https://arxiv.org/abs/1704.00028](https://arxiv.org/abs/1704.00028)
- Foster's codebase: `notebooks/04_gan/02_wgan_gp/wgan_gp.ipynb`

---

**Previous:** [Section 4.4 — GAN Tips and Tricks](./section-04-gan-tips-and-tricks.md)  
**Next:** [Section 4.6 — Gradient Penalty](./section-06-gradient-penalty.md)
