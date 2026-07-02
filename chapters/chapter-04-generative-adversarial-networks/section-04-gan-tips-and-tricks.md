# Section 4.4: GAN Tips and Tricks

> **Source inheritance:** Foster, Ch. 4 — "GAN Training: Tips and Tricks"  
> **Enhanced with:** Mode collapse, discriminator dominance, hyperparameter sensitivity, monitoring guidance  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

[GAN](../../GLOSSARY.md#generative-adversarial-network-gan) training is a dynamic equilibrium problem disguised as supervised learning. Foster devotes an entire section to failure modes because **architecture alone does not guarantee generation** — the discriminator can annihilate generator gradients, the generator can collapse to one image, and loss curves can lie. This section is a field guide: recognize pathology, apply targeted fixes, and know when to abandon vanilla DCGAN for WGAN-GP ([Section 4.5](./section-05-wasserstein-gan-wgan-gp.md)).

---

## Discriminator Overpowers Generator

When $D$ becomes too accurate, $\log(1 - D(G(z)))$ saturates and gradients to $G$ vanish — **training stalls** (Foster Figure 4-9).

| Symptom | Signal |
|---------|--------|
| `d_loss` → 0 quickly | D classifies perfectly |
| Generator output static noise | No G gradient |
| `g_loss` flat or NaN | Saturated sigmoid regions |

**Foster's remedies — weaken D:**

```python
# Increase dropout in discriminator
x = layers.Dropout(0.5)(x)  # try 0.4–0.5 vs 0.3

# Lower D learning rate
d_optimizer = optimizers.Adam(learning_rate=1e-5)  # vs 2e-4 for G

# Fewer D filters — reduce capacity
layers.Conv2D(32, ...)  # instead of 64 on first layer

# Label noise (already in Section 4.3)
real_noisy = 1.0 + 0.1 * tf.random.uniform(...)

# Random label flipping (advanced)
if tf.random.uniform(()) < 0.05:
    real_noisy = tf.zeros_like(real_predictions)  # occasional flip
```

> **Readable form:** If the critic is unbeatable, make the critic weaker or the task harder — not the generator louder.

---

## Generator Overpowers: Mode Collapse

**Mode collapse:** $G$ finds one $z$ region that fools $D$ and maps **all** noise there — output diversity vanishes (Figure 4-10).

Mechanism Foster describes:

1. Train $G$ several steps without updating $D$
2. $G$ discovers single winning mode
3. Gradients collapse; $G$ cannot escape
4. Retraining $D$ only shifts which mode wins

| Symptom | Visual |
|---------|--------|
| Identical samples in grid | Same brick/face repeated |
| Low `g_loss`, poor diversity | D fooled by one image |
| Latent traversal shows no change | $G$ ignores $z$ |

**Remedies — strengthen D:**

- Decrease dropout in D
- Increase D learning rate or filter count
- Reduce G learning rate
- **Increase batch size** — harder to fool D with duplicates
- Ensure alternating updates — never let G run alone too long

```python
# Mini-batch discrimination preview (conceptual)
# Penalize G if generated batch lacks feature variance
feat_real = discriminator.layers[-3](real_images)  # intermediate features
feat_fake = discriminator.layers[-3](generated_images)
mb_std = tf.math.reduce_std(feat_fake, axis=0)
diversity_penalty = tf.reduce_mean(tf.maximum(0.0, 1.0 - mb_std))
```

---

## Uninformative Loss

Foster's critical warning: **generator loss is not image quality**.

Figure 4-6: `g_loss` **increases** while samples improve because $D$ evolves. Comparing `g_loss` at epoch 50 vs epoch 200 is meaningless.

| Monitor | Use |
|---------|-----|
| Sample grids every N epochs | Primary quality signal |
| `d_loss` near 0.69 | Rough balance indicator |
| Nearest-neighbor L1 to training set | Memorization check |
| Feature statistics of batch | Mode collapse detector |

```python
def log_sample_grid(generator, latent_dim, epoch, n=16):
    z = tf.random.normal((n, latent_dim))
    imgs = generator(z, training=False)
    # save imgs — human inspection beats any scalar
```

---

## Hyperparameter Landscape

GANs are **exquisitely sensitive** Foster lists:

| Category | Examples |
|----------|----------|
| Architecture | Filter counts, kernel 4 vs 3, depth |
| Optimization | LR, $\beta_1$, batch size |
| Regularization | Dropout rate, BatchNorm momentum |
| Latent | `latent_dim` 64 vs 100 vs 128 |
| Activations | LeakyReLU slope |

No universal recipe — **educated trial and error** with systematic logging:

```python
# Experiment log template
experiments = [
    {"lr_d": 2e-4, "lr_g": 2e-4, "dropout": 0.3, "note": "baseline"},
    {"lr_d": 1e-4, "lr_g": 2e-4, "dropout": 0.3, "note": "weaker D"},
    {"lr_d": 2e-4, "lr_g": 1e-4, "dropout": 0.5, "note": "heavy D dropout"},
]
```

---

## Practical Training Checklist

Before declaring failure:

1. Verify data scaling $[-1,1]$ matches `tanh` output
2. Confirm separate D/G optimizers and gradient application
3. Plot samples at epochs 1, 5, 10, 25, 50
4. Check `discriminator` accuracy on real vs fake batch
5. Try label smoothing if D loss → 0 in < 10 epochs
6. Try larger batch if mode collapse appears

---

## Architectural Stabilizers (Preview)

| Technique | Chapter reference |
|-----------|------------------|
| Wasserstein loss + GP | [Sections 4.5–4.6](./section-05-wasserstein-gan-wgan-gp.md) |
| Spectral normalization | Chapter 10 (advanced GANs) |
| Progressive growing | ProGAN — Chapter 10 |
| Diffusion models | Chapter 08 — different paradigm |

WGAN-GP addresses root cause: non-saturating loss with Lipschitz-bounded critic.

---

## DCGAN-Specific Tips Recap

From Radford et al. + Foster training experience:

| Tip | Rationale |
|-----|-----------|
| `beta_1=0.5` for Adam | Reduces oscillation |
| Avoid BN on D's first layer | Preserve low-level signal |
| Use strided conv, not pooling | Learned downsampling |
| Kernel size 4, stride 2 | Standard spatial halving |
| One-sided label smoothing | Prevent D overconfidence |

---

## When to Stop Training

Foster notes D often **wins eventually** (Figure 4-6) — but G may already produce acceptable images. Stop based on:

- Sample quality plateau (visual)
- Mode collapse onset (declining diversity)
- Compute budget

Save generator checkpoints frequently — best samples may occur before final epoch.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Chasing low `g_loss` | Premature stop or wrong tweaks | Trust sample grids |
| Training G without D updates | Mode collapse | Strict alternation |
| Tiny batch (e.g., 16) | Unstable moments / collapse | Batch ≥ 64 if possible |
| Ignoring data pipeline bugs | Gray or static outputs | Verify preprocess |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Mode collapse** | Generator outputs limited variety — one or few modes |
| **Vanishing gradients** | Saturated D prevents G learning |
| **Label smoothing** | Noisy 0/1 targets for D |
| **Uninformative loss** | G loss not comparable across training |
| **Discriminator dominance** | D too strong; G starved of gradient |

---

## Reflection Questions

1. Why does mode collapse produce near-zero generator gradients?
2. List three ways to weaken an overpowering discriminator.
3. Why is comparing generator loss at epoch 10 vs 100 misleading?
4. When would you escalate from these tips to WGAN-GP?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 4 — GAN Tips and Tricks. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Salimans, T. et al. (2016). Improved Techniques for Training GANs.
- Arjovsky, M. et al. (2017). Wasserstein GAN.

---

**Previous:** [Section 4.3 — Training the DCGAN](./section-03-training-the-dcgan.md)  
**Next:** [Section 4.5 — Wasserstein GAN (WGAN-GP)](./section-05-wasserstein-gan-wgan-gp.md)
