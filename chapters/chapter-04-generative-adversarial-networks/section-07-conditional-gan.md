# Section 4.7: Conditional GAN

> **Source inheritance:** Foster, Ch. 4 — "Conditional GAN (CGAN)"  
> **Enhanced with:** Blond hair conditioning, label concatenation, WGAN-GP CGAN training  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

A standard [GAN](../../GLOSSARY.md#generative-adversarial-network-gan) samples $z$ and hopes for the right hair color. A **conditional GAN (CGAN)** feeds the label **into both** generator and critic, so the network must produce images **consistent** with the requested attribute. Foster conditions on CelebA **Blond_Hair**: same latent vector + different one-hot label → same face identity, different hair color (Figure 4-17).

Mirza & Osindero (2014) introduced the idea; Foster implements it atop WGAN-GP from [Sections 4.5–4.6](./section-05-wasserstein-gan-wgan-gp.md).

---

## Architecture: Where Labels Enter

| Network | Label injection | Mechanism |
|---------|-----------------|-----------|
| **Generator** | Concatenate to $z$ | `[z, one_hot]` → Reshape → conv stack |
| **Critic** | Extra image channels | Repeat one-hot to $(64,64,2)$ → concat with RGB |

```
Generator:  z (32-d) + label (2-d) → concat (34-d) → Reshape (1,1,34) → Conv2DTranspose...
Critic:     image (64,64,3) + label_map (64,64,2) → concat (64,64,5) → Conv2D...
```

> **Readable form:** Generator gets label as vector; critic gets label as spatial planes.

```python
from tensorflow.keras import layers, models

# Critic
critic_input = layers.Input(shape=(64, 64, 3))
label_input = layers.Input(shape=(64, 64, 2))
x = layers.Concatenate(axis=-1)([critic_input, label_input])
# ... conv blocks, linear output

# Generator
generator_input = layers.Input(shape=(32,))  # latent (Foster uses 32 for CGAN)
label_vec = layers.Input(shape=(2,))
x = layers.Concatenate(axis=-1)([generator_input, label_vec])
x = layers.Reshape((1, 1, 34))(x)
# ... transpose conv blocks, tanh RGB output
```

---

## One-Hot Labels for Blond Hair

CelebA binary attribute → 2-class one-hot:

| Vector | Meaning |
|--------|---------|
| `[1, 0]` | Not blond |
| `[0, 1]` | Blond |

For Fashion-MNIST with 10 classes, label dim = 10 ([Chapter 03](../chapter-03-variational-autoencoders/README.md) classes).

---

## Expanding Labels for the Critic

```python
def expand_labels(one_hot_labels, height=64, width=64):
    # (batch, 2) → (batch, 1, 1, 2) → (batch, 64, 64, 2)
    x = one_hot_labels[:, None, None, :]
    x = tf.repeat(x, repeats=height, axis=1)
    x = tf.repeat(x, repeats=width, axis=2)
    return x
```

Critic sees **constant** label planes across spatial dimensions — spatially informed conditioning without a separate embedding network.

---

## Training with Labels

`train_step` unpacks `(real_images, one_hot_labels)` from dataset:

```python
def train_step(self, data):
    real_images, one_hot_labels = data
    image_one_hot_labels = expand_labels(one_hot_labels)

    for _ in range(self.critic_steps):
        z = tf.random.normal((batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            fake_images = self.generator([z, one_hot_labels], training=True)
            fake_pred = self.critic([fake_images, image_one_hot_labels], training=True)
            real_pred = self.critic([real_images, image_one_hot_labels], training=True)
            c_wass = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
            c_gp = self.gradient_penalty(batch_size, real_images, fake_images, image_one_hot_labels)
            c_loss = c_wass + self.gp_weight * c_gp
        # update critic ...

    z = tf.random.normal((batch_size, self.latent_dim))
    with tf.GradientTape() as tape:
        fake_pred = self.critic(
            [self.generator([z, one_hot_labels], training=True), image_one_hot_labels],
            training=True,
        )
        g_loss = -tf.reduce_mean(fake_pred)
    # update generator ...
```

**Consistency constraint:** If generator produces a blond face but passes `[1,0]` label, critic detects mismatch → strong negative signal.

---

## Controlled Generation at Inference

```python
z = tf.random.normal((8, 32))  # fixed latents

blond = tf.constant([[0., 1.]] * 8)
not_blond = tf.constant([[1., 0.]] * 8)

faces_blond = generator([z, blond], training=False)
faces_dark = generator([z, not_blond], training=False)
```

Figure 4-17: **same $z$**, different label → only hair color changes. Demonstrates disentangled latent organization learned adversarially.

---

## Dataset Pipeline with Labels

```python
# Load CelebA with attributes CSV — map images to (img, one_hot_blond)
def make_cgan_dataset(image_paths, attr_df, batch_size=64):
    # ... yield (image_tensor, label_tensor) pairs
    pass

cgan.fit(labeled_dataset, epochs=25)
```

Foster: labels improve quality even when conditioning is not required — "highly informative extension to pixel input."

---

## CGAN + WGAN-GP Benefits

| Feature | Benefit |
|---------|---------|
| Wasserstein loss | Stable training |
| Gradient penalty | Lipschitz critic |
| Conditioning | Controllable attributes |
| CelebA RGB 64×64 | Sharp face synthesis |

---

## Extensions Beyond Binary Labels

| Setup | Label format |
|-------|--------------|
| MNIST digit class | 10-d one-hot |
| Multi-attribute | Concatenate attribute one-hots |
| Class embedding | Learned `Embedding` layer instead of one-hot |

Projection discriminator and auxiliary classifier GAN (AC-GAN) are advanced variants in Chapter 10.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Mismatched label on G vs D | Ignored conditioning | Same label to both |
| Wrong Reshape after concat | Shape error | $1+32+2 = 35$? verify Foster's 34-d |
| GP without label channels | Invalid critic | Pass labels to GP forward |
| Shuffled labels in batch | Random hair colors | Align labels with images |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **CGAN** | GAN conditioned on class or attribute label |
| **One-hot encoding** | Binary vector selecting attribute state |
| **Label channels** | Spatially repeated one-hot for critic input |
| **Controlled generation** | Specifying output attributes at inference |
| **Consistency** | Generated image must match provided label |

---

## Reflection Questions

1. Why does the critic receive labels as extra channels rather than only the generator?
2. How does Foster demonstrate disentanglement in Figure 4-17?
3. Why pass labels into the gradient penalty forward pass?
4. How would you extend CGAN to generate a specific Fashion-MNIST class?

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

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 4 — Conditional GAN. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Mirza, M., & Osindero, S. (2014). Conditional Generative Adversarial Nets. [https://arxiv.org/abs/1411.1784](https://arxiv.org/abs/1411.1784)
- Foster's codebase: `notebooks/04_gan/03_cgan/cgan.ipynb`

---

**Previous:** [Section 4.6 — Gradient Penalty](./section-06-gradient-penalty.md)  
**Next:** [Section 4.8 — Analysis & Comparison](./section-08-analysis-and-comparison.md)
