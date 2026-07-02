# Section 2.7: Batch Normalization & Dropout

> **Source inheritance:** Foster, Ch. 2 — "Batch Normalization" and "Dropout"  
> **Enhanced with:** Covariate-shift intuition, training vs inference behavior, and the BAD layer ordering used in Foster's CIFAR-10 CNN  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Training Stability Problem

[Section 2.5](./section-05-mlp-training-deep-dive.md) showed how loss spikes and NaN values signal training collapse. One root cause is **covariate shift** — the distribution of activations feeding each layer drifts as weights update. Early layers may output well-scaled values at step 100 but wildly scaled values at step 5,000.

Foster offers two complementary fixes that appear in virtually every modern architecture — discriminative and [generative](../../GLOSSARY.md#generative-model) alike:

1. **Batch normalization** — stabilizes activation distributions during training
2. **Dropout** — prevents [overfitting](../../GLOSSARY.md#overfitting) by randomly disabling units

Both layers have **no learned convolution weights** of their own (batch norm has small scale/shift parameters; dropout has none), yet they dramatically change how networks train. You will see `BatchNormalization` in DCGAN discriminators, VAE encoders, and diffusion U-Nets. Dropout appears less often in modern generative stacks but remains essential to understand.

> **Readable form:** Batch norm keeps activations stable layer to layer. Dropout stops the network from memorizing training examples.

---

## Covariate Shift: The Pile of Books

Scaling input pixels to $[0, 1]$ ([Section 2.4](./section-04-multilayer-perceptron.md)) gives training a stable start. But as weights move away from random initialization, each layer's output distribution can shift — breaking the implicit assumption the next layer's weights were tuned for.

Foster's analogy: you carry a tall pile of books. Each gust of wind shifts the stack. You compensate, but the pile grows less stable until it collapses.

In a neural network, **each layer is a book**. When layer $l$ updates its weights, the distribution of activations seen by layer $l+1$ changes — even if layer $l+1$'s weights haven't moved yet. Over thousands of gradient steps, this **internal covariate shift** can produce runaway weight values and the **exploding gradient** problem.

| Symptom | Likely cause |
|---------|--------------|
| Loss suddenly becomes `NaN` | Weights grew large enough to overflow |
| Training fine for hours, then collapses | Slow drift in activation statistics |
| Very slow convergence | Layers constantly adapting to shifting inputs |

> **Readable form:** Each layer expects its inputs to look similar across training steps. When that breaks, training can explode.

---

## Batch Normalization: The Math

Batch normalization (Ioffe & Szegedy, 2015) normalizes each channel's activations **across the current mini-batch** during training.

For a single channel with batch values $\{x_1, \ldots, x_m\}$:

$$
\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i, \quad
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2
$$
> **Readable form:** The total combines the indexed terms, so each relevant example, state, feature, or dimension contributes once.

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

$$
y_i = \gamma \hat{x}_i + \beta
$$
> **Readable form:** Subtract batch mean, divide by batch standard deviation, then scale by learned gamma and shift by learned beta.

$\gamma$ (scale) and $\beta$ (shift) are **trainable parameters** per channel — the network can undo normalization if that helps the loss. $\epsilon$ is a tiny constant for numerical stability.

For conv layers, normalization runs **per channel** across all spatial positions in the batch: for a tensor of shape `(batch, height, width, channels)`, batch norm computes separate statistics for each of the `channels` dimensions.

```python
from tensorflow.keras import layers, models
import tensorflow as tf

x_in = layers.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, padding="same")(x_in)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.ReLU()(x)
model = models.Model(x_in, x)
model.summary()
# BatchNormalization: 4 params per channel (gamma, beta + 2 moving stats tracked)
```

---

## Training vs Inference: Two Modes

Batch norm behaves differently at **training** and **prediction** time. This is the most commonly misunderstood detail in all of [Keras](../../GLOSSARY.md#keras).

### During training

- Compute $\mu_B$ and $\sigma_B$ from the **current batch**
- Normalize using those batch statistics
- Update **moving averages** of mean and variance:

$$
\mu_{\text{running}} \leftarrow (1 - \text{momentum}) \cdot \mu_{\text{running}} + \text{momentum} \cdot \mu_B
$$
> **Readable form:** Update the running mean by blending the previous running estimate with the current batch mean.

### During inference (prediction)

- No batch to compute statistics from (you may predict one image at a time)
- Use the **stored moving averages** from training

In Keras, `model.fit()` sets `training=True` automatically. For custom prediction or generative sampling:

```python
# Correct: use inference behavior for batch norm and dropout
predictions = model(x_test, training=False)

# Generative sampling — always training=False
latent = tf.random.normal((16, 100))
generated = generator(latent, training=False)
```

> **Readable form:** Training uses this batch's mean and std. Inference uses running averages saved during training.

**Forgetting `training=False`** during GAN sample generation is a classic bug: batch norm uses incorrect statistics and images look noisy or washed out.

### Parameter count

Per channel in the preceding layer:

| Parameter | Trainable? | Role |
|-----------|------------|------|
| $\gamma$ (scale) | Yes | Learned scale after normalization |
| $\beta$ (shift) | Yes | Learned shift after normalization |
| Running mean | No | Accumulated during training |
| Running variance | No | Accumulated during training |

Four values per channel — two learned, two tracked.

---

## Where to Place Batch Normalization

Foster's CIFAR-10 CNN places batch norm **before** the activation (ReLU/LeakyReLU). The mnemonic is **BAD**:

| Order | Layer |
|-------|-------|
| **B** | Batch normalization |
| **A** | Activation |
| **D** | Dropout (when used) |

```python
x = layers.Conv2D(32, 3, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
```

Some successful architectures reverse the order (activation before batch norm). There is no universal rule — Foster's advice: **experiment on a holdout set**. For this course, follow BAD unless a specific generative architecture dictates otherwise.

Batch normalization also acts as mild [regularization](../../GLOSSARY.md#regularization) because batch statistics inject noise. Many modern discriminative models rely on batch norm alone; generative models often combine it with other techniques (spectral normalization in GANs, weight decay in diffusion).

---

## Dropout: Forcing Generalization

[Dropout](../../GLOSSARY.md#dropout) (Hinton et al., 2012; Srivastava et al., 2014) is a deliberately simple [regularization](../../GLOSSARY.md#regularization) technique. During training, each dropout layer **randomly sets a fraction of units to zero**.

Foster's exam analogy: a student who memorizes answers to practice questions fails on unseen exam questions. A student who learns underlying principles generalizes. Dropout prevents the network from relying on any single unit or small group of units that "memorize" specific training examples.

### Training behavior

Each forward pass, independently for each unit:

$$
h_i^{\text{dropped}} = \begin{cases} 0 & \text{with probability } p \\ \frac{h_i}{1-p} & \text{otherwise} \end{cases}
$$
> **Readable form:** Randomly zero out fraction p of activations. Scale survivors up by 1/(1-p) so expected sum stays the same.

The scaling (inverted dropout) keeps the expected activation magnitude stable.

### Inference behavior

**All units are active.** No units are dropped. The network uses its full capacity.

```python
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(rate=0.5)(x)  # drop 50% of units during training only
```

There are **no learnable weights** in a dropout layer — only the `rate` [hyperparameter](../../GLOSSARY.md#hyperparameter).

### Where to apply dropout

Foster notes dropout is most common **after Dense layers** (many parameters, high overfitting risk). It can also follow conv layers, but batch norm often suffices there.

| Location | Typical `rate` | Rationale |
|----------|----------------|-----------|
| After large Dense | 0.5 | Many weights to regularize |
| After conv blocks | 0.25 | Lighter touch |
| Generator output | Rarely | Would hurt sample quality |

Modern GAN generators typically **omit dropout** in the generator path — you want rich, deterministic generation. Discriminators may still use it, though spectral normalization is more common in state-of-the-art GANs.

---

## Side-by-Side Comparison

| | Batch Normalization | Dropout |
|---|---------------------|---------|
| **Primary goal** | Stabilize training / speed convergence | Reduce overfitting |
| **Training** | Normalize per batch; update running stats | Randomly zero units |
| **Inference** | Use running mean/variance | Pass all units through |
| **Learnable params** | $\gamma$, $\beta$ per channel | None |
| **`training` flag matters?** | Yes — critical | Yes — critical |
| **Generative usage** | Ubiquitous (encoders, discriminators, U-Nets) | Less common in generators |

Foster notes that batch normalization alone can reduce overfitting enough that some architectures skip dropout entirely. The only way to know what works: **try both on your holdout set**.

---

## Foster's BAD Stack in Context

The full pattern from Foster's CIFAR-10 CNN (built in [Section 2.8](./section-08-building-and-training-a-cnn.md)):

```python
from tensorflow.keras import layers

def conv_block(x, filters, strides=1):
    x = layers.Conv2D(filters, 3, strides=strides, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    return x

def dense_block(x, units, dropout_rate=0.5):
    x = layers.Dense(units)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    return x
```

LeakyReLU (instead of ReLU) allows small negative gradients — Foster uses it throughout the CNN chapter. The negative slope prevents "dead neurons" that never activate.

---

## Connection to Generative Models

| Technique | Generative role |
|-----------|-----------------|
| Batch norm in encoder | Stable gradients when compressing images to latents (VAE, Chapter 03) |
| Batch norm in discriminator | Stabilizes GAN training alongside the generator (Chapter 04) |
| Batch norm in U-Net | Normalizes activations across denoising steps (Chapter 08) |
| Dropout in discriminator | Mild regularization so critic doesn't overpower generator |
| `training=False` when sampling | **Mandatory** for VAE `decoder(z)`, GAN `generator(z)`, diffusion sampling |

GAN training tip preview: if generated samples look like gray mush after thousands of steps, check whether you called the generator with `training=False`.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| `training=True` at sample time | Blurry or inconsistent generations | `model(z, training=False)` |
| Dropout left "on" during inference | Weakened predictions | Keras handles this in `predict()`; be careful in custom loops |
| Batch norm before huge Dense on tiny batches | Noisy normalization | Increase batch size or use group/layer norm (advanced) |
| Dropout rate too high in generator | Mode collapse, dull outputs | Reduce rate or remove from generator |
| Batch norm + high LR without tuning | Instability | Lower learning rate; batch norm often permits *higher* LR, not unlimited |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Internal covariate shift** | Change in input distribution to a layer as preceding weights update |
| **Batch normalization** | Normalize activations per channel using batch statistics; learn scale and shift |
| **Running average** | Exponential moving average of batch mean/variance used at inference |
| **Dropout** | Stochastic zeroing of units during training for regularization |
| **Dropout rate** | Fraction $p$ of units dropped each forward pass |
| **BAD ordering** | Batch norm → Activation → Dropout (Foster's recommended stack) |
| **Exploding gradient** | Gradients grow exponentially, causing NaN weights |

---

## Reflection Questions

1. Why can't batch normalization use the current batch's statistics when you predict a single image?
2. A `Dropout(0.5)` layer follows `Dense(128)`. How many units are active on average during training? At inference?
3. Why might a GAN generator omit dropout while the discriminator includes it?
4. What happens if you forget `training=False` when calling a VAE decoder on random latent vectors?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 2 — Batch Normalization; Dropout. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)
- Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. [http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- Foster's codebase: [https://github.com/davidADSP/GDL_code](https://github.com/davidADSP/GDL_code) — `notebooks/02_deeplearning/02_cnn/cnn.ipynb`

---

**Previous:** [Section 2.6 — Convolutional Layers](./section-06-convolutional-layers.md)  
**Next:** [Section 2.8 — Building and Training a CNN](./section-08-building-and-training-a-cnn.md)


