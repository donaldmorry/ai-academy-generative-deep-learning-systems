# Section 4.2: DCGAN Architecture

> **Source inheritance:** Foster, Ch. 4 — "Deep Convolutional GAN (DCGAN)"  
> **Enhanced with:** Bricks dataset, discriminator/generator specs, DCGAN guidelines, Conv2DTranspose vs upsampling  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

The 2015 DCGAN paper showed that [GANs](../../GLOSSARY.md#generative-adversarial-network-gan) become dramatically more stable when you replace fully connected layers with **convolutional** ones and follow a handful of architectural rules. Foster builds a DCGAN on 64×64 grayscale LEGO brick renders: the discriminator shrinks spatial resolution while growing channels; the generator does the reverse with `Conv2DTranspose`, turning a 100-D noise vector into a brick image in $[-1, 1]$.

This section is pure architecture — training arrives in [Section 4.3](./section-03-training-the-dcgan.md).

---

## Bricks Dataset Pipeline

```bash
bash scripts/download_kaggle_data.sh joosthazelzet lego-brick-images
```

```python
import tensorflow as tf

train_data = tf.keras.utils.image_dataset_from_directory(
    "/app/data/lego-brick-images/dataset/",
    labels=None,
    color_mode="grayscale",
    image_size=(64, 64),
    batch_size=128,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
)

def preprocess(img):
    # GAN convention: scale to [-1, 1] for tanh generator output
    return (tf.cast(img, "float32") - 127.5) / 127.5

train = train_data.map(lambda x: preprocess(x))
```

| Preprocessing choice | Reason |
|---------------------|--------|
| Grayscale | Simpler first GAN; 1 output channel |
| 64×64 | Standard DCGAN resolution |
| $[-1, 1]$ range | Matches `tanh` on generator final layer |

VAEs used $[0,1]$ + sigmoid ([Chapter 03](../chapter-03-variational-autoencoders/section-01-autoencoder-architecture.md)); GANs prefer symmetric ranges for stronger gradients.

---

## Discriminator Architecture

Binary image classifier: real vs fake. Foster's summary (Table 4-1):

| Layer | Output shape | Notes |
|-------|--------------|-------|
| Input | (None, 64, 64, 1) | Grayscale brick |
| Conv2D 64, k=4, s=2 | (None, 32, 32, 64) | `use_bias=False` |
| LeakyReLU(0.2) + Dropout(0.3) | | |
| Conv2D 128, s=2 | (None, 16, 16, 128) | + BatchNorm |
| Conv2D 256, s=2 | (None, 8, 8, 256) | |
| Conv2D 512, s=2 | (None, 4, 4, 512) | |
| Conv2D 1, k=4, s=1, valid | (None, 1, 1, 1) | Sigmoid |
| Flatten | (None, 1) | Realness score |

```python
from tensorflow.keras import layers, models

discriminator_input = layers.Input(shape=(64, 64, 1))
x = layers.Conv2D(64, 4, strides=2, padding="same", use_bias=False)(discriminator_input)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(128, 4, strides=2, padding="same", use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(256, 4, strides=2, padding="same", use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(512, 4, strides=2, padding="same", use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(1, 4, strides=1, padding="valid", use_bias=False, activation="sigmoid")(x)
discriminator_output = layers.Flatten()(x)
discriminator = models.Model(discriminator_input, discriminator_output)
discriminator.summary()
```

**Spatial walkthrough:** $64 \to 32 \to 16 \to 8 \to 4 \to 1$ via stride-2 convolutions — mirror of VAE encoder ([Section 2.6](../chapter-02-deep-learning/section-06-convolutional-layers.md)).

---

## Generator Architecture

Maps $z \in \mathbb{R}^{100}$ to $(64, 64, 1)$:

| Layer | Output shape |
|-------|--------------|
| Input $z$ | (None, 100) |
| Reshape | (None, 1, 1, 100) |
| Conv2DTranspose 512, k=4, s=1 | (None, 4, 4, 512) |
| Conv2DTranspose 256, s=2 | (None, 8, 8, 256) |
| Conv2DTranspose 128, s=2 | (None, 16, 16, 128) |
| Conv2DTranspose 64, s=2 | (None, 32, 32, 64) |
| Conv2DTranspose 1, s=2, tanh | (None, 64, 64, 1) |

```python
generator_input = layers.Input(shape=(100,))
x = layers.Reshape((1, 1, 100))(generator_input)
x = layers.Conv2DTranspose(512, 4, strides=1, padding="valid", use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(256, 4, strides=2, padding="same", use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
generator_output = layers.Conv2DTranspose(
    1, 4, strides=2, padding="same", use_bias=False, activation="tanh"
)(x)
generator = models.Model(generator_input, generator_output)
```

$1 \to 4 \to 8 \to 16 \to 32 \to 64$ — inverse of discriminator.

---

## DCGAN Design Guidelines (Radford et al.)

| Guideline | Foster's implementation |
|-----------|-------------------------|
| Replace pooling with strided conv | `strides=2` on Conv2D / Conv2DTranspose |
| Use BatchNorm in generator | After each transpose conv |
| Use BatchNorm in discriminator | Not on first layer |
| Use LeakyReLU (slope 0.2) | Both networks |
| Use ReLU in generator middle | LeakyReLU in Foster's code |
| Tanh output + $[-1,1]$ inputs | `preprocess` scaling |
| Avoid fully connected hidden layers | Only $z$ input and 1×1×1 collapse |

---

## Conv2DTranspose vs UpSampling2D

Foster documents checkerboard artifacts from transpose conv (Figure 4-4, Odena et al.):

```python
# Alternative upsampling path
x = layers.UpSampling2D(size=2)(x)
x = layers.Conv2D(256, 4, strides=1, padding="same")(x)
```

| Method | Pros | Cons |
|--------|------|------|
| Conv2DTranspose | Standard in literature; fewer layers | Checkerboard risk |
| UpSample + Conv2D | Smoother in some settings | Extra layer per scale |

Test both on your dataset — Foster notes transpose conv remains dominant in state-of-the-art GANs despite artifacts.

---

## Parameter Counts

| Network | Total params (Foster) |
|---------|----------------------|
| Discriminator | ~2.77M |
| Generator | ~3.58M |

Generator is slightly larger — typical, since creating images is harder than judging them.

---

## Connection to VAE Decoder

The generator **is** a VAE decoder without an encoder:

$$
\text{DCGAN: } z \sim \mathcal{N}(0,I) \xrightarrow{G} \text{image}
$$
> **Readable form:** A DCGAN generator maps random latent noise directly into an image sample.

$$
\text{VAE: } x \xrightarrow{\text{enc}} z \xrightarrow{\text{dec}} \hat{x}
$$
> **Readable form:** A VAE encodes an input into a latent code and decodes that latent code back into a reconstruction.

Same Conv2DTranspose pattern as [Section 3.1](../chapter-03-variational-autoencoders/section-01-autoencoder-architecture.md), different training signal.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Sigmoid output + $[-1,1]$ data | Saturated gradients | Use tanh + symmetric scaling |
| BatchNorm on D's first layer | Unstable training | BN from second layer onward |
| Mismatched spatial sizes | Shape errors | Trace $64 \to 1$ and $1 \to 64$ by hand |
| `use_bias=True` with BatchNorm | Redundant params | `use_bias=False` as in DCGAN paper |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **DCGAN** | Convolutional GAN following Radford et al. architectural guidelines |
| **Conv2DTranspose** | Learned upsampling layer doubling spatial dims with stride 2 |
| **LeakyReLU** | Activation with small negative slope (0.2) preventing dead neurons |
| **Latent dim** | Size of noise vector $z$ (100 for bricks) |
| **Checkerboard artifact** | Grid pattern from transpose conv overlap patterns |

---

## Reflection Questions

1. Why does the discriminator use stride-2 convolutions instead of pooling?
2. Trace the generator output shape from `(None, 100)` to `(None, 64, 64, 1)` layer by layer.
3. Why scale brick images to $[-1, 1]$ instead of $[0, 1]$?
4. How is the generator architecturally similar to the VAE decoder?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 4 — DCGAN. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Radford, A. et al. (2015). DCGAN. [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434)
- Foster's codebase: `notebooks/04_gan/01_dcgan/dcgan.ipynb`
- Odena, A. et al. (2016). Deconvolution and Checkerboard Artifacts. [https://distill.pub/2016/deconv-checkerboard](https://distill.pub/2016/deconv-checkerboard)

---

**Previous:** [Section 4.1 — GAN Introduction](./section-01-gan-introduction.md)  
**Next:** [Section 4.3 — Training the DCGAN](./section-03-training-the-dcgan.md)
