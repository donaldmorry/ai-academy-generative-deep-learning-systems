# Section 3.1: Autoencoder Architecture

> **Source inheritance:** Foster, Ch. 3 — "The Autoencoder Architecture"  
> **Enhanced with:** Encoder-decoder design, Fashion-MNIST preprocessing, Conv2DTranspose upsampling, and generative sampling intuition  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

An **autoencoder** is a [neural network](../../GLOSSARY.md#neural-network) trained to copy its input to its output through a narrow **bottleneck** — the [latent space](../../GLOSSARY.md#latent-variable). Foster tells the story of an infinite 2D wardrobe: you (the **encoder**) place each clothing item at coordinates $(z_1, z_2)$; Brian (the **decoder**) sews the item back from those coordinates alone.

The twist that makes this **generative**: Brian can sew from *any* coordinate, not only ones you used during training. Pick an empty spot in the wardrobe → decode → novel clothing item. Imperfect, but unmistakably generative.

> **Readable form:** Compress image to a short code, decompress back to pixels. Train on reconstruction. Sample random codes to generate new images.

This section builds Foster's Fashion-MNIST autoencoder with Conv2D encoder, Conv2DTranspose decoder, and binary cross-entropy loss.

---

## The Wardrobe Analogy vs the Math

| Story element | Neural network role |
|---------------|---------------------|
| Clothing item | Input image $x$ |
| Wardrobe coordinate | Latent vector $z = f_\phi(x)$ |
| Brian's sewing | Reconstruction $\hat{x} = g_\theta(z)$ |
| Empty coordinate | Random $z$ not seen during training |
| Novel garment | Generated $\hat{x}$ from sampled $z$ |

Formally, the autoencoder learns:

$$
\hat{x} = g_\theta(f_\phi(x))
$$
> **Readable form:** Reconstruction equals decoder applied to encoder output.

Training minimizes distance between $x$ and $\hat{x}$. The bottleneck forces $z$ to capture only what is needed for reconstruction — edges, shapes, garment type — not pixel-level noise.

---

## Fashion-MNIST Preprocessing

Foster uses **Fashion-MNIST**: 28×28 grayscale clothing images (10 classes). Steps:

1. Load via `datasets.fashion_mnist.load_data()`
2. Scale pixels to $[0, 1]$
3. **Pad** to 32×32 (easier conv arithmetic with stride-2 layers)
4. Add channel dimension → shape $(32, 32, 1)$

```python
import numpy as np
from tensorflow.keras import datasets

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

def preprocess(imgs):
    imgs = imgs.astype("float32") / 255.0
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    imgs = np.expand_dims(imgs, -1)  # (N, 32, 32, 1)
    return imgs

x_train = preprocess(x_train)
x_test = preprocess(x_test)
```

Labels are **not** used during autoencoder training — this is unsupervised reconstruction. Labels return in [Section 3.3](./section-03-visualizing-latent-space.md) for coloring scatter plots only.

---

## Encoder Architecture

The encoder compresses $(32, 32, 1) \rightarrow (2,)$ via strided convolutions:

| Layer | Output shape | Role |
|-------|--------------|------|
| Input | (None, 32, 32, 1) | Grayscale image |
| Conv2D(32, 3×3, stride 2) | (None, 16, 16, 32) | Halve spatial, learn low-level features |
| Conv2D(64, 3×3, stride 2) | (None, 8, 8, 64) | Mid-level patterns |
| Conv2D(128, 3×3, stride 2) | (None, 4, 4, 128) | High-level structure |
| Flatten | (None, 2048) | Vectorize |
| Dense(2) | (None, 2) | **2D latent embedding** |

```python
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

encoder_input = layers.Input(shape=(32, 32, 1), name="encoder_input")
x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]  # (4, 4, 128) — needed by decoder
x = layers.Flatten()(x)
encoder_output = layers.Dense(2, name="encoder_output")(x)
encoder = models.Model(encoder_input, encoder_output)
encoder.summary()
```

**Stride 2** halves height and width each conv layer: $32 \rightarrow 16 \rightarrow 8 \rightarrow 4$. Channel depth grows (32 → 64 → 128) — the standard CNN pattern from [Chapter 02](../chapter-02-deep-learning/section-06-convolutional-layers.md).

---

## Decoder and Conv2DTranspose

The decoder mirrors the encoder. Standard `Conv2D` shrinks spatial size; **Conv2DTranspose** doubles it (with `strides=2`).

| Layer | Output shape |
|-------|--------------|
| Input (latent) | (None, 2) |
| Dense(2048) + Reshape | (None, 4, 4, 128) |
| Conv2DTranspose(128, stride 2) | (None, 8, 8, 128) |
| Conv2DTranspose(64, stride 2) | (None, 16, 16, 64) |
| Conv2DTranspose(32, stride 2) | (None, 32, 32, 32) |
| Conv2D(1, sigmoid) | (None, 32, 32, 1) |

```python
decoder_input = layers.Input(shape=(2,), name="decoder_input")
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
decoder_output = layers.Conv2D(
    1, (3, 3), strides=1, activation="sigmoid", padding="same", name="decoder_output"
)(x)
decoder = models.Model(decoder_input, decoder_output)
```

**Sigmoid** on the final layer outputs pixels in $[0, 1]$, matching normalized input range. GAN generators use `tanh` and $[-1, 1]$ scaling instead ([Chapter 04](../chapter-04-generative-adversarial-networks/section-02-dcgan-architecture.md)).

**Conv2DTranspose caveat:** can produce checkerboard artifacts in some architectures. Foster notes `UpSampling2D` + `Conv2D` as an alternative — test both on your dataset.

---

## Wiring Encoder to Decoder

The Functional API makes composition trivial:

```python
autoencoder = models.Model(encoder_input, decoder(encoder_output))
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
```

Forward path: image → encoder → 2D point → decoder → reconstructed image.

Training target equals input:

```python
autoencoder.fit(
    x_train, x_train,
    epochs=5,
    batch_size=100,
    shuffle=True,
    validation_data=(x_test, x_test),
)
```

> **Readable form:** The network learns to output the same image it received — but only after squeezing through 2 numbers.

---

## Loss Function: BCE vs RMSE

Foster discusses two reconstruction losses:

**Binary cross-entropy** (per pixel, treated as independent Bernoullis):

$$
\mathcal{L}_{\text{BCE}} = -\sum_{i} \left[ x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i) \right]
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

**RMSE** (symmetric penalty on pixel error):

$$
\mathcal{L}_{\text{RMSE}} = \sqrt{\frac{1}{n}\sum_i (x_i - \hat{x}_i)^2}
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

| Loss | Visual effect |
|------|---------------|
| BCE | Slightly blurrier (pushes uncertain pixels toward 0.5) |
| RMSE | Sharper edges, sometimes pixelated |

No universal winner — experiment on your data. Fashion-MNIST autoencoder uses BCE with sigmoid outputs.

---

## Why This Is a Generative Model

After training, sample $z \sim \text{Uniform}([-5, 5]^2)$ or pick grid points, decode:

```python
import matplotlib.pyplot as plt

n = 20
z_grid = np.linspace(-2, 2, n)
decoded = []
for x in z_grid:
    for y in z_grid:
        z = np.array([[x, y]])
        decoded.append(decoder.predict(z, verbose=0))
# Plot grid of decoded images
```

Quality depends on latent space structure — [Section 3.3](./section-03-visualizing-latent-space.md) shows why naive sampling fails, motivating the [VAE](../../GLOSSARY.md#variational-autoencoder-vae).

---

## Denoising Autoencoder Preview

Foster notes autoencoders can **denoise**: train on noisy inputs, reconstruct clean targets. The encoder learns noise is not worth storing in latent space. Requires larger bottlenecks than 2D for real denoising — but the principle foreshadows [diffusion models](../../GLOSSARY.md#diffusion-model) (Chapter 08), which denoise iteratively.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Latent dim too small (2D) | Heavy blur, lost detail | Increase Dense units; accept harder visualization |
| Mismatched Reshape shape | Shape error in decoder | Save `shape_before_flattening` from encoder |
| Linear output + BCE | Wrong loss pairing | Sigmoid + BCE, or linear + MSE |
| Forgetting padding | Shape mismatch at 28 vs 32 | Pad Fashion-MNIST to 32×32 |

---

## Connection to Other Chapters

| Concept | Chapter | Link |
|---------|--------|------|
| Conv2D / stride | 02 Deep Learning | Encoder downsampling |
| Conv2DTranspose | 04 DCGAN generator | Same upsampling pattern |
| Latent $z$ → image | 04 GAN generator | GAN samples $z$ from Gaussian; no encoder |
| ELBO + probabilistic $z$ | 03 Sections 4–6 | VAE upgrade of this architecture |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Autoencoder** | Network trained to reconstruct input through a compressed bottleneck |
| **Encoder** | $f_\phi(x)$ mapping data to latent code $z$ |
| **Decoder** | $g_\theta(z)$ mapping latent code back to data space |
| **Latent space** | Low-dimensional representation where similar inputs cluster |
| **Embedding** | The latent vector $z$ encoding compressed information |
| **Bottleneck** | Narrow layer forcing compression (here, Dense(2)) |
| **Conv2DTranspose** | Layer that upsamples spatial dimensions via learned filters |

---

## Reflection Questions

1. Why does the encoder use stride-2 convolutions instead of pooling + conv?
2. What happens if you use a 512-dimensional latent instead of 2D? Trade-offs?
3. Why is `autoencoder.fit(x_train, x_train)` valid — what is the supervised target?
4. How does the wardrobe story explain generation from empty latent coordinates?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). O'Reilly. Ch. 3 — Autoencoders. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Foster's codebase: `notebooks/03_vae/01_autoencoder/autoencoder.ipynb`
- Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning.
- Hinton, G., & Salakhutdinov, R. (2006). Reducing the dimensionality of data with neural networks.

---

**Previous:** [Chapter 03 Overview](./README.md)  
**Next:** [Section 3.2 — Reconstructing Images](./section-02-reconstructing-images.md)



