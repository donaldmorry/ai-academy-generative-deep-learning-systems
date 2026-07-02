# Section 3.7: CelebA Faces

> **Source inheritance:** Foster, Ch. 3 — "Exploring the Latent Space" & CelebA training  
> **Enhanced with:** 200-D latent space, RGB pipeline, BatchNorm, and face generation at scale  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Fashion-MNIST at 2D taught you the mechanics of [variational autoencoders](../../GLOSSARY.md#variational-autoencoder). **CelebA** asks the real question: can a VAE learn a 200-dimensional face manifold from 200,000 RGB celebrity photos and produce **novel convincing faces** from pure Gaussian noise? Foster's answer — after five epochs with batch normalization, LeakyReLU, and $\beta = 2000$ — is yes. This is your first glimpse of generative deep learning at production-adjacent scale.

---

## The CelebA Dataset

**CelebFaces Attributes (CelebA)** contains 202,599 aligned face images with 40 binary attribute labels (Smiling, Eyeglasses, Male, etc.). Labels are optional for VAE training but essential for [latent arithmetic](./section-08-latent-space-arithmetic.md).

Download via Foster's Kaggle script:

```bash
bash scripts/download_kaggle_data.sh jessicali9530 celeba-dataset
```

```python
import tensorflow as tf

train_data = tf.keras.utils.image_dataset_from_directory(
    "/app/data/celeba-dataset/img_align_celeba/img_align_celeba",
    labels=None,
    color_mode="rgb",
    image_size=(64, 64),
    batch_size=128,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
)

def preprocess(img):
    return tf.cast(img, "float32") / 255.0

train = train_data.map(lambda x: preprocess(x))
```

`image_dataset_from_directory` streams batches — critical when 200K images exceed RAM.

---

## Architecture Changes from Fashion-MNIST

| Component | Fashion-MNIST | CelebA faces |
|-----------|---------------|--------------|
| Input shape | $(32, 32, 1)$ | $(64, 64, 3)$ |
| Latent dim | 2 | **200** |
| Activations | ReLU | **LeakyReLU** |
| Normalization | None | **BatchNorm** after conv |
| $\beta$ (recon weight) | 500 | **2000** |
| Decoder output | 1 ch, sigmoid | **3 ch, sigmoid** |

Foster's encoder summary (Table 3-5): four stride-2 Conv2D blocks → Flatten → `z_mean` / `z_log_var` Dense(200).

```python
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    return x

encoder_input = layers.Input(shape=(64, 64, 3))
x = conv_block(encoder_input, 128)
x = conv_block(x, 128)
x = conv_block(x, 128)
x = conv_block(x, 128)
shape_before_flattening = K.int_shape(x)[1:]  # (2, 2, 128)
x = layers.Flatten()(x)
z_mean = layers.Dense(200, name="z_mean")(x)
z_log_var = layers.Dense(200, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = models.Model(encoder_input, [z_mean, z_log_var, z])
```

Decoder mirrors with Conv2DTranspose blocks, final `Conv2DTranspose(3, 3, activation="sigmoid")`.

---

## Why 200 Dimensions?

Face variation spans pose, expression, lighting, identity, hair, accessories — far more factors than clothing type on a gray 28×28 icon.

$$
\dim(z) \approx \text{\# independent factors of variation you need to capture}
$$
> **Readable form:** Two numbers cannot encode a face; two hundred is Foster's empirical sweet spot for 64×64 RGB.

Too few dimensions → blurry mush. Too many without KL → holes return ([Section 3.3](./section-03-visualizing-latent-space.md)).

---

## Training Configuration

```python
vae_faces = VAE(encoder, decoder, beta=2000.0)
vae_faces.compile(optimizer="adam")

vae_faces.fit(train, epochs=5)
```

BatchNorm stabilizes activations across the 128-image batches — Foster notes each batch is slower but total convergence faster.

**Monitor latent marginals** after training (Figure 3-17): first 50 dimensions should resemble $\mathcal{N}(0,1)$. Outliers suggest KL is too weak — reduce $\beta$ on reconstruction.

```python
z_mean, _, _ = vae_faces.encoder.predict(next(iter(train.take(1))), verbose=0)
# For full analysis, encode a larger batch via dataset iteration
```

---

## Reconstruction at 200-D

Figure 3-16: top row originals, bottom row reconstructions. Key features preserved:

- Head angle and pose
- Hairstyle and hair color
- Expression (smile vs neutral)

Missing: skin pore detail, fine eye texture. Foster explicitly states **perfect reconstruction is not the goal** — generation is.

---

## Generating Novel Faces

```python
import numpy as np
import matplotlib.pyplot as plt

grid_width, grid_height = 10, 3
z_sample = np.random.normal(size=(grid_width * grid_height, 200))
reconstructions = decoder.predict(z_sample, verbose=0)

fig = plt.figure(figsize=(18, 5))
for i in range(grid_width * grid_height):
    ax = fig.add_subplot(grid_height, grid_width, i + 1)
    ax.axis("off")
    ax.imshow(np.clip(reconstructions[i], 0, 1))
plt.suptitle("Generated CelebA faces from $\mathcal{N}(0,I)$")
plt.show()
```

Figure 3-18 — 30 faces from random $z$ — is the course's first "wow" moment: structured diversity without an adversarial game ([Chapter 04](../chapter-04-generative-adversarial-networks/section-01-gan-introduction.md)).

---

## VAE Faces vs GAN Faces (Preview)

| Aspect | VAE (this section) | DCGAN (Chapter 04) |
|--------|-------------------|-------------------|
| Training stability | Generally stable | Can collapse |
| Sample sharpness | Softer | Often crisper |
| Latent arithmetic | Natural | Possible but less smooth |
| Likelihood | ELBO bound | Implicit |

Neither dominates — they occupy different points on the fidelity/stability frontier.

---

## Practical Training Tips

| Tip | Rationale |
|-----|-----------|
| Use `tf.data` pipeline | Memory-efficient at 200K images |
| Cache or prefetch | `train.prefetch(tf.data.AUTOTUNE)` |
| Save checkpoints each epoch | Five epochs still costly on CPU |
| Encode subset for arithmetic | Full-dataset encode is slow — sample 10K |

```python
train = train.prefetch(tf.data.AUTOTUNE)
```

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Wrong image path | Empty dataset | Verify Kaggle download structure |
| Grayscale mode | 1-channel mismatch | `color_mode="rgb"` |
| $\beta = 500$ from Fashion | Weak KL on faces | Use Foster's 2000 |
| Sigmoid + pixels outside [0,1] | Washed outputs | Scale inputs to [0,1] |
| Expecting photographic sharpness | "Blurry" complaint | Compare to GAN in Chapter 04 |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **CelebA** | Large-scale celebrity face dataset with attribute labels |
| **tf.data pipeline** | Streaming batch loader for large image corpora |
| **LeakyReLU** | Activation allowing small negative gradients |
| **Batch normalization** | Per-batch activation standardization |
| **High-dimensional latent** | 200-D $z$ capturing rich face factors |

---

## Reflection Questions

1. Why does Foster increase $\beta$ from 500 (Fashion) to 2000 (CelebA)?
2. What does it mean that reconstruction is imperfect but generation succeeds?
3. How would you verify KL regularization is working in 200 dimensions without a 2D scatter plot?
4. Why are attribute labels unused during `fit()` but valuable afterward?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 3 — Exploring the Latent Space. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Liu, Z. et al. (2015). Deep Learning Face Attributes in the Wild (CelebA).
- Foster's codebase: `notebooks/03_vae/03_faces/vae_faces.ipynb`
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.

---

**Previous:** [Section 3.6 — Training the VAE](./section-06-training-the-vae.md)  
**Next:** [Section 3.8 — Latent Space Arithmetic](./section-08-latent-space-arithmetic.md)



