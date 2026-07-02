# Section 1.5: Representation Learning

> **Source inheritance:** Foster, Ch. 1 - representation learning and latent variables  
> **Enhanced with:** Latent space intuition, autoencoder bridge, creative applications  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)    
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Problem with Raw Data

A 256×256 RGB image is 196,608 numbers. Most random combinations are static noise. Real photographs occupy a tiny, twisted subset of that space - a **manifold** of meaningful images.

If a [generative model](../../GLOSSARY.md#generative-model) tried to memorize every valid image directly, it would need infinite storage. Instead, successful models learn **representations** - compressed descriptions that capture the essence of the data.

> **Readable form:** You do not memorize every sentence in English to speak the language. You internalize grammar, vocabulary, and concepts - a compressed representation - then generate new sentences on the fly. Generative models do the same with pixels, notes, or tokens.

---

## What Is Representation Learning?

**Representation learning** is the automatic discovery of useful features from raw data without hand-engineering.

| Raw input | Learned representation | What it captures
|-----------|----------------------|------------------|
| Pixels | Convolutional feature maps | Edges → textures → parts → objects |
| MIDI notes | Embedding vectors | Melody, harmony, rhythm patterns |
| Words | Token embeddings | Semantic meaning, analogies |
| Audio waveform | Spectrogram latents | Timbre, pitch, tempo |

In [Course 3, Chapter 15](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-15-representation-learning/README.md), Goodfellow formalizes this as learning a mapping $f(x)$ that makes downstream tasks easier. In generative modeling, representations serve a dual purpose: **compression** and **generation**.

---

## Latent Variables: The ZIP File of Meaning

A [latent variable](../../GLOSSARY.md#latent-variable) $z$ is a hidden factor that generates observed data $x$:

$$
P(x) = \int P(x \mid z) \, P(z) \, dz
$$
> **Readable form:** Marginal P(x) integrates joint over continuous latent space.

Think of $z$ as a **ZIP file of meaning**:

- **Unzip** (decode): $x = g_\theta(z)$ - turn a compact code into an image
- **Zip** (encode): $z = q_\phi(x)$ - compress an image into its essence
- **Edit the ZIP**: tweak $z$ - add a smile, change hair color, shift musical key

> **Readable form:** The latent space is the model's imagination room. It is small enough to navigate but rich enough to describe all the variations in the data. Stable Diffusion operates in a 4×64×64 latent space instead of 3×512×512 pixel space - that is a 96× compression before generation even begins.

---

## The Manifold Hypothesis

High-dimensional data often lies on or near a low-dimensional manifold $\mathcal{M}$ embedded in $\mathbb{R}^n$.

```
Full pixel space:  ℝ^196608  (mostly noise)
                        ↓
Data manifold:     ℝ^d, d ≪ 196608  (actual images)
                        ↓
Latent space:      z ∈ ℝ^k  (what the model learns)
```

A generative model learns:

1. A decoder $p_\theta(x \mid z)$ - manifold to pixels
2. Often an encoder $q_\phi(z \mid x)$ - pixels to manifold

When $k$ is small, the model cannot memorize - it must learn **structure**.

---

## Autoencoders: The Gateway Drug

The simplest representation-learning architecture is the **autoencoder**:

```python
import tensorflow as tf
from tensorflow import keras

latent_dim = 32
input_shape = (28, 28, 1)

# Encoder: x → z
encoder = keras.Sequential([
    keras.layers.Input(shape=input_shape),
    keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"),
    keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"),
    keras.layers.Flatten(),
    keras.layers.Dense(latent_dim, name="latent"),
], name="encoder")

# Decoder: z → x̂
decoder = keras.Sequential([
    keras.layers.Input(shape=(latent_dim,)),
    keras.layers.Dense(7 * 7 * 64),
    keras.layers.Reshape((7, 7, 64)),
    keras.layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same"),
    keras.layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same"),
    keras.layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same"),
], name="decoder")

# Full autoencoder
z = encoder(keras.Input(shape=input_shape))
reconstruction = decoder(z)
autoencoder = keras.Model(
    inputs=encoder.input,
    outputs=reconstruction,
    name="autoencoder",
)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
```

This is **not yet generative** - a standard autoencoder learns a compressed representation but does not define a distribution over $z$. You cannot sample random $z$ values and expect meaningful images.

**Bold milestone:** The jump from autoencoder to **VAE** (Variational Autoencoder) - adding $P(z)$ and making sampling work - is [Chapter 03](../chapter-03-variational-autoencoders/README.md). You previewed autoencoders in [Course 3, Chapter 14](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-14-autoencoders/README.md).

---

## Hierarchical Representations

Deep networks build **feature hierarchies**:

```
Layer 1:  edges, color blobs
Layer 2:  textures, simple shapes
Layer 3:  object parts (eyes, wheels)
Layer 4:  objects (faces, cars)
Latent z:  identity, pose, style
```

This hierarchy is why a GAN's latent vector can control high-level attributes. In StyleGAN, different layers of the generator control coarse structure (pose) vs fine detail (freckles). In GPT, early layers capture syntax; deeper layers capture semantics and reasoning.

---

## Latent Space Arithmetic

When representations disentangle factors of variation, algebra emerges:

$$
z_{\text{king}} - z_{\text{man}} + z_{\text{woman}} \approx z_{\text{queen}}
$$
> **Readable form:** Vector arithmetic in latent space: king minus man plus woman approximates queen embedding.

For faces:

$$
z_{\text{smiling}} - z_{\text{neutral}} + z_{\text{person A}} \approx z_{\text{person A smiling}}
$$
> **Readable form:** Add smiling direction vector to a person's latent code to approximate that person smiling.

```python
import numpy as np

# Conceptual latent arithmetic (z vectors from a trained encoder)
z_man = encoder.predict(man_image)
z_woman = encoder.predict(woman_image)
z_king = encoder.predict(king_image)

z_queen_approx = z_king - z_man + z_woman
queen_image = decoder.predict(z_queen_approx)
```

This is not magic - it means the latent space organized gender and royalty as (approximately) linear directions.

> **Readable form:** The model built a coordinate system for concepts. Move along the "gender" axis, stay on the "royalty" axis, and you arrive at a new valid point in face-space.

---

## Representation Learning in Modern Systems

| System | Representation | Latent space role
|--------|---------------|-------------------|
| **Stable Diffusion** | VAE encoder/decoder | Compress images to 4×64×64; diffuse in latent space |
| **CLIP** | Image + text encoders | Shared embedding space for conditioning |
| **GPT** | Token embeddings + hidden states | Contextual representations of language |
| **MusicLM** | SoundStream codec | Discrete acoustic tokens for generation |

Every system in this course uses representation learning. The coin flip had no latent variable - just $\theta$. Real data demands compression.

---

## Unsupervised Structure Discovery

Representation learning is typically **unsupervised** - no labels required. The model discovers:

- Clusters (similar faces group together)
- Directions (smile, age, lighting)
- Disentanglement (ideally, one latent dimension = one factor)

This connects to [Course 1, Section 1.3](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-01-machine-learning/section-03-supervised-vs-unsupervised-learning.md) and k-means clustering - but neural representations are far richer than centroid-based clusters.

---

## Key Vocabulary

| Term | Definition
|------|-----------|
| **Latent variable $z$** | Hidden code generating observed $x$ |
| **Encoder** | Maps $x \to z$ |
| **Decoder** | Maps $z \to x$ |
| **Manifold** | Low-dimensional structure in high-dimensional space |
| **Disentanglement** | Separating independent factors in latent dimensions |

---

## Reflection Questions

1. Why can't you sample random $z$ from a standard autoencoder and get valid images?
2. What does "ZIP file of meaning" correspond to mathematically in $P(x) = \int P(x|z)P(z)\,dz$?
3. Name one attribute you would want as a controllable direction in a face latent space.

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 1, 3. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
- Bengio, Y., Courville, A., & Vincent, P. (2013). Representation Learning: A Review and New Perspectives. [https://arxiv.org/abs/1206.5538](https://arxiv.org/abs/1206.5538)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. Ch. 15 - Representation Learning. [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
- Radford, A. et al. (2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434)

---

**Previous:** [Section 1.4 - Our First Generative Model](./section-04-our-first-generative-model.md)  
**Next:** [Section 1.6 - Core Probability Theory](./section-06-core-probability-theory.md)



