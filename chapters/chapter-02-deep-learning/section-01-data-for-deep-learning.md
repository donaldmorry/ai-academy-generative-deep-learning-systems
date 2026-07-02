# Section 2.1: Data for Deep Learning

> **Source inheritance:** Foster, Ch. 2 — "Data for Deep Learning"  
> **Enhanced with:** Structured vs unstructured data, tensor pipelines, and preprocessing patterns reused in VAEs, GANs, and diffusion  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Before you train a [variational autoencoder](../../GLOSSARY.md#variational-autoencoder-vae), a [GAN](../../GLOSSARY.md#gan-generative-adversarial-network), or a [diffusion model](../../GLOSSARY.md#diffusion-model), you must answer a mundane but critical question: **what shape is your data, and how do you feed it to TensorFlow?**

Foster opens Chapter 2 with a distinction that shapes the entire field of [generative modeling](../../GLOSSARY.md#generative-model):

| Data type | Example | Typical ML input |
|-----------|---------|------------------|
| **Structured** | Age, income, website visits | Columns of features per row |
| **Unstructured** | Images, audio, text | Tensors — no natural feature columns |

Logistic regression on raw pixels fails because pixel 234 being brown tells you almost nothing about whether the image shows a house or a dog. Move the chimney one pixel left and the informative pixels change entirely — yet the label stays the same. This **spatial dependence** destroys the assumption that each input feature carries independent information.

[Deep learning](../../GLOSSARY.md#deep-learning) solves this by learning hierarchical features automatically. That is why generative AI focuses on images and text rather than spreadsheets — and why every model in this course starts with careful data preparation.

> **Readable form:** Generative deep learning mostly generates unstructured data. You must normalize pixels, build efficient pipelines, and split data correctly before any encoder, generator, or U-Net sees a batch.

---

## Structured vs Unstructured Data

**Structured data** arrives as a table: each row is an observation, each column is a feature. A random forest can learn that high income + many site visits correlates with subscription.

**Unstructured data** has structure — spatial layout in images, temporal order in audio — but not in columns. Individual pixels, frequency bins, or characters are almost uninformative alone. Meaning lives in **combinations** learned by stacked [neural network](../../GLOSSARY.md#neural-network) layers.

For generative modeling, we almost always target unstructured outputs:

- New images (DCGAN, diffusion)
- New text sequences (autoregressive LSTMs, later Transformers)
- New latent codes decoded to faces (VAE)

The preprocessing pipeline you build here — scale to $[0,1]$, batch tensors, train/val/test splits — is identical whether you are classifying CIFAR-10 or training a discriminator.

---

## Tensors: The Language of Image Data

An image dataset in Keras is a **tensor** — a multidimensional array. CIFAR-10 training images have shape:

$$
\text{shape} = (N,\; 32,\; 32,\; 3)
$$
> **Readable form:** N images, each 32×32 pixels, 3 RGB channels.

Indexing `x_train[54, 12, 13, 1]` returns the green channel value at row 12, column 13 of image 54. Foster uses this example to emphasize that deep learning code manipulates entire tensors at once via linear algebra — the source of GPU speedups.

**Key tensor concepts for this course:**

| Concept | Meaning | Generative use |
|---------|---------|----------------|
| **Batch dimension** | First axis; size chosen at runtime | Train on 32–256 samples per step |
| **Spatial dims** | Height, width | Conv layers preserve or downsample |
| **Channels** | RGB (3) or grayscale (1) | Match generator output activation |
| **dtype** | `float32` after scaling | Required for GPU training |

---

## Normalization: Why Scale to [0, 1]

Raw pixel values are integers in $\{0, 1, \ldots, 255\}$. [Neural networks](../../GLOSSARY.md#neural-network) train more stably when inputs have small magnitude:

$$
x_{\text{norm}} = \frac{x_{\text{raw}}}{255.0}
$$
> **Readable form:** Divide each pixel by 255 so values lie between 0 and 1.

For **GANs** (Chapter 04), Foster rescales to $[-1, 1]$ instead, to pair with `tanh` generator outputs:

$$
x_{\text{gan}} = \frac{x_{\text{raw}} - 127.5}{127.5}
$$
> **Readable form:** Center pixels at zero, scale to unit range — matches tanh output domain.

Always document which normalization your saved weights expect. Loading a model trained on $[-1,1]$ data and feeding $[0,1]$ inputs produces gray, washed-out generations.

---

## Train, Validation, and Test Splits

**Training set** — used to update weights via backpropagation.  
**Validation set** — held out during training; tune hyperparameters and detect [overfitting](../../GLOSSARY.md#overfitting).  
**Test set** — touched only once at the end for unbiased performance estimates.

For generative models, "performance" is trickier than accuracy (Chapter 01 discussed this), but the split discipline is the same:

1. Fit normalization statistics on **training data only**
2. Apply the same transform to val/test
3. Never tune on test

Data leakage through normalization — computing mean/std on the full dataset — inflates validation metrics and hides overfitting. This mistake appears constantly in Kaggle notebooks and production pipelines alike.

---

## TensorFlow Data Pipelines

Foster uses `tf.keras.utils.image_dataset_from_directory` for large image folders (GAN bricks, CelebA). Benefits:

- **Lazy loading** — batches read from disk on demand
- **Automatic resize** — e.g., $64 \times 64$ for DCGAN
- **Shuffling** — random batches each epoch
- **Prefetching** — overlaps CPU decode with GPU compute

```python
import tensorflow as tf
from tensorflow.keras import datasets, utils

# --- Classification example: CIFAR-10 ---
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
NUM_CLASSES = 10

# Normalize pixels to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels for categorical cross-entropy
y_train = utils.to_categorical(y_train, NUM_CLASSES)
y_test = utils.to_categorical(y_test, NUM_CLASSES)

print("Train images:", x_train.shape)   # (50000, 32, 32, 3)
print("Train labels:", y_train.shape)   # (50000, 10)

# --- Generative-style pipeline: unlabeled images from folder ---
# (Used in Chapter 04 for brick images)
train_ds = utils.image_dataset_from_directory(
    "/data/lego-brick-images/dataset/",
    labels=None,              # unsupervised — no class labels
    color_mode="grayscale",
    image_size=(64, 64),
    batch_size=128,
    shuffle=True,
    seed=42,
)

def preprocess_gan(img):
    """Scale to [-1, 1] for tanh generator outputs."""
    return (tf.cast(img, tf.float32) - 127.5) / 127.5

train_ds = train_ds.map(lambda x: preprocess_gan(x))
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
```

The classification pipeline teaches discriminative building blocks. The `image_dataset_from_directory` pattern is what you will reuse for GAN training — same API, different labels (`labels=None`).

---

## One-Hot Encoding for Classification

When the network outputs 10 class probabilities via [softmax](../../GLOSSARY.md#softmax), labels must be one-hot vectors. Class $i$ becomes a length-10 vector with 1 at index $i$:

$$
\mathbf{y}^{(i)} = [0, \ldots, 0, \underbrace{1}_{\text{position } i}, 0, \ldots, 0]
$$
> **Readable form:** Exactly one "hot" entry marks the true class; softmax outputs must sum to 1 across classes.

Generative models often skip labels entirely (unconditional VAE/GAN) or use **conditional** embeddings (Chapter 04, CGAN). The one-hot pattern returns when you add class conditioning.

---

## Why This Matters for Generative Models

Every architecture in Chapters 03–08 consumes preprocessed tensors:

| Model | Typical input shape | Normalization |
|-------|---------------------|---------------|
| VAE (Fashion-MNIST) | $(32, 32, 1)$ | $[0, 1]$ + pad to 32×32 |
| DCGAN (bricks) | $(64, 64, 1)$ | $[-1, 1]$ |
| PixelCNN | $(32, 32, 3)$ | $[0, 1]$ |
| Diffusion (flowers) | $(64, 64, 3)$ | $[-1, 1]$ or $[0, 1]$ per Foster |

Convolutional layers (Section 2.6), batch normalization (Section 2.7), and training loops (Sections 2.4–2.5) all assume correctly shaped, scaled tensors. Fix data first; debugging architecture on broken inputs wastes hours.

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Forgetting `astype('float32')` | Slow CPU math or type errors | Cast before divide |
| Normalizing with test statistics | Optimistic val metrics | Fit scaler on train only |
| Wrong channel order | Color-swapped outputs | Confirm RGB vs BGR vs grayscale |
| No `prefetch` on large datasets | GPU starvation | Add `dataset.prefetch(AUTOTUNE)` |
| Mismatched GAN scaling | Blurry or saturated fakes | Match preprocess to `tanh`/`sigmoid` |

---

## Connection to Prior Courses

| Concept | Where you learned it | Application here |
|---------|---------------------|------------------|
| Feature scaling | Course 1, preprocessing chapters | Pixel normalization |
| Train/test split | Course 1, ML fundamentals | Val split for early stopping |
| Tensors | Course 3, deep learning foundations | Image batch shapes |
| Unsupervised learning | Course 1, Section 1.3 | GAN/VAE use unlabeled images |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Structured data** | Tabular features in columns — age, price, counts |
| **Unstructured data** | Images, text, audio without feature columns |
| **Tensor** | Multidimensional array — the fundamental data structure in TensorFlow |
| **Normalization** | Rescaling inputs (e.g., pixels to $[0,1]$ or $[-1,1]$) for stable training |
| **One-hot encoding** | Vector label with a single 1 at the true class index |
| **tf.data pipeline** | Lazy, batched, prefetching input pipeline for large datasets |

---

## Reflection Questions

1. Why does logistic regression on raw pixels underperform on image classification?
2. When training a GAN, why does Foster rescale to $[-1, 1]$ instead of $[0, 1]$?
3. What goes wrong if you compute normalization constants on the full dataset including test images?
4. How does `image_dataset_from_directory` with `labels=None` differ from supervised loading?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). O'Reilly. Ch. 2 — Data for Deep Learning. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Foster's codebase: [https://github.com/davidADSP/GDL_code](https://github.com/davidADSP/GDL_code) — `notebooks/02_deeplearning/`
- TensorFlow `image_dataset_from_directory`: [https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory)
- Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images (CIFAR-10).

---

**Previous:** [Chapter 02 Overview](./README.md)  
**Next:** [Section 2.2 — What Is a Neural Network?](./section-02-what-is-a-neural-network.md)
