# Section 2.6: Convolutional Layers

> **Source inheritance:** Foster, Ch. 2 — "Convolutional Layers"  
> **Enhanced with:** Output-shape arithmetic, filter depth rules, and spatial hierarchy that carry into DCGAN and diffusion U-Nets  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Why Flatten Fails on Images

Foster's CIFAR-10 MLP ([Section 2.4](./section-04-multilayer-perceptron.md)) reaches roughly 49% test accuracy. The bottleneck is architectural, not optimizer choice: the first step **flattens** each $32 \times 32 \times 3$ image into a 3,072-dimensional vector and feeds it to a [Dense](../../GLOSSARY.md#feature) layer.

That destroys **spatial structure**. Pixel $(5, 12)$ in the top-left corner of a cat's ear is no more related to pixel $(6, 12)$ than to pixel $(28, 3)$ in the background. Yet neighbors in an image are highly correlated — edges, textures, and object parts form local patterns.

Classical models on raw pixels (logistic regression, random forests) fail for the same reason: they treat each pixel as an independent [feature](../../GLOSSARY.md#feature). A [deep learning](../../GLOSSARY.md#deep-learning) network can **learn** informative features from raw pixels — but only if the architecture respects spatial layout.

> **Readable form:** Images are grids of correlated pixels. Flattening throws away the grid. Convolutional layers keep it.

**Convolutional layers** solve this. They slide small learned filters across the image, building a **spatial hierarchy** of feature maps: edges → textures → object parts → whole objects. Every generative image model in this course — DCGAN, VAE encoder-decoders, diffusion U-Nets — is built from these same blocks.

---

## What Is a Convolution?

In deep learning, a **convolution** (more precisely, a cross-correlation in most frameworks) multiplies a small **filter** (also called a **kernel**) pixelwise with a patch of the input image and sums the results.

For a grayscale patch $\mathbf{X}$ and filter $\mathbf{K}$ both of size $3 \times 3$:

$$
(\mathbf{X} * \mathbf{K})_{i,j} = \sum_{m=0}^{2}\sum_{n=0}^{2} X_{i+m,\,j+n} \cdot K_{m,n}
$$
> **Readable form:** At each position, multiply overlapping pixels with filter weights and add them up.

The output is **large and positive** when the patch closely matches the filter, **near zero** when it does not, and **negative** when the patch is the inverse of the filter.

Foster's Figures 2-10 and 2-11 illustrate this with edge detectors: one filter highlights horizontal edges, another vertical edges. The book's notebook `notebooks/02_deeplearning/02_cnn/convolutions.ipynb` walks through the arithmetic by hand — worth running once.

```python
import numpy as np

# Toy 5x5 grayscale image (0=black, 1=white)
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=np.float32)

# Vertical edge filter: bright on left, dark on right
vertical_edge = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
], dtype=np.float32)

def conv2d_single(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = image[i:i+kh, j:j+kw]
            out[i, j] = np.sum(patch * kernel)
    return out

feature_map = conv2d_single(image, vertical_edge)
print(feature_map)  # strong responses along left/right edges of the white block
```

> **Readable form:** Slide the filter across the image. Each position records how strongly that local pattern appears.

---

## Filters Are Learned Weights

A **convolutional layer** is a collection of filters. The values in each filter are **weights learned by backpropagation** — not hand-designed edge detectors.

Initially random, filters gradually adapt to detect useful patterns: edges, color blobs, fur texture, wheel shapes. Each filter produces one **feature map** — a 2-D array showing *where* that pattern appears in the input.

| Concept | Role |
|---------|------|
| **Filter / kernel** | Small weight tensor slid across the input |
| **Feature map** | 2-D output from one filter — "activation map" for one pattern |
| **Depth / channels** | Number of filters = number of output channels |

In Keras, `Conv2D` applies this operation to tensors with two spatial dimensions (height, width):

```python
from tensorflow.keras import layers, models

input_layer = layers.Input(shape=(64, 64, 1))  # grayscale
conv_layer = layers.Conv2D(
    filters=2,
    kernel_size=(3, 3),
    strides=1,
    padding="same",
)(input_layer)

model = models.Model(input_layer, conv_layer)
model.summary()
# Output: (None, 64, 64, 2) — two feature maps, same spatial size
```

> **Readable form:** Two filters on a 64×64 grayscale image produce a 64×64×2 tensor.

---

## Stride: How Far the Filter Steps

The **stride** is the step size when sliding the filter across the input.

| `strides` | Effect |
|-----------|--------|
| `1` | Filter moves one pixel at a time — preserves spatial resolution (with `padding="same"`) |
| `2` | Filter moves two pixels — **halves** height and width |

Increasing stride is a common way to **downsample** without a separate pooling layer: spatial dimensions shrink while the number of channels (filters) grows. Foster's CIFAR-10 CNN uses `strides=2` on alternating conv layers to progressively compress $32 \times 32 \to 16 \times 16 \to 8 \times 8$.

> **Readable form:** Stride 2 means the output grid is half as wide and half as tall as the input.

---

## Padding: Keeping Size Predictable

**Padding** adds zeros around the image border so filters can reach edge pixels.

With `padding="same"` and `strides=1`, output height and width equal input height and width. Foster's Figure 2-12 shows a $3 \times 3$ kernel on a $5 \times 5$ image: with padding, the kernel fits five times in each direction (output $5 \times 5$); without padding, only three times (output $3 \times 3$).

The general output-shape formula with `padding="same"`:

H_{\text{out}} = \left\lceil \frac{H_{\text{in}}}{\text{stride}} \right\rceil, \quad
W_{\text{out}} = \left\lceil \frac{W_{\text{in}}}{\text{stride}} \right\rceil
> **Readable form:** Output spatial size equals input size divided by stride, rounded up.

`padding="same"` is the default choice when you want to stack many conv layers without constantly recalculating shrinking dimensions.

---

## Filter Depth Must Match Input Channels

For RGB images, each filter has depth 3 (one weight slice per color channel). The filter shape is `(kernel_h, kernel_w, input_channels)`.

**Rule (Foster emphasizes this):** the depth of every filter in a layer equals the number of channels output by the preceding layer.

| Layer | Input shape | `kernel_size` | `filters` | Filter tensor shape | Params per filter |
|-------|-------------|---------------|-----------|---------------------|-------------------|
| Conv2D #1 | $(32, 32, 3)$ | $(4, 4)$ | 10 | $4 \times 4 \times 3$ | $48 + 1$ bias |
| Conv2D #2 | $(16, 16, 10)$ | $(3, 3)$ | 20 | $3 \times 3 \times 10$ | $90 + 1$ bias |

Total parameters in Conv2D layer:

$$
\text{params} = (\text{kernel\_h} \times \text{kernel\_w} \times \text{in\_channels} + 1) \times \text{filters}
$$
> **Readable form:** Multiply kernel area by input channels, add one bias per filter, multiply by number of filters.

---

## Stacking Layers: A Spatial Hierarchy

The output of `Conv2D` is a 4-D tensor: `(batch_size, height, width, filters)`. You can stack `Conv2D` layers directly — no `Flatten` required until you need a classification head.

Foster's minimal CIFAR-10 CNN (Example 2-13):

```python
from tensorflow.keras import layers, models

input_layer = layers.Input(shape=(32, 32, 3))
x = layers.Conv2D(10, (4, 4), strides=2, padding="same")(input_layer)
x = layers.Conv2D(20, (3, 3), strides=2, padding="same")(x)
x = layers.Flatten()(x)
output_layer = layers.Dense(10, activation="softmax")(x)
model = models.Model(input_layer, output_layer)
model.summary()
```

### Tensor shape walkthrough

| Layer | Output shape | Params | Notes |
|-------|-------------|--------|-------|
| Input | `(None, 32, 32, 3)` | 0 | `None` = any batch size |
| Conv2D | `(None, 16, 16, 10)` | 490 | stride 2 halves spatial dims; 10 filters |
| Conv2D | `(None, 8, 8, 20)` | 1,820 | depth 10 matches prior channels |
| Flatten | `(None, 1280)` | 0 | $8 \times 8 \times 20 = 1280$ |
| Dense | `(None, 10)` | 12,810 | softmax over 10 classes |

**Spatial hierarchy intuition:**

1. **Early layers** — small receptive fields, detect local edges and color contrasts
2. **Middle layers** — combine low-level features into textures and parts
3. **Late layers** — encode object-level patterns before `Flatten` + `Dense`

This hierarchy is exactly what generative models exploit: VAE encoders compress images through conv blocks; DCGAN generators **reverse** the hierarchy with `Conv2DTranspose`; diffusion U-Nets preserve spatial resolution with skip connections between encoder and decoder conv stacks.

---

## Inspecting Shapes in Practice

Always call `model.summary()` and trace shapes by hand. Foster recommends proving you understand each layer before moving on — this skill becomes critical when you design custom generator architectures in Chapter 04.

```python
import tensorflow as tf

# Quick shape probe without full training
sample = tf.zeros((1, 32, 32, 3))
for i, layer in enumerate(model.layers):
    sample = layer(sample)
    print(f"{layer.name:20s} -> {sample.shape}")
```

Common shape errors:

| Error | Cause |
|-------|-------|
| `InvalidArgumentError` on forward pass | Channel mismatch between stacked conv layers |
| Unexpected tiny spatial dims | Forgot stride 2 halves resolution |
| OOM on GPU | Too many filters at full $32 \times 32$ resolution |

---

## Conv2D vs Dense: Parameter Efficiency

A `Dense(3072, 128)` layer on flattened CIFAR images needs $3072 \times 128 \approx 393\text{K}$ weights — every input pixel connects to every neuron.

A `Conv2D(32, (3, 3))` on $(32, 32, 3)$ input uses $(3 \times 3 \times 3 + 1) \times 32 = 896$ weights — the **same** $3 \times 3$ filter is reused at every spatial position (**weight sharing**).

| Property | Dense on flat pixels | Conv2D |
|----------|------------------------|--------|
| Spatial awareness | None | Built in |
| Parameters | $O(H \times W \times C \times \text{units})$ | $O(k^2 \times C_{\text{in}} \times \text{filters})$ |
| Translation handling | Must relearn pattern at each position | Detects pattern anywhere |

> **Readable form:** Conv layers share weights across space. One edge detector works for the whole image.

---

## Connection to Generative Models

| Conv2D usage | Where it appears |
|--------------|------------------|
| Encoder downsampling | VAE encoder (Chapter 03) — compress image to latent vector |
| Discriminator feature extraction | DCGAN critic (Chapter 04) |
| Noise prediction backbone | Diffusion U-Net (Chapter 08) |
| Autoregressive conditioning | PixelCNN masked conv layers (Chapter 05) |

The **transpose** of convolution — `Conv2DTranspose` — upsamples in generators. If you understand how `strides=2` shrinks spatial size in a conv layer, you understand how stride-2 transpose conv **grows** it back.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Wrong `input_shape` channels | Shape error on first layer | RGB = 3, grayscale = 1 |
| `Flatten` too early | Loses spatial structure | Keep conv blocks before `Flatten` |
| Ignoring stride effects | Surprise shape after 3 layers | Use output formula or `model.summary()` |
| Confusing filters and channels | Miscounted parameters | `filters` = output depth = number of feature maps |
| No activation after conv | Linear stacks only | Add ReLU/LeakyReLU (see [Section 2.7](./section-07-batch-normalization-and-dropout.md)) |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Filter / kernel** | Small learned weight tensor slid across the input |
| **Feature map** | 2-D spatial output from one filter showing where a pattern activates |
| **Stride** | Step size of the filter; stride 2 halves spatial dimensions |
| **Padding** | Zero-padding at borders; `"same"` preserves size when stride = 1 |
| **Channels** | Depth dimension — 3 for RGB input, grows with filter count |
| **Spatial hierarchy** | Stacked conv layers building from edges to complex patterns |
| **Weight sharing** | Same filter applied at every position — core efficiency of CNNs |

---

## Reflection Questions

1. A $28 \times 28$ grayscale image passes through `Conv2D(16, (3,3), strides=1, padding="same")`. What is the output shape?
2. Why does the second conv layer's filter depth equal the first layer's `filters` argument?
3. How would replacing `Flatten` + `Dense` with global average pooling change the architecture?
4. Where in a DCGAN generator would you expect `Conv2DTranspose` instead of `Conv2D`?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 2 — Convolutional Layers. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Foster's codebase: [https://github.com/davidADSP/GDL_code](https://github.com/davidADSP/GDL_code) — `notebooks/02_deeplearning/02_cnn/convolutions.ipynb`
- Dumoulin, V., & Visin, F. (2018). A Guide to Convolution Arithmetic for Deep Learning. [https://arxiv.org/abs/1603.07285](https://arxiv.org/abs/1603.07285)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. Ch. 9 — Convolutional Networks. [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

---

**Previous:** [Section 2.5 — MLP Training Deep Dive](./section-05-mlp-training-deep-dive.md)  
**Next:** [Section 2.7 — Batch Normalization & Dropout](./section-07-batch-normalization-and-dropout.md)

