# Section 2.2: What Is a Neural Network?

> **Source inheritance:** Foster, Ch. 2 — "What Is a Neural Network?" / "Deep Neural Networks"  
> **Enhanced with:** Forward pass intuition, feature hierarchy, and connections to encoder/decoder stacks  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

A [neural network](../../GLOSSARY.md#neural-network) is a stack of layers that transforms input data through a series of weighted sums and nonlinear activations until it produces an output. Foster describes it as the workhorse of [deep learning](../../GLOSSARY.md#deep-learning): multiple hidden layers learn **high-level features** from unstructured data without hand-crafted engineering.

For generative modeling, the same building blocks appear everywhere:

- **VAE encoder** — compresses an image to a [latent variable](../../GLOSSARY.md#latent-variable) vector $z$
- **VAE decoder / GAN generator** — expands $z$ back to pixels
- **GAN discriminator** — classifies real vs fake
- **Diffusion U-Net** — predicts noise at each timestep

Understanding layers, forward passes, and feature hierarchies is prerequisite vocabulary for every chapter ahead.

> **Readable form:** A neural network is a differentiable function composer. Each layer refines the representation until the output matches your target — class probabilities, reconstructed pixels, or predicted noise.

---

## Anatomy of a Network

Foster's smiling-face example (Figure 2-2 in the book) walks through a concrete hierarchy:

1. **Unit A** — receives one channel of one pixel (raw, low-level)
2. **Unit B** — combines inputs to detect an **edge**
3. **Unit C** — combines edges to detect **teeth**
4. **Unit D** — combines high-level parts to detect a **smile**

No human told Unit C to look for teeth. The hierarchy emerges from **training** — minimizing prediction error via backpropagation.

| Layer type | Role in hierarchy | Generative analogue |
|------------|-------------------|---------------------|
| Early layers | Edges, textures | Encoder downsampling |
| Middle layers | Parts, patterns | Bottleneck / latent code |
| Late layers | Whole-object semantics | Decoder upsampling |

**Hidden layers** sit between input and output. "Deep" means many such layers — ResNet has 152; diffusion U-Nets stack dozens of residual blocks.

---

## The Forward Pass

For a single unit in a [dense layer](../../GLOSSARY.md#dense-layer):

$$
a = \sigma\left(\sum_{j} w_j x_j + b\right)
$$
> **Readable form:** Activation equals nonlinear function sigma applied to weighted sum of inputs plus bias.

where:

- $x_j$ — inputs from the previous layer
- $w_j$ — learned weights
- $b$ — bias (lets output be nonzero even when inputs are zero)
- $\sigma$ — activation function (ReLU, sigmoid, etc.)

A **forward pass** feeds input through every layer sequentially until the output layer produces predictions. Training runs millions of forward passes; generation (sampling from a VAE decoder or GAN generator) is also forward-only inference.

---

## Multilayer Perceptrons (MLPs)

When every layer is **fully connected** (each unit connects to every unit in the previous layer), the network is a **multilayer perceptron** (MLP).

MLPs flatten images to vectors — workable for CIFAR-10 baselines, but they discard spatial structure. That is why image [generative models](../../GLOSSARY.md#generative-model) use **convolutional** layers (Section 2.6) instead.

Still, MLPs teach the training loop (Section 2.4) and the Functional API (Section 2.3) without conv complexity. Foster trains an MLP on CIFAR-10 as a discriminative warm-up before generative chapters.

---

## Learning High-Level Features

The critical property Foster emphasizes: **no feature engineering required**. Traditional ML needs you to hand-design edges, histograms, or SIFT descriptors. Neural networks discover useful features by gradient descent alone.

For generative models this is doubly important:

- A VAE encoder must learn what to **compress** into 2D (or 128D) latent space
- A GAN generator must learn which pixel patterns fool the discriminator
- A diffusion model must learn which noise patterns correspond to petals vs stems

All of this is feature learning at scale.

---

## Training in One Paragraph

1. Initialize weights to small random values
2. Forward pass: compute predictions
3. Compute **loss** vs ground truth
4. **Backpropagation**: propagate error gradients backward
5. **Optimizer** updates weights to reduce loss
6. Repeat over batches and [epochs](../../GLOSSARY.md#epoch)

Foster's CIFAR-10 MLP improves from ~34% to ~52% accuracy in 10 epochs — far above random 10%, proving the network learned something, even with a naive architecture.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Minimal 2-hidden-layer MLP on flattened 32x32x3 images
inputs = layers.Input(shape=(32, 32, 3))
x = layers.Flatten()(inputs)
x = layers.Dense(200, activation="relu")(x)
x = layers.Dense(150, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)

mlp = models.Model(inputs, outputs)
mlp.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
mlp.summary()
# Train with mlp.fit(x_train, y_train, epochs=10, batch_size=32)
```

The same `Model` → `compile` → `fit` pattern trains autoencoders, GANs (with custom train steps), and diffusion U-Nets.

---

## Activations: ReLU, Sigmoid, Softmax

| Activation | Formula / behavior | Typical use |
|------------|-------------------|-------------|
| **ReLU** | $\max(0, x)$ | Hidden layers — stable, fast |
| **LeakyReLU** | Small slope for $x < 0$ | GAN discriminators — avoids dead units |
| **Sigmoid** | $\sigma(x) \in (0,1)$ | Binary output, VAE decoder pixels |
| **Softmax** | Normalized exponentials | Multi-class classification head |
| **tanh** | $\tanh(x) \in (-1,1)$ | GAN generator output |

Softmax for $J$ units:

$$
y_i = \frac{e^{x_i}}{\sum_{j=1}^{J} e^{x_j}}
$$
> **Readable form:** Softmax exponentiates each logit and divides by the sum — outputs are positive and sum to 1.

---

## Parameter Counting

Foster insists you understand parameter arithmetic. For a Dense layer with $n_{\text{in}}$ inputs and $n_{\text{out}}$ units:

$$
\text{params} = n_{\text{out}} \times (n_{\text{in}} + 1)
$$
> **Readable form:** Each output unit has one weight per input plus one bias — hence the +1.

Example: Dense(200) after Flatten(3072) → $200 \times (3072 + 1) = 614{,}600$ parameters. Large parameter counts mean slower training and higher overfitting risk — watch `model.summary()` before scaling to CelebA.

---

## Generative Preview: Encoder as Neural Network

In Chapter 03, the VAE encoder is a Conv2D stack ending in `Dense(2)` — same forward-pass mechanics, different objective (reconstruction instead of classification):

```
Image → Conv2D → Conv2D → Conv2D → Flatten → Dense(2) → z
```

The decoder inverts the path with `Conv2DTranspose`. You are not learning a new paradigm — only a new loss and latent sampling strategy.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Linear activations only | Network collapses to linear model | Use ReLU (or similar) between hidden layers |
| MLP on large images | Millions of parameters, poor accuracy | Switch to CNNs |
| Ignoring `model.summary()` | Surprise OOM errors | Check param counts before training |
| Confusing train vs inference | Dropout/BatchNorm behave wrong | Pass `training=False` when generating |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Neural network** | Stacked layers of weighted nonlinear transformations |
| **Forward pass** | Input → hidden layers → output (inference direction) |
| **Backpropagation** | Algorithm computing gradients of loss w.r.t. weights |
| **Hidden layer** | Intermediate layer between input and output |
| **MLP** | Fully connected multilayer perceptron |
| **Activation function** | Nonlinearity applied after weighted sum (ReLU, sigmoid, etc.) |

---

## Historical Note

Foster traces deep neural networks from Rosenblatt's perceptron through the modern resurgence driven by GPU compute and large datasets. The smiling-face MLP in Chapter 2 is deliberately small — the same layer types scale to ResNet-152 (image recognition) and to the encoder stacks you will build for CelebA VAEs and DCGAN discriminators. The constant across scales is the forward pass: weighted sums, nonlinear activations, gradient-based learning.

---

## Reflection Questions

1. Why can a deep network learn "teeth" without being told to look for teeth?
2. How many parameters does a Dense(150) layer have after a 200-unit hidden layer?
3. Why do generative image models prefer convolutions over flatten + Dense?
4. What is the difference between a forward pass during training and during sampling from a trained decoder?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 2 — Deep Neural Networks. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. Ch. 6 — Deep Feedforward Networks. [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
- Foster's codebase: `notebooks/02_deeplearning/01_mlp/mlp.ipynb`

---

**Previous:** [Section 2.1 — Data for Deep Learning](./section-01-data-for-deep-learning.md)  
**Next:** [Section 2.3 — TensorFlow and Keras](./section-03-tensorflow-and-keras.md)



