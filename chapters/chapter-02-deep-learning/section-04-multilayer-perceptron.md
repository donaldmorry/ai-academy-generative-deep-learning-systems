# Section 2.4: Multilayer Perceptron

> **Source inheritance:** Foster, Ch. 2 — "Multilayer Perceptron (MLP)"  
> **Enhanced with:** CIFAR-10 end-to-end pipeline, layer anatomy, and the compile-fit-evaluate pattern reused in every generative model  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Foster's first complete Keras project trains a **multilayer perceptron** (MLP) on CIFAR-10 — 60,000 tiny color photographs sorted into 10 classes. The MLP is **discriminative**: it predicts labels, it does not generate images. Yet the workflow you practice here — load data, build architecture, compile, fit, evaluate, predict — is the exact skeleton for training [autoencoders](../../GLOSSARY.md#autoencoder), [GANs](../../GLOSSARY.md#gan-generative-adversarial-network), and diffusion U-Nets in later chapters.

Think of this section as learning to drive in a parking lot before entering highway traffic. The car is the same; the destination changes.

> **Readable form:** Preprocess tensors, stack Dense layers, pick a loss, call fit. Every generative model in this course repeats those steps with different architectures and objectives.

---

## The CIFAR-10 Dataset

CIFAR-10 ships with Keras: 50,000 training and 10,000 test images at $32 \times 32$ RGB resolution.

| Class index | Class name |
|-------------|------------|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

Each pixel channel is an integer $0$–$255$. Neural nets prefer small magnitudes, so we scale:

$$
x_{\text{norm}} = \frac{x_{\text{raw}}}{255.0} \in [0, 1]
$$
> **Readable form:** Divide raw 8-bit pixel values by 255 so model inputs fall between 0 and 1.

Labels are integers $0$–$9$ but the output layer has 10 units with **softmax**, so we **one-hot encode**:

```python
import numpy as np
from tensorflow.keras import datasets, utils, layers, models, optimizers

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
NUM_CLASSES = 10

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = utils.to_categorical(y_train, NUM_CLASSES)
y_test = utils.to_categorical(y_test, NUM_CLASSES)

print(x_train.shape)  # (50000, 32, 32, 3)
print(y_train.shape)  # (50000, 10)
```

---

## Building the Architecture

Foster shows both Sequential and Functional APIs; we use **Functional API** (recommended for generative models ahead):

```python
input_layer = layers.Input(shape=(32, 32, 3))
x = layers.Flatten()(input_layer)
x = layers.Dense(units=200, activation="relu")(x)
x = layers.Dense(units=150, activation="relu")(x)
output_layer = layers.Dense(units=10, activation="softmax")(x)
model = models.Model(input_layer, output_layer)
model.summary()
```

### Layer-by-layer anatomy

| Layer | Output shape | Role |
|-------|--------------|------|
| `Input` | (None, 32, 32, 3) | Entry point; `None` = variable batch |
| `Flatten` | (None, 3072) | $32 \times 32 \times 3 = 3072$ vector |
| `Dense(200, relu)` | (None, 200) | First hidden representation |
| `Dense(150, relu)` | (None, 150) | Second hidden representation |
| `Dense(10, softmax)` | (None, 10) | Class probabilities |

**Parameter count** for `Dense(200)` after Flatten:

$$
200 \times (3072 + 1) = 614{,}600
$$
> **Readable form:** A dense layer with 200 units and 3072 inputs has one weight per input plus one bias for each unit.

The $+1$ is the bias per output unit. Total model: **646,260** trainable parameters.

---

## Input, Flatten, and Dense Layers

**Input layer** — declares expected shape per sample `(32, 32, 3)` without fixing batch size. You can pass 1 image or 1,000 in one call; TensorFlow vectorizes across the batch dimension on GPU.

**Flatten** — required because `Dense` expects 1D input. It discards spatial structure: pixel $(i,j)$ neighbors become unrelated indices in a 3072-vector. That is the main reason MLPs underperform CNNs on images — and why Chapter 03's VAE uses `Conv2D` instead.

**Dense layer** — each unit computes $\text{ReLU}(\mathbf{w}^\top \mathbf{x} + b)$ and connects to **every** unit in the previous layer. ReLU ($\max(0,x)$) between hidden layers prevents the network from collapsing to a single linear map.

---

## Activation Functions in This Model

**ReLU** (hidden layers):

$$
\text{ReLU}(x) = \max(0, x)
$$
> **Readable form:** ReLU passes positive values unchanged and clips negative values to zero.

**Softmax** (output layer) — converts 10 logits to a probability distribution:

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}
$$
> **Readable form:** Softmax exponentials each score and normalizes so all class probabilities sum to 1.

Activations can be inline (`activation='relu'`) or separate `layers.Activation('relu')` — identical effect.

---

## Compiling: Loss and Optimizer

```python
opt = optimizers.Adam(learning_rate=0.0005)
model.compile(
    loss="categorical_crossentropy",
    optimizer=opt,
    metrics=["accuracy"],
)
```

**Categorical cross-entropy** — correct pairing with softmax for single-label classification:

$$
\mathcal{L} = -\sum_{i=1}^{10} y_i \log p_i
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

where $y_i$ is 1 for the true class and 0 elsewhere.

**Adam optimizer** — adaptive learning rate; Foster uses `lr=5e-4`. Default Adam often works; learning rate is the first hyperparameter to tune if loss diverges.

| Problem type | Output activation | Loss |
|--------------|-----------------|------|
| Multi-class (one label) | softmax | categorical_crossentropy |
| Binary (one output) | sigmoid | binary_crossentropy |
| Regression | linear | mse |

Generative models swap this table: VAE uses reconstruction loss + KL; GAN uses binary cross-entropy on discriminator outputs; diffusion uses MSE on noise.

---

## Training with fit

```python
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    shuffle=True,
    validation_data=(x_test, y_test),
)
```

**Training loop mechanics:**

1. Weights initialize to small random values
2. Each **step** passes one **batch** (32 images) forward, computes loss, backpropagates
3. One **epoch** = every training image seen once ($50000/32 \approx 1563$ steps)
4. Repeat for 10 epochs

Foster's run: loss $1.84 \rightarrow 1.37$, accuracy $34\% \rightarrow 52\%$. Random guessing = 10%, so the MLP learned real signal despite ignoring spatial structure.

**Why not full-batch gradients?** Computing loss on all 50,000 images per step is expensive. Mini-batches (32–256) approximate the gradient cheaply.

---

## Evaluation and Prediction

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.1%}")  # ~49%

probs = model.predict(x_test[:8])
predicted_classes = np.argmax(probs, axis=1)
```

Test accuracy ~49% with train ~52% — slight overfitting, acceptable for a tutorial MLP. `predict` returns probabilities, not class indices; use `argmax` for labels.

For generative models, `predict` becomes `decoder(z)` or `generator(noise)` — forward pass only, no labels.

---

## Inspecting Training History

```python
import matplotlib.pyplot as plt

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss")
```

Rising val_loss while train_loss falls signals [overfitting](../../GLOSSARY.md#overfitting). Remedies (Section 2.7): dropout, batch norm, more data, early stopping.

---

## Bridge to Generative Models

| MLP concept | Chapter 03 VAE | Chapter 04 GAN |
|-------------|---------------|---------------|
| `Input(shape)` | `encoder_input (32,32,1)` | `latent vector (100,)` |
| Hidden stack | Conv2D encoder | Conv2D discriminator |
| Output + loss | Sigmoid pixels + BCE | Sigmoid real/fake + BCE |
| `fit(x, x)` | Autoencoder target = input | Custom `train_step` |
| `predict` | `decoder(z)` samples | `generator(z)` samples |

The MLP teaches **Keras mechanics**. Generative chapters teach **new objectives** on top of the same API.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Integer labels + softmax | Wrong shapes / poor training | `to_categorical` |
| Sigmoid + 10 outputs + BCE | Wrong loss semantics | Use softmax + categorical CE |
| No `astype('float32')` | Integer division bugs | Cast before `/ 255.0` |
| Ignoring `model.summary()` | Surprise OOM | Check params before scaling up |
| Training without val split | Overfitting unnoticed | `validation_data=` or `validation_split=` |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **MLP** | Multilayer perceptron — fully connected feedforward network |
| **Flatten** | Layer reshaping spatial dims to a 1D vector for Dense input |
| **Dense** | Fully connected layer — every unit connects to all prior units |
| **Categorical cross-entropy** | Classification loss for one-hot labels and softmax outputs |
| **Softmax** | Normalized exponential — outputs valid probability distribution |
| **batch_size** | Samples per gradient step; trades stability vs memory |
| **epoch** | One complete pass through the training dataset |

---

## Reflection Questions

1. Why is Flatten harmful for image classification compared to Conv2D?
2. Compute the parameter count for `Dense(150)` following `Dense(200)`.
3. What loss/activation pair would you use for binary cat-vs-dog classification with one output neuron?
4. How does the training loop change when the target equals the input (autoencoder)?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 2 — Multilayer Perceptron. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Foster's codebase: [https://github.com/davidADSP/GDL_code](https://github.com/davidADSP/GDL_code) — `notebooks/02_deeplearning/01_mlp/mlp.ipynb`
- Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images (CIFAR-10).

---

**Previous:** [Section 2.3 — TensorFlow and Keras](./section-03-tensorflow-and-keras.md)  
**Next:** [Section 2.5 — MLP Training Deep Dive](./section-05-mlp-training-deep-dive.md)


