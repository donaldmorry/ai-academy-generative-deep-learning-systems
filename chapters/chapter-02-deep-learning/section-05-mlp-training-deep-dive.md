# Section 2.5: MLP Training Deep Dive

> **Source inheritance:** Foster, Ch. 2 — loss functions, optimizers, training dynamics  
> **Enhanced with:** Gradient descent intuition, callback patterns, and diagnostics that transfer to GAN/diffusion training  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Building an MLP takes twenty lines; **training** it well takes judgment. Foster dedicates significant Chapter 2 space to **loss functions**, **optimizers**, batching, and the `fit` loop because these choices recur in every [generative model](../../GLOSSARY.md#generative-model) you will build.

A VAE minimizes reconstruction error plus KL divergence. A GAN alternates two losses. A diffusion model minimizes noise prediction MSE. All rely on the same machinery: compute a scalar loss, backpropagate gradients, update weights with an optimizer.

> **Readable form:** Training is minimize loss with respect to weights. Everything else — architecture, data, schedules — serves that loop.

---

## The Training Loop in Detail

At each training **step**:

1. Sample a **batch** of inputs $\mathbf{X}$ and targets $\mathbf{Y}$
2. **Forward pass:** $\hat{\mathbf{Y}} = f_\theta(\mathbf{X})$
3. **Loss:** $\mathcal{L} = \text{Loss}(\mathbf{Y}, \hat{\mathbf{Y}})$
4. **Backward pass:** compute $\nabla_\theta \mathcal{L}$
5. **Optimizer step:** $\theta \leftarrow \theta - \eta \cdot \text{update rule}$

Repeat for all batches in an [epoch](../../GLOSSARY.md#epoch), then repeat epochs until convergence or early stopping.

```python
import tensorflow as tf
from tensorflow.keras import callbacks

cbs = [
    callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
    callbacks.ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_loss"),
    callbacks.TensorBoard(log_dir="./logs/mlp"),
]

history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(x_test, y_test),
    callbacks=cbs,
)
```

Callbacks automate decisions you would otherwise make by staring at loss curves.

---

## Loss Functions: Three Workhorses

### Mean Squared Error (regression)

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$
> **Readable form:** Average squared difference between prediction and target.

Used for continuous outputs. VAE reconstruction with RMSE/MSE produces slightly sharper but sometimes pixelated images (Foster, Ch. 3).

### Categorical cross-entropy (multi-class)

$$
\mathcal{L} = -\sum_{i=1}^{C} y_i \log p_i
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

One-hot $y$, softmax probabilities $p$. CIFAR-10 MLP uses this.

### Binary cross-entropy (binary / multi-label)

$$
\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log p_i + (1-y_i)\log(1-p_i)\right]
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

**GAN discriminators** predict real vs fake with sigmoid output + BCE. **VAE decoders** on pixels often use BCE per pixel (treats each pixel as independent Bernoulli).

| Task | Output | Loss |
|------|--------|------|
| 10-class image | softmax(10) | categorical_crossentropy |
| Real vs fake | sigmoid(1) | binary_crossentropy |
| Reconstructed pixels | sigmoid per pixel | binary_crossentropy |
| Predicted noise | linear | mse |

---

## Optimizers

**SGD** — vanilla gradient descent with learning rate $\eta$:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}
$$
> **Readable form:** This derivative tells how the loss changes when the parameter changes; backprop uses it to decide the update direction.

**Adam** (Adaptive Moment Estimation) — Foster's default. Maintains per-parameter learning rates using first and second moment estimates of gradients. Robust default for most Keras models.

```python
from tensorflow.keras import optimizers

# Foster's CIFAR MLP setting
adam = optimizers.Adam(learning_rate=0.0005)

# Common generative settings
adam_gan = optimizers.Adam(learning_rate=2e-4, beta_1=0.5)  # DCGAN recommendation
```

**Learning rate tradeoff:**

| LR too high | LR too low |
|-------------|------------|
| Loss spikes, NaN weights | Painfully slow convergence |
| Training unstable | May stall in poor local minima |

GANs are notoriously sensitive — Chapter 04 discusses separate LRs for generator and discriminator.

---

## Batch Size and Gradient Noise

**Small batch (32):** noisy gradients, more updates per epoch, regularizing effect.  
**Large batch (256):** smoother gradients, faster per-step linear algebra on GPU, may need LR warmup.

Foster recommends batch sizes between 32 and 256. Modern practice sometimes **increases batch size** during training for stability.

For generative models:

- DCGAN: batch 128 (Foster bricks dataset)
- VAE Fashion-MNIST: batch 100
- Diffusion: often 64–128 depending on image size and GPU memory

---

## Overfitting Signals

| Signal | Interpretation |
|--------|----------------|
| Train loss ↓, val loss ↑ | Classic overfitting |
| Train acc ↑, val acc flat | Memorizing training set |
| Both losses plateau | Converged or LR too small |
| Loss = NaN | LR too high, bad loss pairing, bad data scale |

**Remedies** (Section 2.7): [dropout](../../GLOSSARY.md#dropout), batch normalization, data augmentation, [early stopping](../../GLOSSARY.md#early-stopping), more data.

Generative overfitting looks different: **mode collapse** in GANs (generator outputs one image), or memorization in small VAE datasets (perfect reconstructions, poor samples).

---

## Monitoring Training

```python
import matplotlib.pyplot as plt

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"], label="train")
    ax1.plot(history.history["val_loss"], label="val")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(history.history["accuracy"], label="train")
    ax2.plot(history.history["val_accuracy"], label="val")
    ax2.set_title("Accuracy"); ax2.legend()
    plt.show()
```

Log **TensorBoard** scalars during GAN training — discriminator loss and generator loss oscillate; neither alone tells the full story.

---

## Custom Loss Preview (Generative)

VAE total loss (Chapter 03):

$$
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{KL}}
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

GAN generator loss — fool the discriminator:

$$
\mathcal{L}_G = -\mathbb{E}_z[\log D(G(z))]
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

Diffusion simple objective (Chapter 08):

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$
> **Readable form:** VAE = reconstruction + regularize latent; GAN generator = make discriminator think fakes are real; diffusion = predict the noise that was added.

Keras `model.compile(loss=custom_fn)` or override `train_step` for multi-term losses.

---

## train_step vs fit

Standard `fit` assumes one model, one loss, one optimizer. **GANs** need:

```python
@tf.function
def train_step(real_images):
    # 1) Train discriminator on real + fake
    # 2) Train generator to fool discriminator
    # Freeze appropriate variables each sub-step
    ...
```

Chapter 04's `DCGAN` subclass implements this. Understanding `fit` first makes `train_step` comprehensible.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Wrong loss for activation | NaN or stuck loss | Match sigmoid↔BCE, softmax↔CE |
| No callbacks | Train too long, overfit | EarlyStopping on val_loss |
| Ignoring val metrics | False confidence | Always plot train vs val |
| Same LR for G and D | Mode collapse / instability | Separate optimizers (GAN) |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Loss function** | Scalar measuring prediction error — what training minimizes |
| **Optimizer** | Algorithm updating weights from gradients (Adam, SGD, RMSprop) |
| **Learning rate** | Step size for weight updates |
| **Batch** | Subset of data per gradient computation |
| **Backpropagation** | Efficient algorithm for computing loss gradients |
| **Early stopping** | Halt training when validation metric stops improving |

---

## Reflection Questions

1. When would you choose MSE over binary cross-entropy for image reconstruction?
2. Why does Adam adapt per-parameter learning rates?
3. What does it mean if training accuracy is 95% but validation accuracy is 60%?
4. Why can't a standard GAN use a single `model.fit` call without customization?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 2 — training, loss, optimizers. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
- TensorFlow callbacks: [https://www.tensorflow.org/api_docs/python/tf/keras/callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks)

---

**Previous:** [Section 2.4 — Multilayer Perceptron](./section-04-multilayer-perceptron.md)  
**Next:** [Section 2.6 — Convolutional Layers](./section-06-convolutional-layers.md)



