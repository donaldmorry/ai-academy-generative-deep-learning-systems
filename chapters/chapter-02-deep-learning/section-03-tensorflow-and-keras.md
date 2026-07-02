# Section 2.3: TensorFlow and Keras

> **Source inheritance:** Foster, Ch. 2 — "TensorFlow and Keras"  
> **Enhanced with:** Functional API patterns, custom training steps preview for GANs, and production-ready workflows  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

[TensorFlow](../../GLOSSARY.md#tensorflow) is Google's open-source framework for tensor computation — automatic differentiation, GPU kernels, and distributed training. [Keras](../../GLOSSARY.md#keras) is the high-level API layered on top: build models by stacking `layers`, compile with optimizer + loss, train with `fit`.

Foster recommends **TensorFlow + Keras** for every example in *Generative Deep Learning*. The same stack trains MLPs today and billion-parameter diffusion models tomorrow. Mastering the Functional API now saves painful refactors when you wire encoder → decoder (VAE) or generator ↔ discriminator (GAN).

> **Readable form:** TensorFlow does the math; Keras describes the architecture. You focus on layers and losses; the framework handles gradients.

---

## TensorFlow: Tensors and Autodiff

A **tensor** generalizes scalars, vectors, and matrices to arbitrary dimensions. Image batches are 4D tensors $(N, H, W, C)$. Latent vectors are 2D $(N, d_z)$.

TensorFlow's core capabilities:

| Capability | Purpose |
|------------|---------|
| `tf.Tensor` operations | Vectorized math on GPU |
| `tf.GradientTape` | Record ops for automatic differentiation |
| `tf.data.Dataset` | Efficient input pipelines |
| SavedModel export | Deploy trained generators |

Generative training often needs custom loops — GANs alternate discriminator and generator updates; diffusion samples noise timesteps randomly. `GradientTape` and subclassed `models.Model` (Chapter 04) build on the same foundation as `model.fit`.

---

## Keras: Three Ways to Build Models

### 1. Sequential API

Linear stack — one layer after another:

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(200, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
```

Fast for simple MLPs. **Insufficient** for VAEs (encoder branches), GANs (two networks), or U-Nets (skip connections).

### 2. Functional API (recommended)

Foster explicitly recommends the Functional API even for beginners:

```python
input_layer = layers.Input(shape=(32, 32, 3))
x = layers.Flatten()(input_layer)
x = layers.Dense(200, activation="relu")(x)
output_layer = layers.Dense(10, activation="softmax")(x)
model = models.Model(input_layer, output_layer)
```

**Why it matters for generative models:**

```python
# VAE pattern (Chapter 03)
encoder_output = encoder(encoder_input)
autoencoder_output = decoder(encoder_output)
autoencoder = models.Model(encoder_input, autoencoder_output)

# GAN pattern (Chapter 04) — generator inside discriminator training
fake_images = generator(latent_vector)
discriminator_output = discriminator(fake_images)
```

Branching, shared layers, and multi-input/multi-output models require Functional API.

### 3. Model Subclassing

Override `call()` and `train_step()` for full control:

```python
class DCGAN(tf.keras.Model):
  def train_step(self, data):
      # Alternate D and G updates — Chapter 04
      ...
```

Used for GANs, diffusion with custom noise sampling, and any logic `fit` cannot express.

---

## Eager Execution

TensorFlow 2.x runs eagerly by default — operations execute immediately like NumPy. You can debug with `print(tensor.numpy())` mid-forward-pass. `@tf.function` optionally compiles hot paths for speed.

For learning, eager mode is ideal. For production GAN training at scale, decorating `train_step` with `@tf.function` accelerates loops.

---

## Inspecting Models

```python
model.summary()      # layer shapes, parameter counts
model.get_layer("encoder_output")  # access by name
tf.keras.utils.plot_model(model, show_shapes=True)  # diagram
```

Foster's Table 2-1 shows how `None` in the batch dimension means variable batch size — pass 1 or 1000 images without redefining the graph.

**Parameter sanity check:** If a single Dense layer shows millions of parameters, consider fewer units or switch to convolutions.

---

## Compiling: Optimizer + Loss + Metrics

```python
from tensorflow.keras import optimizers

opt = optimizers.Adam(learning_rate=0.0005)
model.compile(
    loss="categorical_crossentropy",
    optimizer=opt,
    metrics=["accuracy"],
)
```

| Component | Role |
|-----------|------|
| **Loss** | Scalar error per batch — what gradients minimize |
| **Optimizer** | Weight update rule (Adam, RMSprop, SGD) |
| **Metrics** | Logged diagnostics (accuracy, not used for gradients) |

Generative models swap losses:

- VAE: reconstruction + [KL divergence](../../GLOSSARY.md#kl-divergence) ([ELBO](../../GLOSSARY.md#elbo-evidence-lower-bound))
- GAN: binary cross-entropy for D and G
- Diffusion: MSE between predicted and true noise

---

## Training with `fit`

```python
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    shuffle=True,
    validation_data=(x_val, y_val),
)
```

**Batch size** — typically 32–256; larger batches stabilize gradients but need more memory. Foster notes modern practice of increasing batch size during training.

**Epochs** — one full pass through training data. Generative models often need hundreds of epochs (GANs) or hundreds of thousands of steps (diffusion).

**Callbacks** — `ModelCheckpoint`, `EarlyStopping`, `TensorBoard`:

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best.weights.h5", save_best_only=True),
]
model.fit(..., callbacks=callbacks)
```

---

## Custom Training Loop Preview

GANs cannot use vanilla `fit` on a single model — you must freeze one network while training the other. Pattern:

```python
with tf.GradientTape() as tape:
    z = tf.random.normal((batch_size, latent_dim))
    fake = generator(z, training=True)
    d_fake = discriminator(fake, training=False)  # freeze D
    g_loss = loss_fn(tf.ones_like(d_fake), d_fake)

grads = tape.gradient(g_loss, generator.trainable_variables)
g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
```

Chapter 04 implements the full `DCGAN` class. The section here: Keras gives you escape hatches when `compile` + `fit` are not enough.

---

## TensorFlow Probability (optional)

Chapter 01 used `tensorflow_probability` for a Gaussian generative model. TFP integrates with Keras for:

- Mixture distributions (Chapter 05, PixelCNN++)
- Normalizing flows (Chapter 06)
- Reparameterized sampling in VAEs

```python
import tensorflow_probability as tfp
tfd = tfp.distributions

# Latent prior for VAE
prior = tfd.MultivariateNormalDiag(loc=tf.zeros(2), scale_diag=tf.ones(2))
z = prior.sample(16)  # batch of latent codes
```

---

## Environment Setup

Foster's repository (`GDL_code`) expects:

- Python 3.8+
- `tensorflow>=2.10`
- Jupyter for notebooks under `notebooks/02_deeplearning/`

Verify GPU:

```python
print(tf.config.list_physical_devices("GPU"))
```

Generative training without GPU is possible on MNIST-scale data but impractical for CelebA, DCGAN, or diffusion.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Sequential API for U-Net | Cannot add skip connections | Use Functional API |
| Wrong import path | Deprecated warnings | `from tensorflow.keras import ...` |
| Forgetting `training` flag | BN/Dropout wrong at sample time | `model(x, training=False)` |
| Not saving weights | Lost after crash | `ModelCheckpoint` callback |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **TensorFlow** | Low-level ML framework for tensor ops and autodiff |
| **Keras** | High-level neural network API on TensorFlow |
| **Functional API** | Build models by connecting layer outputs to inputs |
| **Model subclassing** | Custom `Model` with overridden `train_step` |
| **compile** | Attach optimizer, loss, and metrics to a model |
| **fit** | Train using data passed to the model |

---

## Reflection Questions

1. Why does Foster recommend Functional API over Sequential even for linear MLPs?
2. What Keras method shows parameter counts per layer?
3. When will you need `train_step` instead of `fit` in this course?
4. How does `validation_data` in `fit` help detect overfitting?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 2 — TensorFlow and Keras. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- TensorFlow Keras guide: [https://www.tensorflow.org/guide/keras](https://www.tensorflow.org/guide/keras)
- Chollet, F. *Deep Learning with Python* (2nd ed.) — Keras creator's tutorial
- Foster's codebase: [https://github.com/davidADSP/GDL_code](https://github.com/davidADSP/GDL_code)

---

**Previous:** [Section 2.2 — What Is a Neural Network?](./section-02-what-is-a-neural-network.md)  
**Next:** [Section 2.4 — Multilayer Perceptron](./section-04-multilayer-perceptron.md)
