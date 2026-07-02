# Section 2.8: Building and Training a CNN

> **Source inheritance:** Foster, Ch. 2 — "Building the CNN" and "Training and Evaluating the CNN"  
> **Enhanced with:** Full CIFAR-10 pipeline, accuracy jump from MLP to CNN, and the design philosophy that carries into generative architectures  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Putting It All Together

You now have three new layer types from Foster's toolkit:

| Layer | Section | Role |
|-------|--------|------|
| `Conv2D` | [2.6](./section-06-convolutional-layers.md) | Extract spatial features |
| `BatchNormalization` | [2.7](./section-07-batch-normalization-and-dropout.md) | Stabilize activations |
| `Dropout` | [2.7](./section-07-batch-normalization-and-dropout.md) | Regularize dense layers |

This section assembles them into a complete **convolutional neural network** on CIFAR-10, trains it with the same compile-fit-evaluate pattern from [Section 2.4](./section-04-multilayer-perceptron.md), and measures the improvement over the flattened MLP.

The result: **71.5% test accuracy** vs the MLP's **49.0%** — with **fewer total parameters** despite more layers. That counterintuitive win (more layers, fewer weights) is why understanding architecture matters before you touch any [generative model](../../GLOSSARY.md#generative-model).

> **Readable form:** Stack conv blocks with batch norm and leaky ReLU, add a dense head with dropout, train like the MLP. Accuracy jumps 22 points.

Foster's notebook: `notebooks/02_deeplearning/02_cnn/cnn.ipynb`.

---

## Data Preparation

Same CIFAR-10 pipeline as the MLP — scale pixels, one-hot labels:

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, utils, layers, models, optimizers

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = utils.to_categorical(y_train, NUM_CLASSES)
y_test = utils.to_categorical(y_test, NUM_CLASSES)

print(f"Train: {x_train.shape}, Test: {x_test.shape}")
# Train: (50000, 32, 32, 3), Test: (10000, 32, 32, 3)
```

**Critical difference from the MLP:** we do **not** flatten. Images enter the network as $(32, 32, 3)$ tensors so conv layers can exploit spatial structure.

---

## Architecture Design

Foster's CNN (Example 2-16) uses four conv blocks, a dense bottleneck, and a softmax head. We follow the **BAD** pattern from [Section 2.7](./section-07-batch-normalization-and-dropout.md): Batch norm → Activation (LeakyReLU) → Dropout (dense head only).

```python
def build_cnn():
    input_layer = layers.Input(INPUT_SHAPE)

    # Block 1: 32 filters, preserve spatial size
    x = layers.Conv2D(32, 3, strides=1, padding="same")(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Block 2: 32 filters, downsample to 16x16
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Block 3: 64 filters, preserve 16x16
    x = layers.Conv2D(64, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Block 4: 64 filters, downsample to 8x8
    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Classification head
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return models.Model(input_layer, output_layer)

model = build_cnn()
model.summary()
```

### Design choices explained

| Choice | Rationale |
|--------|-----------|
| `kernel_size=3` | Standard 3×3 filters — good receptive field vs parameter tradeoff |
| Alternating `strides=1` and `strides=2` | Extract features, then downsample — $32 \to 16 \to 8$ |
| Filter growth 32 → 64 | More channels as spatial dims shrink — classic pyramid |
| `LeakyReLU` | Avoids dead ReLU neurons; small negative slope |
| `Dense(128)` + `Dropout(0.5)` | Compact decision layer with strong regularization |
| No dropout after conv layers | Batch norm provides sufficient regularization here |

### Shape and parameter walkthrough

| Stage | Output shape | Key params |
|-------|-------------|------------|
| Input | `(None, 32, 32, 3)` | — |
| Conv block 1 (32 filters, s=1) | `(None, 32, 32, 32)` | 896 |
| Conv block 2 (32 filters, s=2) | `(None, 16, 16, 32)` | 9,248 |
| Conv block 3 (64 filters, s=1) | `(None, 16, 16, 64)` | 18,496 |
| Conv block 4 (64 filters, s=2) | `(None, 8, 8, 64)` | 36,928 |
| Flatten → Dense(128) → Dropout | `(None, 128)` | 524,416 |
| Dense(10, softmax) | `(None, 10)` | 1,290 |
| **Total** | | **~592K** |

Foster's exercise: **calculate every row by hand** before trusting `model.summary()`. For Conv2D:

$$
\text{params} = (k_h \times k_w \times C_{\text{in}} + 1) \times \text{filters}
$$
> **Readable form:** First conv: (3×3×3 + 1) × 32 = 896. Flatten: 8×8×64 = 4096 units into Dense(128).

The dense head dominates parameter count — a pattern you will see again when attaching classification heads to pretrained generators.

---

## Compile: Loss, Optimizer, Metrics

Identical pattern to the MLP ([Section 2.5](./section-05-mlp-training-deep-dive.md)):

```python
adam = optimizers.Adam(learning_rate=0.0005)

model.compile(
    optimizer=adam,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
```

| Component | Choice | Why |
|-----------|--------|-----|
| Loss | `categorical_crossentropy` | One-hot labels + softmax output |
| Optimizer | Adam, lr=0.0005 | Foster's CIFAR setting; stable default |
| Metric | `accuracy` | Easy to interpret on balanced 10-class data |

Multi-class classification loss:

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{i,c}\,\log \hat{y}_{i,c}
$$
> **Readable form:** Average negative log probability assigned to the correct class.

---

## Train: The fit Loop

```python
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=10,
    validation_split=0.1,
    verbose=1,
)
```

| [Hyperparameter](../../GLOSSARY.md#hyperparameter) | Value | Notes |
|----------|-------|-------|
| `batch_size` | 64 | Foster uses 32–256 range; 64 is a solid default |
| `epochs` | 10 | Enough to see convergence on CIFAR-10 |
| `validation_split` | 0.1 | 5,000 held-out training images for monitoring |

Plot `history.history["loss"]` vs `val_loss` and `accuracy` vs `val_accuracy` ([Section 2.5](./section-05-mlp-training-deep-dive.md)). Healthy training: curves track together. Val loss rising while train loss falls signals [overfitting](../../GLOSSARY.md#overfitting) — increase [dropout](../../GLOSSARY.md#dropout), add augmentation, or use [early stopping](../../GLOSSARY.md#early-stopping).

---

## Evaluate: Holdout Test Accuracy

```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.1%}")
# Foster's result: ~71.5%
```

| Model | Test accuracy | Total params |
|-------|--------------|--------------|
| MLP (Flatten + Dense) | ~49.0% | ~1.2M |
| CNN (this section) | ~71.5% | ~592K |

The CNN wins on **both** accuracy and efficiency because conv layers share weights across spatial positions ([Section 2.6](./section-06-convolutional-layers.md)).

---

## Visualize Predictions

Qualitative evaluation catches failure modes accuracy hides. Foster's Figure 2-17 plots random test images with predicted vs true labels — roughly half correct at ~71% accuracy. Cats vs dogs and animals vs vehicles are common confusion pairs.

```python
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
preds = model.predict(x_test, verbose=0)
pred_labels, true_labels = np.argmax(preds, 1), np.argmax(y_test, 1)
# Plot random samples with green/red titles for correct/incorrect
```

---

## Design Philosophy and Generative Bridge

Foster's Chapter 2 takeaway: deep networks are **flexible by design** — guidelines exist (filter pyramids, BAD ordering, 3×3 kernels), but experimentation on a holdout set is how you find what works. The **middle layers** capture high-level features that matter most for generation.

| This CNN component | Generative counterpart |
|--------------------|------------------------|
| Conv2D downsampling (strides=2) | VAE encoder; GAN discriminator |
| Conv2DTranspose upsampling | GAN generator; VAE decoder |
| Flatten + Dense bottleneck | Latent vector $z$ in VAE |
| Batch norm throughout | Stable training in DCGAN, U-Net |
| Softmax + cross-entropy | Reconstruction, adversarial, or noise-prediction loss |
| `model.evaluate(accuracy)` | FID, IS, visual sample quality, ELBO |

The training loop changes ([Section 2.5](./section-05-mlp-training-deep-dive.md)), but `layers.Input`, `Conv2D`, `BatchNormalization`, and `Model` carry forward unchanged. Chapter 03 replaces the softmax head with a VAE encoder that outputs `z_mean` and `z_log_var` from the same conv backbone — same layers, different loss.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Flattening before conv layers | MLP-level accuracy (~49%) | Feed `(32,32,3)` directly to Conv2D |
| Integer labels + `categorical_crossentropy` | Wrong loss semantics | `utils.to_categorical` |
| No validation monitoring | Overfitting unnoticed | `validation_split` or `validation_data` |
| `model.predict` without `training=False` in custom loops | Suboptimal batch norm stats | Pass `training=False` explicitly |
| Copying MLP learning rate blindly | Slow or unstable CNN training | Start with Foster's `0.0005`, tune on val loss |
| Judging only by accuracy | Misses systematic class errors | Visualize predictions per class |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **CNN** | Convolutional neural network — stacks conv layers for spatial feature learning |
| **Conv block** | Conv2D + batch norm + activation — repeatable building unit |
| **Downsampling** | Reducing spatial dimensions via stride-2 conv or pooling |
| **Classification head** | Flatten + Dense layers mapping features to class probabilities |
| **Test set** | Held-out data never seen during training — final performance estimate |
| **Validation split** | Fraction of training data used to monitor generalization during `fit` |

---

## Reflection Questions

1. Why does this CNN have fewer parameters than the MLP but higher accuracy?
2. Hand-calculate the output shape after the second `Conv2D(32, strides=2)` layer starting from $(32, 32, 3)$.
3. Which single change would you try first to push accuracy above 75% — more filters, more epochs, or data augmentation?
4. How does the compile step differ for a VAE compared to this classifier?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 2 — Building the CNN; Training and Evaluating the CNN. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Foster's codebase: [https://github.com/davidADSP/GDL_code](https://github.com/davidADSP/GDL_code) — `notebooks/02_deeplearning/02_cnn/cnn.ipynb`
- Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images (CIFAR-10). [https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

---

**Previous:** [Section 2.7 — Batch Normalization & Dropout](./section-07-batch-normalization-and-dropout.md)  
**Next:** [Lab 02 — Deep Learning Practice](./section-lab-02-mlp-and-cnn-training-with-keras.md)
