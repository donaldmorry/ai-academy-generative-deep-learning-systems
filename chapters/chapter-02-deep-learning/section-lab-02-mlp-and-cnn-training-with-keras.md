# Lab 02: MLP and CNN Training with Keras

> **Prerequisites:** Sections 02.1-02.8
> **Estimated time:** 10-12 hours
> **Tools:** Python 3.10+, Jupyter, TensorFlow 2.x, Foster GDL_code
> **Glossary:** [GLOSSARY.md](../../GLOSSARY.md) | **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Lab Objectives

1. Train an MLP on a tabular dataset from the book (or UCI fallback) using Keras.
2. Build a CNN for MNIST or Fashion-MNIST with Conv2D, BatchNorm, Dropout, and Dense head.
3. Plot training/validation accuracy and loss; diagnose overfitting.
4. Export the trained CNN weights-you will reuse Conv2D patterns in Chapter 04 (DCGAN).
5. *Deliverable:* Notebook with both models, learning curves, and a short architecture comparison.

### Math You'll Use

$$
y = \sigma(Wx + b)
$$
> **Readable form:** Layer output equals activation of weighted sum plus bias.


$$
\mathcal{L} = -\sum_i y_i \log \hat{y}_i
$$
> **Readable form:** Cross-entropy loss penalizes wrong class probabilities.


---

1. Train an MLP on a tabular dataset from the book (or UCI fallback) using Keras.
2. Build a CNN for MNIST or Fashion-MNIST with Conv2D, BatchNorm, Dropout, and Dense head.
3. Plot training/validation accuracy and loss; diagnose overfitting.
4. Export the trained CNN weights-you will reuse Conv2D patterns in Chapter 04 (DCGAN).
5. *Deliverable:* Notebook with both models, learning curves, and a short architecture comparison.

---

## Part A: Core Implementation

Follow Foster's notebook for this chapter. Adapt the starter code below.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., None]
x_test = x_test[..., None]

model = keras.Sequential([
    layers.Conv2D(32, 3, activation="relu", padding="same", input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size=128)
```

**Task:** Run training for at least 5 epochs. Save loss curves and one sample grid.

---

## Part B: Evaluation & Diagnostics

```python
import matplotlib.pyplot as plt

# Plot training history
if 'history' in dir():
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val')
    plt.legend(); plt.title('Loss')
    plt.savefig(f'lab_{mod_num}_loss.png', dpi=150)
```

**Deliverable:** At least two figures - training curves and generated samples or reconstructions.

---

## Part C: Written Reflection

Answer in a markdown cell or separate document:

1. What was the hardest part of **mlp and cnn training with keras** to implement?
2. Which section topic proved most useful during the lab?
3. How would sample quality change with 10× more training data?
4. What generative chapter (03-13) shares the most DNA with this lab?

---

## Part D: Extended Exercises

Complete at least two exercises for **MLP and CNN Training with Keras**:

1. **Hyperparameter sweep** - Vary learning rate or batch size; plot validation loss.
2. **Ablation** - Remove one component and compare outputs.
3. **Cross-chapter comparison** - Contrast with a model from an earlier chapter.
4. **Failure log** - Document one failed run and the fix applied.

```python
import tensorflow as tf
tf.random.set_seed(42)
print('GPU devices:', tf.config.list_physical_devices('GPU'))
```

---

## Troubleshooting

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| OOM on GPU | Batch too large | Halve batch size |
| Loss flat | LR wrong | Try 1e-4, 1e-3, 1e-2 |
| Poor samples | Under-training | Train longer; review Foster tips |

---

## Common Lab Mistakes

| Mistake | What Goes Wrong | Fix |
|---------|-----------------|-----|
| Skipping GPU check | Hours of CPU training | `tf.config.list_physical_devices('GPU')` |
| No random seed | Irreproducible results | Set `tf.random.set_seed(42)` and `np.random.seed(42)` |
| Wrong tensor shapes | Immediate crash | Print shapes after every layer |
| Not saving checkpoints | Lost work on crash | Use `ModelCheckpoint` callback |

---

## Submission Checklist

- [ ] Code runs end-to-end without errors
- [ ] Figures saved with clear labels
- [ ] Short write-up (200-400 words) interpreting results
- [ ] Connection to at least two section topics from this chapter

---

## Further Reading

- Foster, Ch. 2 - *Deep Learning*
- [Chapter 02 README](./README.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Extended Notes

Re-read Chapter 2 after completing the lab. Focus on assumptions that make **mlp and cnn training with keras** valid.

### Practice checklist

- [ ] Run the Foster notebook for **mlp and cnn training with keras** end-to-end
- [ ] Save at least one diagnostic figure
- [ ] Write a 3-sentence summary in your own words
- [ ] Link this section to one later chapter

### Bridge to generative AI

Modern systems (diffusion models, LLMs, VAEs) reuse the ideas in **mlp and cnn training with keras**.
The vocabulary changes; the mathematics rhymes.

### Historical context

Foster situates **mlp and cnn training with keras** within the evolution from VAEs and GANs to transformers.

---

## Extended Notes

Re-read Chapter 2 after completing the lab. Focus on assumptions that make **mlp and cnn training with keras** valid.

### Practice checklist

- [ ] Run the Foster notebook for **mlp and cnn training with keras** end-to-end
- [ ] Save at least one diagnostic figure
- [ ] Write a 3-sentence summary in your own words
- [ ] Link this section to one later chapter
**Previous:** [Section 02.8 - Building and Training a CNN](./section-08-building-and-training-a-cnn.md)
**Next:** [Chapter 02 README](./README.md)

---

## Assessment Practice

Use the shared [Assessment Appendix](../../ASSESSMENT_APPENDIX.md) for concept audits, worked examples, implementation checks, experiment logs, oral-exam prompts, and deliverable checklists.
