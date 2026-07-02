# Lab 06: RealNVP on Two Moons

> **Prerequisites:** Sections 06.1-06.8
> **Estimated time:** 10-12 hours
> **Tools:** Python 3.10+, Jupyter, TensorFlow 2.x, Foster GDL_code
> **Glossary:** [GLOSSARY.md](../../GLOSSARY.md) | **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Lab Objectives

1. Implement affine coupling layers with alternating checkerboard masks in Keras.
2. Train RealNVP on the Two Moons dataset; plot learned log-density contours.
3. Sample 1000 points from the trained flow; compare to ground-truth distribution.
4. Write a comparison table: flows vs VAEs vs autoregressive on likelihood, speed, flexibility.
5. *Deliverable:* Density contour plots, samples, and model comparison table.

### Math You'll Use

$$
\log p_X(x) = \log p_Z(f^{-1}(x)) + \log\left|\det\frac{\partial f^{-1}}{\partial x}\right|
$$
> **Readable form:** Change of variables adds log Jacobian determinant.


$$
y_1 = x_1, \quad y_2 = x_2 \odot \exp(s(x_1)) + t(x_1)
$$
> **Readable form:** RealNVP affine coupling transforms half the dimensions.


---

1. Implement affine coupling layers with alternating checkerboard masks in Keras.
2. Train RealNVP on the Two Moons dataset; plot learned log-density contours.
3. Sample 1000 points from the trained flow; compare to ground-truth distribution.
4. Write a comparison table: flows vs VAEs vs autoregressive on likelihood, speed, flexibility.
5. *Deliverable:* Density contour plots, samples, and model comparison table.

---

## Part A: Core Implementation

Follow Foster's notebook for this chapter. Adapt the starter code below.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Foster 06-normalizing-flow-models: RealNVP on Two Moons
inputs = layers.Input(shape=(None,))
x = layers.Dense(128, activation="relu")(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
print(model.summary())
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

1. What was the hardest part of **realnvp on two moons** to implement?
2. Which section topic proved most useful during the lab?
3. How would sample quality change with 10× more training data?
4. What generative chapter (03-13) shares the most DNA with this lab?

---

## Part D: Extended Exercises

Complete at least two exercises for **RealNVP on Two Moons**:

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

- Foster, Ch. 6 - *Normalizing Flow Models*
- [Chapter 06 README](./README.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Extended Notes

Re-read Chapter 6 after completing the lab. Focus on assumptions that make **realnvp on two moons** valid.

### Practice checklist

- [ ] Run the Foster notebook for **realnvp on two moons** end-to-end
- [ ] Save at least one diagnostic figure
- [ ] Write a 3-sentence summary in your own words
- [ ] Link this section to one later chapter

### Bridge to generative AI

Modern systems (diffusion models, LLMs, VAEs) reuse the ideas in **realnvp on two moons**.
The vocabulary changes; the mathematics rhymes.

### Historical context

Foster situates **realnvp on two moons** within the evolution from VAEs and GANs to transformers.

---

## Extended Notes
**Previous:** [Section 06.8 - FFJORD & Summary](./section-08-ffjord-and-normalizing-flows-summary.md)
**Next:** [Chapter 06 README](./README.md)

---

## Assessment Practice

Use the shared [Assessment Appendix](../../ASSESSMENT_APPENDIX.md) for concept audits, worked examples, implementation checks, experiment logs, oral-exam prompts, and deliverable checklists.
