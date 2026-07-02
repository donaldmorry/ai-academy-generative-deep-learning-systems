# Lab 07: EBM on MNIST with Langevin Sampling

> **Prerequisites:** Sections 07.1-07.7
> **Estimated time:** 9-11 hours
> **Tools:** Python 3.10+, Jupyter, TensorFlow 2.x, Foster GDL_code
> **Glossary:** [GLOSSARY.md](../../GLOSSARY.md) | **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Lab Objectives

1. Build a CNN-based energy function for MNIST in Keras.
2. Implement Langevin dynamics sampling; tune step size and number of steps.
3. Train with contrastive divergence; log energy of real vs generated samples.
4. Generate 64 MNIST-like digits; discuss sample diversity and artifacts.
5. *Deliverable:* Generated samples, energy histograms, and MCMC hyperparameter notes.

### Math You'll Use

$$
p(x) = \frac{\exp(-E(x))}{Z}, \quad Z = \int \exp(-E(x))\,dx
$$
> **Readable form:** Boltzmann distribution; partition function Z is intractable.


$$
x_{t+1} = x_t - \frac{\eta}{2}\nabla_x E(x_t) + \sqrt{\eta}\,\epsilon
$$
> **Readable form:** Langevin dynamics: gradient descent on energy plus noise.


---

1. Build a CNN-based energy function for MNIST in Keras.
2. Implement Langevin dynamics sampling; tune step size and number of steps.
3. Train with contrastive divergence; log energy of real vs generated samples.
4. Generate 64 MNIST-like digits; discuss sample diversity and artifacts.
5. *Deliverable:* Generated samples, energy histograms, and MCMC hyperparameter notes.

---

## Part A: Core Implementation

Follow Foster's notebook for this chapter. Adapt the starter code below.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Foster 07-energy-based-models: EBM on MNIST with Langevin Sampling
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

1. What was the hardest part of **ebm on mnist with langevin sampling** to implement?
2. Which section topic proved most useful during the lab?
3. How would sample quality change with 10× more training data?
4. What generative chapter (03-13) shares the most DNA with this lab?

---

## Part D: Extended Exercises

Complete at least two exercises for **EBM on MNIST with Langevin Sampling**:

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

- Foster, Ch. 7 - *Energy-Based Models*
- [Chapter 07 README](./README.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Extended Notes

Re-read Chapter 7 after completing the lab. Focus on assumptions that make **ebm on mnist with langevin sampling** valid.

### Practice checklist

- [ ] Run the Foster notebook for **ebm on mnist with langevin sampling** end-to-end
- [ ] Save at least one diagnostic figure
- [ ] Write a 3-sentence summary in your own words
- [ ] Link this section to one later chapter

### Bridge to generative AI

Modern systems (diffusion models, LLMs, VAEs) reuse the ideas in **ebm on mnist with langevin sampling**.
The vocabulary changes; the mathematics rhymes.

### Historical context

Foster situates **ebm on mnist with langevin sampling** within the evolution from VAEs and GANs to transformers.

---

## Extended Notes
**Previous:** [Section 07.7 - Other EBMs & Connections](./section-07-other-ebms-and-connections.md)
**Next:** [Chapter 07 README](./README.md)

---

## Assessment Practice

Use the shared [Assessment Appendix](../../ASSESSMENT_APPENDIX.md) for concept audits, worked examples, implementation checks, experiment logs, oral-exam prompts, and deliverable checklists.
