# Lab 08: DDPM on Flowers with U-Net

> **Prerequisites:** Sections 08.1-08.8
> **Estimated time:** 13-15 hours
> **Tools:** Python 3.10+, Jupyter, TensorFlow 2.x, Foster GDL_code
> **Glossary:** [GLOSSARY.md](../../GLOSSARY.md) | **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Lab Objectives

1. Implement the forward diffusion process with a cosine noise schedule.
2. Build a time-conditioned U-Net denoiser in Keras; embed diffusion timestep.
3. Train on the Flowers dataset; monitor noise-prediction loss.
4. Run the full reverse sampling loop (1000 steps or DDIM-accelerated variant).
5. *Deliverable:* Generated flower images, training curves, and schedule comparison notes.

### Math You'll Use

$$
q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\,x_0,\,(1-\bar{\alpha}_t)I)
$$
> **Readable form:** Forward diffusion: scaled signal plus Gaussian noise at step t.


$$
\mathcal{L} = \mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$
> **Readable form:** DDPM trains network to predict noise epsilon.


---

1. Implement the forward diffusion process with a cosine noise schedule.
2. Build a time-conditioned U-Net denoiser in Keras; embed diffusion timestep.
3. Train on the Flowers dataset; monitor noise-prediction loss.
4. Run the full reverse sampling loop (1000 steps or DDIM-accelerated variant).
5. *Deliverable:* Generated flower images, training curves, and schedule comparison notes.

---

## Part A: Core Implementation

Follow Foster's notebook for this chapter. Adapt the starter code below.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Foster 08-diffusion-models: DDPM on Flowers with U-Net
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

1. What was the hardest part of **ddpm on flowers with u-net** to implement?
2. Which section topic proved most useful during the lab?
3. How would sample quality change with 10× more training data?
4. What generative chapter (03-13) shares the most DNA with this lab?

---

## Part D: Extended Exercises

Complete at least two exercises for **DDPM on Flowers with U-Net**:

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

- Foster, Ch. 8 - *Diffusion Models*
- [Chapter 08 README](./README.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Extended Notes

Re-read Chapter 8 after completing the lab. Focus on assumptions that make **ddpm on flowers with u-net** valid.

### Practice checklist

- [ ] Run the Foster notebook for **ddpm on flowers with u-net** end-to-end
- [ ] Save at least one diagnostic figure
- [ ] Write a 3-sentence summary in your own words
- [ ] Link this section to one later chapter

### Bridge to generative AI

Modern systems (diffusion models, LLMs, VAEs) reuse the ideas in **ddpm on flowers with u-net**.
The vocabulary changes; the mathematics rhymes.

### Historical context

Foster situates **ddpm on flowers with u-net** within the evolution from VAEs and GANs to transformers.

---

## Extended Notes
**Previous:** [Section 08.8 - Analysis & Connections](./section-08-analysis-and-connections.md)
**Next:** [Chapter 08 README](./README.md)

---

## Assessment Practice

Use the shared [Assessment Appendix](../../ASSESSMENT_APPENDIX.md) for concept audits, worked examples, implementation checks, experiment logs, oral-exam prompts, and deliverable checklists.
