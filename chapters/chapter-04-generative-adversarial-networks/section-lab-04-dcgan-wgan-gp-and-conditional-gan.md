# Lab 04: DCGAN, WGAN-GP, and Conditional GAN

> **Prerequisites:** Sections 04.1-04.8
> **Estimated time:** 12-15 hours
> **Tools:** Python 3.10+, Jupyter, TensorFlow 2.x, Foster GDL_code
> **Glossary:** [GLOSSARY.md](../../GLOSSARY.md) | **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Lab Objectives

1. Train a DCGAN on the Bricks dataset; log generator/discriminator losses and sample every N epochs.
2. Document one training failure (e.g., mode collapse) and the fix you applied.
3. Implement WGAN-GP; compare training stability to vanilla DCGAN.
4. Train a CGAN on MNIST with class-conditioned generation (generate specific digits).
5. *Deliverable:* Sample grids from all three models and a training stability comparison write-up.

### Math You'll Use

$$
\min_G \max_D \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1-D(G(z)))]
$$
> **Readable form:** GAN minimax: D maximizes real/fake discrimination; G fools D.


$$
W = \sup_{\|f\|_L \leq 1} \mathbb{E}_{p}[f(x)] - \mathbb{E}_{q}[f(x)]
$$
> **Readable form:** Wasserstein distance via 1-Lipschitz critic.


---

1. Train a DCGAN on the Bricks dataset; log generator/discriminator losses and sample every N epochs.
2. Document one training failure (e.g., mode collapse) and the fix you applied.
3. Implement WGAN-GP; compare training stability to vanilla DCGAN.
4. Train a CGAN on MNIST with class-conditioned generation (generate specific digits).
5. *Deliverable:* Sample grids from all three models and a training stability comparison write-up.

---

## Part A: Core Implementation

Follow Foster's notebook for this chapter. Adapt the starter code below.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_generator(latent_dim=100):
    return tf.keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(latent_dim,)),
        layers.BatchNormalization(), layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, 5, strides=1, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.LeakyReLU(),
        layers.Conv2DTranspose(64, 5, strides=2, padding="same", use_bias=False),
    ], name="generator")
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

1. What was the hardest part of **dcgan, wgan-gp, and conditional gan** to implement?
2. Which section topic proved most useful during the lab?
3. How would sample quality change with 10× more training data?
4. What generative chapter (03-13) shares the most DNA with this lab?

---

## Part D: Extended Exercises

Complete at least two exercises for **DCGAN, WGAN-GP, and Conditional GAN**:

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

- Foster, Ch. 4 - *Generative Adversarial Networks*
- [Chapter 04 README](./README.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Extended Notes

Re-read Chapter 4 after completing the lab. Focus on assumptions that make **dcgan, wgan-gp, and conditional gan** valid.

### Practice checklist

- [ ] Run the Foster notebook for **dcgan, wgan-gp, and conditional gan** end-to-end
- [ ] Save at least one diagnostic figure
- [ ] Write a 3-sentence summary in your own words
- [ ] Link this section to one later chapter

### Bridge to generative AI

Modern systems (diffusion models, LLMs, VAEs) reuse the ideas in **dcgan, wgan-gp, and conditional gan**.
The vocabulary changes; the mathematics rhymes.

### Historical context

Foster situates **dcgan, wgan-gp, and conditional gan** within the evolution from VAEs and GANs to transformers.

---

## Extended Notes

Re-read Chapter 4 after completing the lab. Focus on assumptions that make **dcgan, wgan-gp, and conditional gan** valid.

### Practice checklist

- [ ] Run the Foster notebook for **dcgan, wgan-gp, and conditional gan** end-to-end
- [ ] Save at least one diagnostic figure
- [ ] Write a 3-sentence summary in your own words
- [ ] Link this section to one later chapter

### Bridge to generative AI

Modern systems (diffusion models, LLMs, VAEs) reuse the ideas in **dcgan, wgan-gp, and conditional gan**.
The vocabulary changes; the mathematics rhymes.

### Historical context

Foster situates **dcgan, wgan-gp, and conditional gan** within the evolution from VAEs and GANs to transformers.

---

**Previous:** [Section 04.8 - Analysis & Comparison](./section-08-analysis-and-comparison.md)
**Next:** [Chapter 04 README](./README.md)

---

## Assessment Practice

Use the shared [Assessment Appendix](../../ASSESSMENT_APPENDIX.md) for concept audits, worked examples, implementation checks, experiment logs, oral-exam prompts, and deliverable checklists.
