# Lab 03: VAE on CelebA with Latent Arithmetic

> **Prerequisites:** Sections 03.1-03.8
> **Estimated time:** 12-14 hours
> **Tools:** Python 3.10+, Jupyter, TensorFlow 2.x, Foster GDL_code
> **Glossary:** [GLOSSARY.md](../../GLOSSARY.md) | **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Lab Objectives

1. Train an autoencoder on Fashion-MNIST; visualize 2D latent space with t-SNE or direct projection.
2. Convert to a VAE; implement the combined reconstruction + KL loss in Keras.
3. Train on CelebA (64×64); sample 16 faces from N(0, I).
4. Perform latent arithmetic (e.g., neutral → smiling) and morph between two identities.
5. *Deliverable:* Grid of generated faces, arithmetic examples, and ELBO training curves.

### Math You'll Use

$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x)\,\|\,p(z))
$$
> **Readable form:** VAE loss balances reconstruction against KL to prior.


$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)
$$
> **Readable form:** Reparameterization trick makes sampling differentiable.


---

1. Train an autoencoder on Fashion-MNIST; visualize 2D latent space with t-SNE or direct projection.
2. Convert to a VAE; implement the combined reconstruction + KL loss in Keras.
3. Train on CelebA (64×64); sample 16 faces from N(0, I).
4. Perform latent arithmetic (e.g., neutral → smiling) and morph between two identities.
5. *Deliverable:* Grid of generated faces, arithmetic examples, and ELBO training curves.

---

## Part A: Core Implementation

Follow Foster's notebook for this chapter. Adapt the starter code below.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

latent_dim = 32
encoder_in = layers.Input(shape=(64, 64, 3))
x = layers.Conv2D(32, 4, strides=2, activation="relu", padding="same")(encoder_in)
x = layers.Conv2D(64, 4, strides=2, activation="relu", padding="same")(x)
x = layers.Flatten()(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)
z = Sampling()([z_mean, z_log_var])
encoder = Model(encoder_in, [z_mean, z_log_var, z], name="encoder")
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

1. What was the hardest part of **vae on celeba with latent arithmetic** to implement?
2. Which section topic proved most useful during the lab?
3. How would sample quality change with 10× more training data?
4. What generative chapter (03-13) shares the most DNA with this lab?

---

## Part D: Extended Exercises

Complete at least two exercises for **VAE on CelebA with Latent Arithmetic**:

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

- Foster, Ch. 3 - *Variational Autoencoders*
- [Chapter 03 README](./README.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Extended Notes

Re-read Chapter 3 after completing the lab. Focus on assumptions that make **vae on celeba with latent arithmetic** valid.

### Practice checklist

- [ ] Run the Foster notebook for **vae on celeba with latent arithmetic** end-to-end
- [ ] Save at least one diagnostic figure
- [ ] Write a 3-sentence summary in your own words
- [ ] Link this section to one later chapter

### Bridge to generative AI

Modern systems (diffusion models, LLMs, VAEs) reuse the ideas in **vae on celeba with latent arithmetic**.
The vocabulary changes; the mathematics rhymes.

### Historical context

Foster situates **vae on celeba with latent arithmetic** within the evolution from VAEs and GANs to transformers.

---

## Extended Notes

Re-read Chapter 3 after completing the lab. Focus on assumptions that make **vae on celeba with latent arithmetic** valid.

### Practice checklist

- [ ] Run the Foster notebook for **vae on celeba with latent arithmetic** end-to-end
- [ ] Save at least one diagnostic figure
- [ ] Write a 3-sentence summary in your own words
- [ ] Link this section to one later chapter

### Bridge to generative AI

Modern systems (diffusion models, LLMs, VAEs) reuse the ideas in **vae on celeba with latent arithmetic**.
The vocabulary changes; the mathematics rhymes.

**Previous:** [Section 03.8 - Latent Space Arithmetic](./section-08-latent-space-arithmetic.md)
**Next:** [Chapter 03 README](./README.md)

---

## Assessment Practice

Use the shared [Assessment Appendix](../../ASSESSMENT_APPENDIX.md) for concept audits, worked examples, implementation checks, experiment logs, oral-exam prompts, and deliverable checklists.
