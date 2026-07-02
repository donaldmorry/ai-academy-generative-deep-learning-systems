# Section 5.1: Autoregressive Framework

> **Source inheritance:** Foster, Ch. 5 — "Introduction" & autoregressive factorization  
> **Enhanced with:** Chain rule, explicit likelihood, comparison to VAE/GAN latent models  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

VAEs and GANs introduce a latent variable $z$ and learn to decode it. **Autoregressive models** take a different path: factorize the joint distribution into a **product of conditionals** and generate one token — a word, a pixel — at a time. Foster's LSTM prison story captures the essence: each inmate updates their opinion based on the latest word and yesterday's collective state; the guard picks the next word from disclosed opinions.

No latent bottleneck required — the model learns $P(x)$ **directly** (for a chosen ordering).

---

## Chain Rule Factorization

For data $x = (x_1, x_2, \ldots, x_n)$:

$$
P(x) = \prod_{i=1}^{n} P(x_i \mid x_1, \ldots, x_{i-1})
$$
> **Readable form:** Joint probability equals product of "next item given everything before."

**Log-likelihood** (what we maximize in training):

$$
\log P(x) = \sum_{i=1}^{n} \log P(x_i \mid x_{<i})
$$
> **Readable form:** The probability is obtained by summing over the hidden cases or outcomes that satisfy the query.

Each term is a supervised prediction problem — predict position $i$ from prefix $x_{<i}$.

| Model family | Likelihood | Generation |
|--------------|------------|------------|
| VAE | ELBO lower bound | Sample $z$, decode |
| GAN | Implicit | Sample $z$, generate |
| **Autoregressive** | **Exact** (tractable) | Sequential sampling |

---

## Ordering Choices Matter

The factorization depends on variable order:

| Domain | Typical order |
|--------|---------------|
| Text | Left-to-right tokens |
| Images (PixelCNN) | Row-major raster scan |
| Audio | Time steps |

Different orderings define **different** models — not interchangeable. PixelCNN uses top-left to bottom-right ([Section 5.6](./section-06-pixelcnn.md)).

---

## Autoregressive vs Latent Models

Foster contrasts with Chapters 03–04:

```
VAE/GAN:  z ~ p(z)  --->  decoder/generator  --->  x

Autoregressive:  x_1  --->  x_2  --->  x_3  ---> ... --->  x_n
                  P(x_1)   P(x_2|x_1)  P(x_3|x_1,x_2)
```

**Advantages:**

- Tractable exact likelihood (no adversarial game, no ELBO gap)
- Stable maximum likelihood training
- Flexible architectures (LSTM, Transformer)

**Disadvantages:**

- **Sequential sampling** — slow for high-dimensional data
- Ordering constraint may be unnatural for some domains
- Long-range dependencies require memory (LSTM gates, attention)

---

## Minimal TensorFlow Illustration

Toy 1-D sequence of categorical tokens:

```python
import tensorflow as tf

# Vocabulary size 10, sequence length 5
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10, 32, input_length=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(10, activation="softmax"),
])

# Input: first 4 tokens; target: 5th token
# x_in shape (batch, 4), y shape (batch,) integer labels
# model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
```

Each forward pass predicts **one** next token; full sequence likelihood multiplies (sums in log space) over positions.

---

## Teacher Forcing Preview

Training uses **ground-truth** prefixes (teacher forcing):

$$
\text{Train on } P(x_i \mid x_1^{\text{true}}, \ldots, x_{i-1}^{\text{true}})
$$
> **Readable form:** The expression assigns probability to the event or value using the stated model assumptions.

Sampling uses **model's own** predictions as prefix — **exposure bias** ([Section 5.4](./section-04-training-and-sampling-text.md)).

---

## Chapter 05 Roadmap

| Section | Model | Data |
|--------|-------|------|
| 5.1 | Framework | Theory |
| 5.2–5.4 | LSTM | Epicurious recipes |
| 5.5 | RNN extensions | GRU, stacked, bidirectional |
| 5.6–5.8 | PixelCNN | Fashion-MNIST images |

Transformers (Chapter 9 in Foster) are the modern autoregressive backbone — Chapter 05 builds the sequential intuition LSTMs provide.

---

## Connection to Prior Courses

| Concept | Link |
|---------|------|
| Chain rule / probability | [Course 3, Chapter 03](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-03-probability-information-theory/README.md) |
| MLE | [Chapter 01](../chapter-01-generative-modeling/section-04-our-first-generative-model.md) |
| RNN basics | [Course 1, Chapter 13](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/section-04-recurrent-neural-networks.md) |

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Ignoring ordering in images | Invalid factorization | Fixed raster order in PixelCNN |
| Expecting fast image sampling | Hours per image | Use VAE/GAN/diffusion for speed |
| Confusing with Markov assumption | Wrong — full prefix used | AR uses entire $x_{<i}$ via hidden state |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Autoregressive model** | Factorizes $P(x)$ as product of conditionals |
| **Chain rule** | $P(x) = \prod_i P(x_i \mid x_{<i})$ |
| **Tractable likelihood** | Exact $\log P(x)$ computable |
| **Ordering** | Sequence in which variables are generated |
| **Teacher forcing** | Training with true prefixes |

---

## Reflection Questions

1. Why is autoregressive likelihood called "tractable" compared to GANs?
2. How does the chain rule apply to generating a sentence word by word?
3. What is the main sampling-speed disadvantage vs VAEs?
4. Why must PixelCNN fix a pixel ordering before applying convolutions?

---


## Foster Notebook Reference

Re-run the chapter notebook in [GDL_code](https://github.com/davidADSP/GDL_code) and compare your tensor shapes, loss curves, and saved sample grids to Foster's figures. Document one hyperparameter you changed and how outputs shifted — this habit transfers directly to Part III architectures (Transformers, Stable Diffusion, MuseGAN).

| Checkpoint | Action |
|------------|--------|
| After `model.summary()` | Verify spatial dims match hand calculation |
| Mid-training | Save sample grid or diagnostic plot |
| After training | Compare to Foster figure captions in the PDF |

---

## Extension Reading

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.) — full chapter walkthrough
- Goodfellow, Bengio & Courville (2016). *Deep Learning* — generative models part
- Original papers cited in Foster's chapter references



## Lab Integration Notes

When completing the chapter lab, tie this section's implementation checklist to your deliverable:

1. **Reproduce** Foster's primary figure for this topic (save PNG).
2. **Ablate** one hyperparameter (learning rate, latent dim, noise schedule, critic steps, etc.).
3. **Compare** to one other Part II model family on the same metric (likelihood, sample sharpness, or training stability).
4. **Write** three sentences explaining failure modes you observed and the fix you applied.

These steps mirror how generative modeling is evaluated in research and production — not only final image quality, but reproducibility and diagnosed trade-offs.

## Tensor Shape Debugging Template

```python
import tensorflow as tf

def trace_shapes(model, sample_input):
    x = sample_input
    for layer in model.layers:
        try:
            x = layer(x)
            print(f"{layer.name:30s} {x.shape}")
        except Exception as e:
            print(f"{layer.name:30s} ERROR: {e}")
            break
```

Use after every architecture change before committing to long training runs.


## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 5 — Introduction. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- van den Oord, A. et al. (2016). Pixel Recurrent Neural Networks.
- Goodfellow, I. et al. (2016). *Deep Learning*. Ch. 12 — Applications.

---

**Previous:** [Section 4.8 — Analysis & Comparison](../chapter-04-generative-adversarial-networks/section-08-analysis-and-comparison.md)  
**Next:** [Section 5.2 — LSTM for Text](./section-02-lstm-for-text.md)



