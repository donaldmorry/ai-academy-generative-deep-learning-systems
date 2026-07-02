# Section 5.3: Embeddings & LSTM Layers

> **Source inheritance:** Foster, Ch. 5 — "Building the LSTM" & LSTM cell internals  
> **Enhanced with:** Embedding layer, return_sequences, stacked LSTM, Keras implementation  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Integer token IDs from [Section 5.2](./section-02-lstm-for-text.md) are meaningless to a matrix multiply — token 7 ("the") and token 8 ("with") appear as unrelated scalars. An **Embedding** layer learns a dense vector for each vocabulary entry, placing similar words near each other in $\mathbb{R}^d$. The **LSTM** then carries hidden state across timesteps, implementing Foster's prison analogy: 256 inmates updating opinions as each new word arrives.

This section builds Foster's recipe LSTM: `Embedding(10000, 100)` → `LSTM(128, return_sequences=True)` → `Dense(10000, softmax)`.

---

## Embedding Layer

Maps token index $w_i \in \{0, \ldots, |V|-1\}$ to vector $\mathbf{e}_i \in \mathbb{R}^{d_{\text{emb}}}$:

$$
\mathbf{e}_i = E[w_i], \quad E \in \mathbb{R}^{|V| \times d_{\text{emb}}}
$$
> **Readable form:** Lookup table — each word gets a learned vector of length 100.

```python
from tensorflow.keras import layers, models

embedding_size = 100
total_words = 10000
n_units = 128

text_in = layers.Input(shape=(None,), dtype="int32")
x = layers.Embedding(total_words, embedding_size)(text_in)
```

| Hyperparameter | Foster value | Trade-off |
|----------------|--------------|-----------|
| `total_words` | 10000 | Vocab coverage vs softmax size |
| `embedding_size` | 100 | Capacity vs parameters |

Embeddings are **learned end-to-end** — no need for pretrained GloVe in Foster's example (unlike [Course 1 NLP](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-13-natural-language-processing/README.md)).

---

## LSTM Cell Intuition

Foster's prison story maps to gates:

| Gate | Story role | Effect |
|------|------------|--------|
| **Forget** | How much yesterday to discard | Reset irrelevant context |
| **Input** | New thoughts from current word | Write new information |
| **Output** | What to disclose to guard | Hidden state for prediction |

Mathematically (simplified):

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t]), \quad i_t = \sigma(W_i \cdot [h_{t-1}, x_t])
$$

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t, \quad h_t = o_t \odot \tanh(C_t)
$$
> **Readable form:** Forget old cell state, add new content, output filtered hidden state.

LSTMs solve **vanishing gradients** in vanilla RNNs — critical for recipes where "chill until set" depends on ingredients named 50 words earlier.

---

## return_sequences=True

For **sequence-to-sequence** next-token prediction at **every** position:

```python
x = layers.LSTM(n_units, return_sequences=True)(x)
probabilities = layers.Dense(total_words, activation="softmax")(x)
model = models.Model(text_in, probabilities)
```

| `return_sequences` | Output shape | Use case |
|--------------------|--------------|----------|
| `False` | `(batch, units)` | Single classification |
| `True` | `(batch, time, units)` | Per-timestep prediction |

Output shape: `(None, 200, 10000)` — softmax over full vocabulary at each of 200 positions.

---

## Model Summary (Foster Table 5-1)

| Layer | Output shape | Params |
|-------|--------------|--------|
| Input | (None, None) | 0 |
| Embedding | (None, None, 100) | 1,000,000 |
| LSTM | (None, None, 128) | ~117,000 |
| Dense softmax | (None, None, 10000) | ~1,290,000 |

Embedding + final Dense dominate parameters — vocabulary size drives cost.

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
)
```

`sparse_categorical_crossentropy`: integer targets `y` without one-hot encoding.

---

## Training Objective

Maximize $\sum_t \log P(w_t \mid w_{<t})$ — each timestep contributes cross-entropy:

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(w_t \mid w_1, \ldots, w_{t-1})
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

Teacher forcing: LSTM sees **true** prefix tokens during training ([Section 5.4](./section-04-training-and-sampling-text.md)).

---

## Stacked LSTM Preview

[Section 5.5](./section-05-rnn-extensions.md) stacks two LSTM layers:

```python
x = layers.LSTM(128, return_sequences=True)(embedding)
x = layers.LSTM(128, return_sequences=True)(x)
```

Deeper stacks capture hierarchical syntax — clauses within recipes.

---

## Connection to Chapter 01

Autoregressive factorization from [Section 5.1](./section-01-autoregressive-framework.md) — LSTM parameterizes each conditional $P(w_t \mid w_{<t})$ with shared weights across timesteps.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| `return_sequences=False` | Shape error on Dense | Use `True` for seq2seq |
| Embedding size mismatch | Wrong vocab in Embedding | Match `max_tokens` |
| One-hot targets + wrong loss | 10K-dim labels | Use sparse CE |
| No GPU for large softmax | Slow epochs | Reduce vocab or batch |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Embedding** | Learned dense vector per token ID |
| **LSTM** | Gated RNN cell with cell state $C_t$ and hidden $h_t$ |
| **return_sequences** | Output hidden state at every timestep |
| **Teacher forcing** | Training with ground-truth prefixes |
| **Sparse categorical CE** | Cross-entropy with integer class labels |

---

## Reflection Questions

1. Why are embeddings necessary before LSTM processing?
2. What does `return_sequences=True` enable for next-word prediction?
3. Which two layers dominate parameter count and why?
4. How do LSTM gates address vanishing gradients?

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

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 5 — Building the LSTM. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*.
- Foster's codebase: `notebooks/05_autoregressive/01_lstm/lstm.ipynb`

---

**Previous:** [Section 5.2 — LSTM for Text](./section-02-lstm-for-text.md)  
**Next:** [Section 5.4 — Training & Sampling Text](./section-04-training-and-sampling-text.md)
