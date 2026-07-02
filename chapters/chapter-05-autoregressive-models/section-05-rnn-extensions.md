# Section 5.5: RNN Extensions

> **Source inheritance:** Foster, Ch. 5 — "Recurrent Neural Network (RNN) Extensions"  
> **Enhanced with:** Stacked LSTMs, GRU, bidirectional cells, when to use each  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

A single [LSTM](../../GLOSSARY.md#lstm-long-short-term-memory) layer captures sequential dependencies, but complex recipes need **hierarchical** representations — phrases within sentences, steps within procedures. Foster stacks two LSTM layers and surveys **GRU** (fewer gates, faster) and **Bidirectional** wrappers (forward + backward context). For **generation**, only unidirectional models apply — you cannot see the future when writing the next word.

---

## Stacked (Multi-Layer) LSTM

First LSTM outputs sequence; second LSTM processes that sequence:

```python
text_in = layers.Input(shape=(None,))
x = layers.Embedding(total_words, embedding_size)(text_in)
x = layers.LSTM(n_units, return_sequences=True)(x)  # layer 1
x = layers.LSTM(n_units, return_sequences=True)(x)  # layer 2
probabilities = layers.Dense(total_words, activation="softmax")(x)
model = models.Model(text_in, probabilities)
```

Foster Table 5-2: ~2.54M parameters vs ~2.41M for single layer — modest increase, better long-range structure.

| Depth | Captures |
|-------|----------|
| 1 LSTM | Local word patterns |
| 2 LSTM | Clause-level structure |
| 3+ | Diminishing returns; harder to train |

---

## Gated Recurrent Unit (GRU)

GRU merges forget/input gates into **reset** and **update** gates; no separate cell state $C_t$ — only hidden $h_t$.

| | LSTM | GRU |
|---|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| State | $C_t$ and $h_t$ | $h_t$ only |
| Parameters | More | **Fewer** |
| Performance | Often similar on NLP |

```python
x = layers.GRU(128, return_sequences=True)(embedding)
```

Foster Figure 5-12: reset gate controls how much past hidden state enters new beliefs; update gate blends old and new $h_t$.

---

## Bidirectional Layers

Processes sequence forward **and** backward — hidden state is concatenation:

```python
layer = layers.Bidirectional(layers.GRU(100, return_sequences=True))
# Output features: 200 (= 100 forward + 100 backward)
```

**Use for:** classification, tagging, encoding (full text available).

**Not for:** autoregressive **generation** — backward context is unknown at inference.

| Task | Bidirectional? |
|------|----------------|
| Recipe **generation** | No |
| Sentiment on full recipe | Yes |
| BERT-style pretraining | Yes (Chapter 09) |

---

## Choosing an Architecture

| Goal | Recommendation |
|------|----------------|
| Fast baseline | Single LSTM |
| Better quality | Stacked LSTM (2 layers) |
| Fewer params | GRU instead of LSTM |
| Encode full document | Bidirectional GRU/LSTM |

Foster's recipe generator stays **unidirectional stacked LSTM** or GRU.

---

## Parameter and Speed Trade-offs

```python
for cell_type in [layers.LSTM, layers.GRU]:
    m = models.Sequential([
        layers.Embedding(10000, 100, input_length=200),
        cell_type(128, return_sequences=True),
        cell_type(128, return_sequences=True),
        layers.Dense(10000, activation="softmax"),
    ])
    m.summary()
```

Compare `Total params` and wall-clock per epoch on your hardware.

---

## Connection to PixelCNN

Image autoregression ([Section 5.6](./section-06-pixelcnn.md)) abandons recurrence for **masked convolutions** — all prefix pixels visible in parallel within a layer. RNNs serialize; CNNs parallelize per layer but still sample pixels sequentially at inference. Foster's text chapter uses LSTM for **discrete tokens**; the PixelCNN chapter applies the same autoregressive factorization to **spatial neighborhoods** on MNIST.

---

## GRU vs LSTM: Gate Equations

A **GRU** merges forget and input gates into an update gate $z_t$ and uses a reset gate $r_t$:

$$
z_t = \sigma(W_z [h_{t-1}, x_t]), \quad r_t = \sigma(W_r [h_{t-1}, x_t])
$$

$$
\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t]), \quad h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$
> **Readable form:** The GRU blends previous state $h_{t-1}$ with a candidate $\tilde{h}_t$ using gate $z_t$. Fewer parameters than LSTM → faster training on small corpora like Foster's recipe dataset.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Bidirectional for generation | Cheating future tokens | Unidirectional only |
| Stacked LSTM without `return_sequences` on layer 1 | Shape error | `True` on all but last if needed |
| GRU/LSTM mismatch in resume | Load failure | Match cell type when reloading |
| Too deep stack | Vanishing signal | 2 layers usually enough |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Stacked RNN** | Multiple recurrent layers depth-wise |
| **GRU** | Gated recurrent unit — simplified LSTM |
| **Bidirectional** | Forward + backward passes concatenated |
| **Hidden state** | $h_t$ carrying temporal summary |
| **return_sequences** | Output per-timestep for stacking |

---

## Reflection Questions

1. Why is Bidirectional inappropriate for recipe generation?
2. What is the output dimension of `Bidirectional(GRU(100))`?
3. When might a GRU outperform an LSTM at equal unit count?
4. Why must the first stacked LSTM use `return_sequences=True`?

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

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 5 — RNN Extensions. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder (GRU).

---

**Previous:** [Section 5.4 — Training & Sampling Text](./section-04-training-and-sampling-text.md)  
**Next:** [Section 5.6 — PixelCNN](./section-06-pixelcnn.md)
