# Section 5.4: Training & Sampling Text

> **Source inheritance:** Foster, Ch. 5 — "Training the LSTM" & text generation  
> **Enhanced with:** Teacher forcing, temperature sampling, exposure bias, recipe generation loop  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Foster trains the recipe LSTM for multiple epochs on next-token prediction, then **samples** new recipes autoregressively: seed with a prompt, read softmax at the final timestep, draw a token, append, repeat. **Temperature** rescales logits before sampling — low $T$ repeats safe phrases; high $T$ invents odd ingredient pairings. This section closes the text loop from [Sections 5.2–5.3](./section-03-embeddings-and-lstm-layers.md).

---

## Training Loop

```python
history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
    ],
)
```

Objective — maximize log-likelihood over all positions:

$$
\mathcal{L} = -\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T} \log P_\theta(w_{b,t} \mid w_{b,1}, \ldots, w_{b,t-1})
$$
> **Readable form:** Cross-entropy at every timestep, averaged over batch and sequence length.

`sparse_categorical_crossentropy` expects integer `y` from [Section 5.2](./section-02-lstm-for-text.md) `prepare_inputs`.

---

## Teacher Forcing

During training, position $t$ receives **ground-truth** tokens $w_{<t}$ in the input tensor. The model never sees its own mistakes during backprop — **teacher forcing**.

| Phase | Prefix source |
|-------|---------------|
| Training | True tokens from dataset |
| Sampling | Model's own prior predictions |

**Exposure bias:** errors compound at inference when the model drifts off-manifold. Foster accepts this for pedagogy; production systems use scheduled sampling or beam search.

---

## Autoregressive Sampling

```python
import numpy as np

def sample_from(probs, temperature=1.0):
    probs = np.asarray(probs, dtype=np.float64)
    logits = np.log(probs + 1e-8) / temperature
    exp_logits = np.exp(logits - np.max(logits))
    p = exp_logits / exp_logits.sum()
    return np.random.choice(len(p), p=p)

def generate_text(seed_text, max_len=200, temperature=1.0):
    tokens = vectorize_layer([seed_text]).numpy()[0].tolist()
    for _ in range(max_len):
        window = tokens[-200:]
        x = np.array([window])
        probs = model.predict(x, verbose=0)[0, len(window) - 1]
        next_id = sample_from(probs, temperature)
        if next_id == 0:
            break
        tokens.append(int(next_id))
    return " ".join(vocab[i] for i in tokens if i < len(vocab))
```

Seed with `"Recipe for lemon tart |"` to anchor format.

---

## Temperature

$$
P(w_i) \propto \exp\left(\frac{\log p_i}{T}\right)
$$
> **Readable form:** The expression assigns probability to the event or value using the stated model assumptions.

| Temperature | Effect |
|-------------|--------|
| $T = 0.5$ | Sharp — repetitive, "safe" recipes |
| $T = 1.0$ | Matches training distribution |
| $T = 1.5$ | Flat — creative, sometimes nonsensical |

Lab 05 deliverable: generate five recipes at each temperature and compare coherence.

---

## Greedy vs Stochastic Decoding

**Greedy:** `argmax` on probs — deterministic, often loops ("stir stir stir").

**Stochastic:** sample from softmax — diversity at cost of occasional UNK tokens.

```python
next_id_greedy = int(np.argmax(probs))
next_id_sample = sample_from(probs, temperature=1.0)
```

Foster uses stochastic sampling for creative generation; greedy is a useful debugging baseline.

---

## Monitoring Training Quality

| Signal | Interpretation |
|--------|----------------|
| `loss` decreasing | Model learning local n-gram patterns |
| `val_loss` rising | Memorizing training recipes |
| Sample at epoch 5 vs 10 | Subjective style match to Epicurious |

Always inspect **generated text** — cross-entropy can fall while samples remain gibberish if the model exploits dataset quirks.

---

## Foster's Generation Workflow

1. Train LSTM to convergence (or early stopping on `val_loss`)
2. Choose seed string matching training format (`Recipe for TITLE |`)
3. Run autoregressive loop until stop token or `max_len`
4. Repeat at temperatures 0.5, 1.0, 1.5 for Lab comparison
5. Document failure modes (infinite loops, UNK spam, no punctuation)

---

## Sequential Cost vs GANs/VAEs

| Model | Forward passes per sample |
|-------|---------------------------|
| LSTM text | $O(\text{sequence length})$ |
| VAE / GAN image | $O(1)$ |
| PixelCNN | $O(\text{pixels})$ |

Text LSTMs are fast enough for recipes; PixelCNN on images is painfully slow ([Section 5.6](./section-06-pixelcnn.md)).

---

## Connection to Chapter 09

Transformers replace LSTM recurrence with self-attention but keep **autoregressive** training and sampling. Temperature and exposure bias remain central to GPT-style models Foster covers in Chapter 9.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Wrong softmax index | Garbage words | Index last valid timestep in window |
| $T=0$ in log | NaN | Use `temperature=max(T, 1e-8)` |
| No seed prompt | Random genre | Prefix `"Recipe for ..."` |
| Greedy only | Loops | Stochastic sampling |
| Ignoring UNK token | `<UNK>` in output | Expand vocabulary or post-filter |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Teacher forcing** | Training with true prefix tokens |
| **Exposure bias** | Train/inference distribution mismatch |
| **Temperature** | Scales logits before softmax sampling |
| **Autoregressive sampling** | Iterative next-token generation |
| **Greedy decoding** | Always pick argmax token |

---

## Reflection Questions

1. Why does teacher forcing not apply during sampling?
2. How does temperature affect the entropy of the next-token distribution?
3. What happens if you always take `argmax` instead of sampling?
4. Why is exposure bias fundamental to one-step-ahead training?
5. How many forward passes does generating a 200-token recipe require?

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

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 5. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Foster's codebase: `notebooks/05_autoregressive/01_lstm/lstm.ipynb`
- Bengio, S. et al. (2015). Scheduled Sampling for Sequence Prediction with RNNs.

---

**Previous:** [Section 5.3 — Embeddings & LSTM Layers](./section-03-embeddings-and-lstm-layers.md)  
**Next:** [Section 5.5 — RNN Extensions](./section-05-rnn-extensions.md)
