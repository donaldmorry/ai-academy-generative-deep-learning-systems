# Section 5.2: LSTM for Text

> **Source inheritance:** Foster, Ch. 5 — "The Recipes Dataset" & tokenization  
> **Enhanced with:** Epicurious corpus, TextVectorization, padded sequences, training set construction  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Foster's LSTM text generator starts with **20,000+ Epicurious recipes** — unstructured prose with titles, directions, and culinary vocabulary. Before any [neural network](../../GLOSSARY.md#neural-network) sees data, you must **tokenize**: split text into discrete units, build a vocabulary, and create input/target sequence pairs. Text is not pixels — you cannot backprop through raw strings; integer token IDs bridge language to tensors.

This section covers download through `TextVectorization`, producing padded integer sequences ready for [Section 5.3](./section-03-embeddings-and-lstm-layers.md).

---

## The Recipes Dataset

```bash
bash scripts/download_kaggle_data.sh hugodarwood epirecipes
```

```python
import json
import re
import string
import tensorflow as tf
from tensorflow.keras import layers

with open("/app/data/epirecipes/full_format_recipes.json") as f:
    recipe_data = json.load(f)

filtered_data = [
    "Recipe for " + x["title"] + " | " + " ".join(x["directions"])
    for x in recipe_data
    if x.get("title") and x.get("directions")
]
```

Foster's Example 5-3 shows rich procedural text — whisking, chilling, layering — ideal for testing whether a model captures **style** not just vocabulary.

---

## Text vs Image Data

Foster lists four structural differences:

| Property | Text | Images |
|----------|------|--------|
| Units | Discrete tokens | Continuous pixels |
| Dimensions | 1D time | 2D space |
| Sensitivity | One wrong word breaks meaning | Robust to pixel noise |
| Rules | Grammar / semantics | None explicit |

These motivate **embedding layers** and **sequential** models rather than plain Conv2D on characters.

---

## Tokenization Pipeline

**Word-level** tokenization (Foster's choice): lowercase, punctuation as separate tokens, 10K vocabulary.

```python
def pad_punctuation(s):
    s = re.sub(f"([{string.punctuation}])", r" \1 ", s)
    s = re.sub(" +", " ", s)
    return s

text_data = [pad_punctuation(x) for x in filtered_data]
text_ds = tf.data.Dataset.from_tensor_slices(text_data).batch(32).shuffle(1000)

vectorize_layer = layers.TextVectorization(
    standardize="lower",
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=201,  # 200 input + 1 target
)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()
```

| Token ID | Meaning |
|----------|---------|
| 0 | Padding / stop |
| 1 | `<UNK>` — outside top 10K |
| 2+ | Frequent words by rank |

Example 5-6: token 2 is `.`, token 7 is `the` — punctuation and function words dominate early ranks.

---

## Building Input / Target Pairs

Autoregressive training: predict token $t+1$ from tokens $1..t$.

```python
def prepare_inputs(text_batch):
    tokenized = vectorize_layer(text_batch)
    x = tokenized[:, :-1]   # first 200 tokens
    y = tokenized[:, 1:]    # shifted by one
    return x, y

train_ds = text_ds.map(prepare_inputs)
```

> **Readable form:** Input is prefix; target is next token at every position.

---

## Sequence Length Trade-offs

| `output_sequence_length` | Effect |
|--------------------------|--------|
| Short (50) | Faster, less context |
| 200 (Foster) | Captures paragraph structure |
| 500+ | Memory heavy; diminishing returns |

Padding with zeros (stop token) — the model sees trailing zeros; the LSTM processes full length unless masking is applied.

---

## Vocabulary Size

`max_tokens=10000` balances coverage vs model size:

- Larger vocab → fewer `<UNK>` tokens → bigger softmax ($|V| \times d$)
- Smaller vocab → more unknown ingredients (e.g., "persillade" → UNK)

```python
print("Vocab size:", len(vocab))
print("Sample tokens:", vocab[2:12])
```

---

## tf.data Pipeline

```python
train_ds = (
    tf.data.Dataset.from_tensor_slices(text_data)
    .shuffle(10000)
    .batch(32)
    .map(lambda t: prepare_inputs(vectorize_layer(t)))
    .prefetch(tf.data.AUTOTUNE)
)
```

Streaming batches avoids materializing all tokenized recipes in RAM.

---

## Word vs Character Tokenization

| | Words | Characters |
|---|-------|------------|
| Vocab size | Large (10K+) | Small (~100) |
| OOV handling | UNK token | Can spell new words |
| Sequence length | Shorter | Much longer |
| Coherence | Better at word level | Harder grammar |

Foster chooses words for recipe coherence — ingredients and verbs must be exact tokens.

---

## Connection to Autoregressive Framework

[Section 5.1](./section-01-autoregressive-framework.md) chain rule:

$$
P(\text{recipe}) = \prod_i P(w_i \mid w_{<i})
$$
> **Readable form:** The joint probability is built by multiplying the conditional factors specified by the model.

Each column in `y` supervises one conditional — full recipe likelihood sums log-probabilities across positions in training.

---

## Foster's LSTM Prison Analogy

| Story element | Data/code counterpart |
|---------------|----------------------|
| New word posted daily | Next token in sequence |
| 256 prisoners | LSTM hidden units |
| Guard at door | Softmax over vocabulary |
| Edward's past stories | Training corpus |

Architecture details in [Section 5.3](./section-03-embeddings-and-lstm-layers.md).

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| No punctuation padding | `"salt,"` as one token | `pad_punctuation` |
| `adapt` on test data | Leakage | Adapt train only |
| Forgetting +1 length | Missing final target | `sequence_length = N+1` |
| Empty recipes in JSON | Crashes | Filter null titles/directions |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Tokenization** | Splitting text into integer-coded units |
| **TextVectorization** | Keras layer building vocab + encoding |
| **Stop token** | ID 0 — padding marker |
| **UNK** | Out-of-vocabulary fallback token |
| **Shifted target** | Next-token autoregressive setup |

---

## Reflection Questions

1. Why can't we backpropagate directly on raw strings?
2. What does token ID 1 represent in Foster's vocabulary?
3. Why pad sequences to 201 when using 200-token input windows?
4. When would character tokenization outperform word tokenization?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 5 — Recipes & Tokenization. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Foster's codebase: `notebooks/05_autoregressive/01_lstm/lstm.ipynb`

---

**Previous:** [Section 5.1 — Autoregressive Framework](./section-01-autoregressive-framework.md)  
**Next:** [Section 5.3 — Embeddings & LSTM Layers](./section-03-embeddings-and-lstm-layers.md)



