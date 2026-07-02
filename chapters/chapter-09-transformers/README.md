# Chapter 09: Transformers

> **Source:** *Generative Deep Learning (2nd ed.) — David Foster*, Chapter 9
> **Part:** Part III — Applications
> **Estimated time:** 12–14 hours
> **Prerequisites:** Course 4, Chapter 05 — autoregressive modeling and text tokenization; Course 3, Chapter 12 — attention and sequence-to-sequence models

---

## Chapter Overview

Transformers replaced RNNs as the dominant sequence architecture and power modern LLMs. You will build GPT from scratch in Keras on a wine reviews dataset: self-attention, queries/keys/values, multihead attention, causal masking, transformer blocks, and positional encoding. Training and analysis of your GPT model precede a survey of T5, GPT-3/4, and ChatGPT. This chapter bridges autoregressive LSTMs (Chapter 05) to multimodal systems (Chapter 13) that use transformers for both text and vision.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain self-attention: queries, keys, values, and scaled dot-product attention
2. Implement multihead attention and causal masking for autoregressive language modeling
3. Build a complete GPT model with transformer blocks and positional encoding in Keras
4. Train GPT on the Wine Reviews dataset and generate coherent review text
5. Analyze attention patterns and generation quality at varying temperatures
6. Describe architectural differences between GPT, T5, and encoder-decoder transformers
7. Summarize the scaling laws and alignment techniques behind GPT-3/4 and ChatGPT

---

## Sections

| # | Section | Topics |
|---|--------|--------|
| 9.1 | [Transformer Introduction](./section-01-transformer-introduction.md) | Attention is all you need; GPT overview; use cases |
| 9.2 | [Attention Mechanism](./section-02-attention-mechanism.md) | Q, K, V; scaled dot-product; attention weights interpretation |
| 9.3 | [Multihead Attention](./section-03-multihead-attention.md) | Parallel heads; projection layers; Keras MultiHeadAttention |
| 9.4 | [Causal Masking](./section-04-causal-masking.md) | Autoregressive constraint; look-ahead prevention; masking implementation |
| 9.5 | [Transformer Block](./section-05-transformer-block.md) | Layer norm; feed-forward; residual connections; stacking blocks |
| 9.6 | [Positional Encoding](./section-06-positional-encoding.md) | Sinusoidal encoding; learned positions; sequence order |
| 9.7 | [Training GPT](./section-07-training-gpt.md) | Wine reviews data; tokenization; training loop; generation |
| 9.8 | [GPT-3, GPT-4 & ChatGPT](./section-08-gpt-3-gpt-4-and-chatgpt.md) | Scaling; few-shot learning; RLHF; capabilities and limits |
| 9.9 | [T5 & Other Transformers](./section-09-t5-and-other-transformers.md) | Encoder-decoder; text-to-text; architecture comparison |

---

## Lab / Project

See also: [Lab 09](./section-lab-09-build-and-train-gpt-on-wine-reviews.md)

**Lab 09: Build and Train GPT on Wine Reviews**

1. Tokenize the Wine Reviews dataset; create input/target sequence pairs.
2. Implement a GPT model with 4+ transformer blocks in Keras.
3. Train with causal masking; plot loss and generate 10 novel reviews.
4. Visualize attention weights for a chosen input sequence.
5. *Deliverable:* Trained GPT checkpoint, generated reviews, and attention heatmap.

---

## Connections to Other Courses

| Topic in this chapter | Where it deepens |
|---------------------|------------------|
| Attention mechanism | Seq2seq, neural MT (Course 3, Ch 12) |
| Autoregressive text generation | LSTM language models (Course 4, Chapter 05) |
| Large language models | NLP applications (Course 1, Chapter 13) |

---

## Prerequisites

- Course 4, Chapter 05 — autoregressive modeling and text tokenization
- Course 3, Chapter 12 — attention and sequence-to-sequence models

---

## Self-Assessment

1. How do queries, keys, and values interact in scaled dot-product attention?
2. Why is causal masking necessary for GPT-style language models?
3. What components make up a transformer block?
4. How does positional encoding provide sequence order information?
5. What architectural difference separates GPT from T5?
6. What is RLHF, and why was it important for ChatGPT?
7. How does transformer-based generation compare to your Chapter 05 LSTM?

---

**Previous:** [Chapter 08 — Diffusion Models](../chapter-08-diffusion-models/README.md)
**Next:** [Chapter 10 — Advanced GANs](../chapter-10-advanced-gans/README.md)
