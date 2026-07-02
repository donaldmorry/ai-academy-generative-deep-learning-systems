# Section 13.7: Flamingo

> **Source inheritance:** Foster, Ch. 13 — "Flamingo"
> **Enhanced with:** Multimodal Models in TensorFlow/Keras; connections to modern generative AI
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Flamingo changes the direction of the multimodal problem. DALL-E 2, Imagen, and Stable Diffusion generate images from text. Flamingo accepts interleaved text and visual data, then generates text continuations. Foster presents it as a visual language model rather than a text-to-image model.

The architectural trick is to connect strong pretrained vision and language systems without retraining everything end to end. Flamingo freezes major components and trains adapters that let the language model attend to compact visual tokens.

> **Readable form:** Flamingo models $P_\theta(\text{text continuation} \mid \text{text context}, \text{images/videos})$ by converting visual evidence into fixed-size tokens that a language model can cross-attend to.

---

## Vision Encoder

Foster describes the Vision Encoder as the component that converts images or video frames into visual feature grids. Flamingo uses a pretrained NFNet-F6 rather than the Vision Transformer used by CLIP's image encoder. The encoder is trained with a CLIP-like contrastive objective and then frozen.

For video, frames are sampled and encoded separately, then temporal encodings are added before features are flattened and passed onward. This makes videos look like a sequence of visual evidence rather than a single static image.

| Input type | Vision processing |
|------------|-------------------|
| Image | Encode to a 2D feature grid, then flatten |
| Video | Sample frames, encode each frame, add temporal information, concatenate |
| Interleaved prompt | Keep track of which visual input belongs to which text chunk |

---

## Perceiver Resampler

The raw visual features can be long and variable length. Foster's explanation of the Perceiver Resampler focuses on memory: ordinary Transformer attention becomes expensive as sequence length grows, so Flamingo compresses visual features into a fixed number of latent tokens.

The Perceiver Resampler uses cross-attention from learned latent queries to the visual features. The result is a compact visual representation that can be passed to the language model regardless of image resolution or number of video frames.

| Problem | Perceiver Resampler response |
|---------|------------------------------|
| Variable image sizes | Produce fixed-length latent visual tokens |
| Long video feature sequences | Attend through compact latents instead of full self-attention |
| Language model interface | Provide a small visual token set for cross-attention |
| Memory pressure | Avoid quadratic attention over all raw visual features |

---

## Few-Shot Multimodal Language Model

Flamingo's language model is based on a pretrained Chinchilla-style decoder Transformer. Foster highlights gated cross-attention layers as the mechanism that lets visual information enter without destroying the frozen language model's existing capabilities.

The gates are initialized so the visual pathway contributes nothing at the start of training. Training then learns how much visual information to blend into the language stream.

Flamingo also structures prompts into chunks. Image markers and end-of-chunk markers tell the model which text should attend to which visual input. Masked cross-attention prevents text from attending to unrelated images in other chunks.

---

## Capabilities and Boundaries

Foster's examples emphasize visual dialogue, conversational prompting, image understanding, and video understanding. The model can be prompted with a few demonstrations and then asked to continue in the same pattern.

| Capability | What to inspect |
|------------|-----------------|
| Visual question answering | Does the answer depend on visible evidence or language priors? |
| Video reasoning | Does the model track temporal consequence, not just objects? |
| Few-shot adaptation | Does performance change when examples are reordered or removed? |
| Dialogue | Does the model maintain context while grounding claims in images? |

Flamingo points toward general multimodal assistants, but it also raises the central evaluation issue: fluent language can hide weak visual grounding.

---

## Mathematical Foundation

$$
\mathcal{L}_{\text{CLIP}} = -\log\frac{\exp(s(i,t)/\tau)}{\sum_j \exp(s(i,t_j)/\tau)}
$$
> **Readable form:** CLIP contrastive loss pulls matched image-text pairs together in embedding space.


$$
\hat{x} = \text{U-Net}(z_t, t, c)
$$
> **Readable form:** Stable Diffusion denoises VAE latents conditioned on text c.


---

## TensorFlow / Keras Implementation

Foster's Chapter 13 notebooks follow a consistent Keras workflow for **flamingo**:

```python
import tensorflow as tf

def clip_contrastive_loss(image_emb, text_emb, temperature=0.07):
    logits = tf.matmul(image_emb, text_emb, transpose_b=True) / temperature
    labels = tf.range(tf.shape(logits)[0])
    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss_t = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=tf.transpose(logits))
    return (tf.reduce_mean(loss_i) + tf.reduce_mean(loss_t)) / 2
```

> **Readable form:** Define tensors with explicit shapes, build layers with the Functional API, compile with an optimizer and loss matched to your generative objective, then `fit` on data with validation curves and checkpointing.

---

## Worked Example

Follow Foster's Chapter 13 notebook for **flamingo** step by step:

1. **Data** — Load the chapter dataset; inspect shapes and preprocessing.
2. **Model** — Build the architecture Foster specifies; print `model.summary()`.
3. **Train** — Run enough epochs to see loss decrease; save checkpoints.
4. **Generate** — Sample from the trained model; inspect outputs qualitatively.
5. **Analyze** — Plot diagnostics: attention maps, latent grids, FID curves, or reward.

---

## Design Trade-offs

| Choice | Benefit | Cost |
|--------|---------|------|
| Larger model capacity | Richer distributions, sharper samples | Memory, training time |
| More training data | Better generalization | Data collection, compute |
| Simpler architecture first | Faster debugging | May underfit complex data |
| Strong regularization | Stability, less overfitting | Blurrier or less diverse outputs |

---

## Connections to Other Courses

- **Diffusion models** → DDPM training (Course 4, Chapter 08)
- **CLIP & transformers** → GPT and attention (Course 4, Chapter 09)
- **VAE latent spaces** → Compression for latent diffusion (Course 4, Chapter 03)

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Flamingo** | A DeepMind visual language model that generates text from interleaved visual and textual input. |
| **Vision Encoder** | The pretrained visual feature extractor used to encode images or video frames. |
| **NFNet-F6** | The Normalizer-Free ResNet variant Foster identifies as Flamingo's vision backbone. |
| **Perceiver Resampler** | A chapter that compresses variable-length visual features into fixed-length latent visual tokens. |
| **Gated cross-attention** | Trainable layers that let the language model attend to visual tokens while preserving frozen language layers. |
| **Few-shot multimodal prompting** | Conditioning the model with a small set of image/text examples before asking for a new response. |

---

## Reflection Questions

1. What role does CLIP play in DALL·E 2 and Stable Diffusion?
2. How does Stable Diffusion's latent diffusion differ from pixel-space DDPM?
3. What are the three main stages of the DALL·E 2 pipeline?
4. How does Imagen use cascaded diffusion to improve resolution?
5. What is Flamingo's Perceiver resampler, and why is it needed?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). O'Reilly. Ch. 13. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Foster's codebase: [https://github.com/davidADSP/GDL_code](https://github.com/davidADSP/GDL_code)
- TensorFlow Keras: [https://www.tensorflow.org/guide/keras](https://www.tensorflow.org/guide/keras)

---

## Source-Grounded Review

Flamingo should be reviewed as an interface design between frozen models. The vision side produces evidence, the Perceiver compresses it, and the language model learns when to consult it.

### Grounding Audit

- Remove the image and see whether the answer changes.
- Change the question while keeping the image fixed.
- Swap image order in a multi-image prompt.
- Ask for evidence: which visible region supports the answer?
- Compare a fluent answer with the actual visual content.

### Architecture Drill

Draw the path from an image to a generated answer: vision encoder, flattened features, Perceiver Resampler, visual tokens, gated cross-attention, language model output. Mark which parts are frozen and which parts are trained.
**Previous:** [Section 13.6](./section-06-stable-diffusion.md)
**Next:** [Section 13.8](./section-08-comparison-and-synthesis.md)

---

## Assessment Practice

Use the shared [Assessment Appendix](../../ASSESSMENT_APPENDIX.md) for concept audits, worked examples, implementation checks, experiment logs, oral-exam prompts, and deliverable checklists.
