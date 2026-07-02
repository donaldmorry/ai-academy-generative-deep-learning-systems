# Section 13.2: DALL·E 2 Architecture

> **Source inheritance:** Foster, Ch. 13 — "DALL·E 2 Architecture"
> **Enhanced with:** Multimodal Models in TensorFlow/Keras; connections to modern generative AI
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Foster presents DALL-E 2 as a three-stage text-to-image pipeline: a text encoder turns the prompt into a CLIP text embedding, a prior predicts a matching CLIP image embedding, and a decoder generates the image while conditioning on both the original text and the predicted image embedding.

This split matters. DALL-E 2 does not ask one network to learn language understanding, image-language alignment, and pixel synthesis from scratch. It uses CLIP as a pretrained semantic bridge, then trains generative components around that bridge.

> **Readable form:** DALL-E 2 approximates a conditional generation chain: $t \rightarrow e_t \rightarrow e_i \rightarrow x$, where $t$ is text, $e_t$ is a CLIP text embedding, $e_i$ is a CLIP image embedding, and $x$ is the generated image.

---

## Text Encoder

The text encoder's job is to convert a discrete prompt into a continuous vector that captures the prompt's conceptual meaning. Foster emphasizes that DALL-E 2 uses CLIP rather than training this encoder from scratch. The text encoder is therefore already trained to place language near matching visual evidence.

A good way to study this component is to ask what information a single prompt embedding can plausibly preserve:

| Prompt property | Why it matters downstream |
|-----------------|---------------------------|
| Objects | The decoder must synthesize the requested entities |
| Attributes | Colors, materials, style, and size must attach to the right objects |
| Relations | Spatial and semantic relations drive composition |
| Style | The generator must change texture, lighting, and rendering conventions |
| Rare combinations | The model must combine concepts not seen together in training |

When a DALL-E 2-style model produces a visually strong but semantically wrong image, inspect whether the text embedding and bridge preserve the relevant relation.

---

## CLIP Embeddings

CLIP supplies two aligned embedding functions: one for text and one for images. DALL-E 2 uses the text side first, then learns to predict a plausible image-side embedding. This is the point where the pipeline crosses from language space into image space.

The image embedding is not a picture. It is a semantic target that says what kind of image should exist. The decoder still has to turn that target into pixels. This separation explains why DALL-E 2 can support image variations: an input image can be embedded by CLIP's image encoder, then decoded into related samples.

---

## Prior

The prior learns the bridge $P(e_i \mid e_t)$ from CLIP text embeddings to CLIP image embeddings. Foster describes two approaches tried by the DALL-E 2 authors:

| Prior type | Mechanism | Practical implication |
|------------|-----------|-----------------------|
| Autoregressive prior | Generates the image embedding sequentially with a Transformer | Conceptually familiar from language models, but slower because output elements are generated in order |
| Diffusion prior | Starts from noised image embeddings and denoises while conditioning on text embeddings | Reported as stronger and more efficient in the chapter's account |

The prior is easy to overlook because it does not directly produce pixels. It is nevertheless the second half of the semantic bridge: CLIP gives aligned spaces, and the prior learns how to move from the text point to the image point.

---

## Decoder

The decoder is the visual generator. Foster connects it to GLIDE: a diffusion model with a U-Net denoiser and a Transformer text encoder. DALL-E 2 adds the predicted CLIP image embedding as an additional conditioning signal.

The decoder first produces a low-resolution image and then uses diffusion upsamplers to reach high resolution. This keeps the most expensive generative work focused on smaller images until the final stages.

| Decoder stage | Source-grounded role |
|---------------|----------------------|
| Initial denoising | Generate a 64 x 64 image from noise conditioned on text and image embedding |
| First upsampler | Increase resolution while preserving the generated content |
| Second upsampler | Produce the final high-resolution image |
| Conditioning path | Combine text prompt information with the predicted CLIP image embedding |

The architectural section is modularity: each part can be studied as a separate learned transformation, but final quality depends on the interfaces between them.

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

Foster's Chapter 13 notebooks follow a consistent Keras workflow for **dall·e 2 architecture**:

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

Follow Foster's Chapter 13 notebook for **dall·e 2 architecture** step by step:

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
| **Text encoder** | A model that maps the prompt into a continuous embedding used by downstream generative components. |
| **CLIP text embedding** | The language-side vector produced by CLIP, trained to align with matching image embeddings. |
| **CLIP image embedding** | The vision-side vector that acts as the semantic image target for DALL-E 2's prior and decoder. |
| **Prior** | The learned model that maps from a CLIP text embedding to a plausible CLIP image embedding. |
| **Decoder** | The diffusion-based image generator that turns conditioning information into pixels. |
| **Upsampler** | A diffusion model that increases image resolution after a lower-resolution image has been generated. |

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

A DALL-E 2 architecture review should follow the data path, not the brand name:

1. Prompt enters the CLIP text encoder.
2. CLIP text embedding is passed to the prior.
3. Prior predicts a CLIP image embedding.
4. Decoder denoises image noise while conditioning on text and image embedding.
5. Upsamplers increase resolution.

### Architecture Audit

| If you see this failure | Inspect this component first |
|-------------------------|------------------------------|
| Objects missing from prompt | Text encoder or prior |
| Attributes attached to wrong object | Prior and decoder conditioning |
| Correct composition but blurry detail | Decoder or upsamplers |
| Good image but wrong text rendering | CLIP embedding bottleneck and decoder training data |
| Low diversity across samples | Prior sampling and decoder randomness |

### Study Drill

Recreate the pipeline using boxes and arrows. Then annotate each arrow with the representation type: tokens, CLIP text embedding, CLIP image embedding, noisy image tensor, low-resolution image, high-resolution image.

### Representation Boundary Drill

For each DALL-E 2 boundary, write one invariant that should be preserved:

| Boundary | Invariant to preserve |
|----------|-----------------------|
| Prompt to CLIP text embedding | The requested objects, attributes, style, and relations remain distinguishable |
| Text embedding to image embedding | The visual concept implied by the prompt stays aligned with the language concept |
| Image embedding to denoising decoder | The generated image follows the semantic target rather than drifting to a common training pattern |
| Low-resolution output to upsamplers | Composition is preserved while detail increases |

Then choose one prompt with two interacting objects. Swap an attribute between the objects and explain which boundary would need to preserve that swap. This directly targets the attribute-binding issue Foster discusses later in the DALL-E 2 section.

### Source Connection

Foster contrasts DALL-E 2 with GLIDE to show why CLIP latents matter. GLIDE trains text-to-image diffusion directly from the prompt, while DALL-E 2 carries forward a predicted CLIP image embedding. In your notes, make that difference explicit: GLIDE has text conditioning; DALL-E 2 has text conditioning plus a semantic image target predicted by the prior.

This distinction also explains why the prior is worth studying separately. If the prior predicts an embedding for a plausible but wrong image, the decoder can produce a polished failure. The generated sample may look strong while missing the prompt relationship that the bridge lost.

### Quick Comparison Prompts

- Prompt A: a red cube above a blue cube.
- Prompt B: a blue cube above a red cube.
- Prompt C: a cube with the words deep learning printed on it.
- Prompt D: the same cube rendered as a pencil sketch.

Use A and B to test relation and attribute binding. Use C to test text rendering. Use D to test style transfer while preserving object identity.
**Previous:** [Section 13.1](./section-01-multimodal-introduction.md)
**Next:** [Section 13.3](./section-03-clip-deep-dive.md)
