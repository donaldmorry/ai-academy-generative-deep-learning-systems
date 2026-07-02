# Section 13.6: Stable Diffusion

> **Source inheritance:** Foster, Ch. 13 — "Stable Diffusion"
> **Enhanced with:** Multimodal Models in TensorFlow/Keras; connections to modern generative AI
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Stable Diffusion is Foster's example of how latent diffusion changes the economics of text-to-image generation. DALL-E 2 and Imagen denoise image representations closer to pixel space; Stable Diffusion wraps diffusion inside an autoencoder so the denoising U-Net works in a compressed latent space.

The chapter also emphasizes availability: Stable Diffusion's code and weights were released publicly through the open-source ecosystem, allowing users to run and adapt the model outside a proprietary API.

> **Readable form:** Stable Diffusion approximates $P_\theta(z \mid c)$ in autoencoder latent space, then decodes $z$ back to an image, where $c$ is the text conditioning signal.

---

## Latent Diffusion

Latent diffusion moves the expensive denoising process from pixels to a lower-dimensional representation. An autoencoder first learns to encode images into latents and decode latents back to images. The diffusion model then learns to denoise those latents.

| Step | Representation | Model role |
|------|----------------|------------|
| Encode training image | Latent vector or latent feature map | Autoencoder compresses image detail |
| Add noise | Noisy latent | Forward diffusion process |
| Denoise | Less-noisy latent | U-Net predicts the denoising update |
| Decode | Image | Autoencoder decoder reconstructs pixels |

The payoff is speed and memory efficiency. The risk is that the autoencoder bottleneck can lose information before diffusion ever begins.

---

## VAE / Autoencoder Role

Foster describes Stable Diffusion as wrapping diffusion with an autoencoder. In course terms, this connects Chapter 03's latent-space thinking with Chapter 08's denoising objective. The autoencoder handles high-resolution detail; the diffusion model handles generative sampling in the compressed space.

When inspecting outputs, separate these failures:

| Symptom | Possible source |
|---------|-----------------|
| Fine textures look smeared | Autoencoder decode bottleneck |
| Composition ignores prompt | Text conditioning or guidance |
| Object shapes are unstable | Denoising trajectory or insufficient steps |
| Output changes too little across seeds | Sampling settings or guidance too high |

---

## U-Net Denoiser

The denoising U-Net receives noisy latents, a timestep, and text conditioning. The U-Net architecture is well suited because it preserves spatial structure through skip connections while processing features at multiple resolutions.

Stable Diffusion's U-Net is lighter than a pixel-space U-Net would be for the same output resolution, because its spatial grid is the autoencoder latent grid rather than the full image grid.

---

## Text Conditioning and CLIP

The first version of Stable Diffusion used a pretrained CLIP text model for conditioning, while Stable Diffusion 2 used OpenCLIP. The practical section is the same as in DALL-E 2: text enters through an embedding model, and the diffusion process is steered by that conditioning.

Prompt engineering works because the text encoder and denoiser jointly define a conditional distribution. Changing adjectives, style phrases, negative prompts, or guidance scale changes the denoising path, not just a label attached after generation.

---

## Open-Source Ecosystem

Foster singles out public weights and code as a major difference from DALL-E 2 and Imagen. Open release changes the learning experience: students can inspect pipelines, swap schedulers, test prompts, fine-tune adapters, and measure hardware trade-offs.

| Practical lever | What it changes |
|-----------------|-----------------|
| Number of denoising steps | Speed versus detail and stability |
| Guidance scale | Prompt adherence versus oversaturation or artifacts |
| Sampler/scheduler | The denoising trajectory through latent space |
| Seed | Stochastic variation under the same prompt |
| Autoencoder | Reconstruction detail and compression artifacts |

Stable Diffusion is therefore both a model and a laboratory for studying conditional diffusion systems.

---

## Mathematical Foundation

$$
\hat{x} = \text{U-Net}(z_t, t, c)
$$
> **Readable form:** Stable Diffusion denoises VAE latents conditioned on text c.


$$
\mathcal{L}_{\text{CLIP}} = -\log\frac{\exp(s(i,t)/\tau)}{\sum_j \exp(s(i,t_j)/\tau)}
$$
> **Readable form:** CLIP contrastive loss pulls matched image-text pairs together in embedding space.


---

## TensorFlow / Keras Implementation

Foster's Chapter 13 notebooks follow a consistent Keras workflow for **stable diffusion**:

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

Follow Foster's Chapter 13 notebook for **stable diffusion** step by step:

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
| **Stable Diffusion** | An open text-to-image latent diffusion model released with public code and weights. |
| **Latent diffusion** | Diffusion performed in an autoencoder latent space rather than directly in pixel space. |
| **Autoencoder** | The encoder-decoder pair that compresses images to latents and reconstructs images from latents. |
| **Denoising U-Net** | The neural network that predicts how to remove noise from latent variables at each timestep. |
| **Text conditioning** | The mechanism that steers denoising using prompt embeddings from a text encoder. |
| **Guidance scale** | An inference setting that adjusts how strongly the denoising process follows the prompt. |

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

Stable Diffusion is the clearest Chapter 13 example of representation choice changing feasibility. Pixel-space diffusion is powerful but expensive. Latent-space diffusion keeps the generative process conceptually similar while making training and inference more practical.

### Pipeline Audit

- Identify the text encoder.
- Identify the autoencoder encoder and decoder.
- Identify the denoising U-Net.
- Identify the scheduler or sampler.
- Record prompt, negative prompt, seed, steps, guidance scale, image size, and model version.

### Diagnostic Drill

Run the same prompt with three seeds and the same seed with three guidance scales. Label each output for alignment and fidelity. The goal is to see that prompt adherence, diversity, and visual quality are coupled but not identical.

### Latent-Space Trace

Write a trace for one generated image using Foster's Stable Diffusion description:

| Trace step | Question to answer |
|------------|--------------------|
| Text encoding | What prompt tokens or phrases should dominate the conditioning vector? |
| Initial latent noise | What is randomized by the seed before denoising begins? |
| Denoising U-Net | How do timestep and text conditioning steer each update? |
| Scheduler | How many updates are applied and how large are they? |
| Autoencoder decoder | What visual detail must be reconstructed from the final latent? |

This trace helps separate three common confusions. The prompt is not drawn directly. The U-Net does not operate on final pixels. The VAE decoder does not decide the prompt semantics; it reconstructs an image from the latent that diffusion produced.

### Source Connection

Foster frames Stable Diffusion as a practical break from pixel-space diffusion. The autoencoder does heavy representational compression, which lets the denoising model stay smaller and faster. That choice also introduces a dependency: if the latent representation cannot preserve a fine visual feature, more denoising steps cannot fully recover it.

When writing lab notes, compare this with Imagen's cascaded diffusion and DALL-E 2's decoder. All three use diffusion, but they place the denoising work at different representation boundaries. That is the architectural fact to remember.

### Quick Experiment Prompts

- A prompt with a simple object and style.
- A prompt with two objects and a spatial relation.
- A prompt that asks for readable text.
- A prompt that requires a rare combination of style and subject.

For each prompt, record whether the first failure is alignment, fidelity, or latent reconstruction detail. This keeps the analysis tied to Foster's distinction between prompt following, image realism, and representation choice.
**Previous:** [Section 13.5](./section-05-imagen.md)
**Next:** [Section 13.7](./section-07-flamingo.md)
