# Lab 13: Stable Diffusion Pipeline Exploration

> **Prerequisites:** Sections 13.1-13.8
> **Estimated time:** 12-14 hours
> **Tools:** Python 3.10+, Jupyter, TensorFlow 2.x, Foster GDL_code
> **Glossary:** [GLOSSARY.md](../../GLOSSARY.md) | **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Lab Objectives

1. Run Stable Diffusion inference via Hugging Face Diffusers (or book codebase).
2. Map each pipeline component (VAE, U-Net, text encoder) to course chapters.
3. Generate images with varied prompts; test prompt engineering techniques.
4. Compare outputs from different guidance scales and step counts.
5. Sketch architecture diagrams for DALL·E 2 vs Stable Diffusion.
6. *Deliverable:* Prompt gallery, component mapping document, and architecture sketches.

### Math You'll Use

$$
\mathcal{L}_{\text{CLIP}} = -\log\frac{\exp(s(i,t)/\tau)}{\sum_j \exp(s(i,t_j)/\tau)}
$$
> **Readable form:** CLIP contrastive loss aligns image-text pairs.


$$
x = \text{Decoder}(\text{Prior}(\text{TextEmb}(y)))
$$
> **Readable form:** DALL-E 2: text embedding to prior to decoder.


---

1. Run Stable Diffusion inference via Hugging Face Diffusers (or book codebase).
2. Map each pipeline component (VAE, U-Net, text encoder) to course chapters.
3. Generate images with varied prompts; test prompt engineering techniques.
4. Compare outputs from different guidance scales and step counts.
5. Sketch architecture diagrams for DALL·E 2 vs Stable Diffusion.
6. *Deliverable:* Prompt gallery, component mapping document, and architecture sketches.

---

## Part A: Pipeline Implementation

Foster's Chapter 13 lab should be an inference and architecture exploration, not a generic dense-model training run. Use a Stable Diffusion pipeline if your machine has the required dependencies and model access. If not, document the pipeline with pseudocode and run the lightweight component-mapping tasks in Parts B-D.

```python
import torch
from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=True)

generator = torch.Generator(device=device).manual_seed(42)

prompt = "a small robot sketching a watercolor city skyline, detailed, soft light"
negative_prompt = "blurry, distorted, unreadable text"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=generator,
).images[0]

image.save("lab_13_stable_diffusion_seed42.png")
```

**Task:** Generate at least six images from three prompts and two seeds. Save the prompt, seed, guidance scale, inference steps, image size, scheduler, and model identifier for every sample.

### Component Map

| Stable Diffusion component | Foster Chapter 13 role | Earlier course connection |
|----------------------------|------------------------|---------------------------|
| Text encoder | Converts prompt text into conditioning embeddings | Transformers and CLIP-style representation learning |
| U-Net denoiser | Iteratively removes noise from latent variables | Chapter 08 diffusion models |
| VAE / autoencoder | Encodes and decodes between pixels and latent space | Chapter 03 VAEs and latent spaces |
| Scheduler | Defines the denoising trajectory at inference | Diffusion sampling algorithms |
| Guidance scale | Controls prompt-conditioning strength | Conditional generation and classifier-free guidance |

---

## Part B: Evaluation & Diagnostics

Create a small prompt study that separates alignment from fidelity, following the DrawBench mindset Foster discusses for Imagen.

```python
import pandas as pd

runs = [
    {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": 42,
        "steps": 30,
        "guidance_scale": 7.5,
        "model_id": model_id,
        "alignment_1_to_5": None,
        "fidelity_1_to_5": None,
        "failure_note": "",
    }
]

pd.DataFrame(runs).to_csv("lab_13_prompt_log.csv", index=False)
```

Score each sample twice:

| Score | Question |
|-------|----------|
| Alignment | Does the image follow the requested objects, attributes, relations, and style? |
| Fidelity | Does the image look coherent, detailed, and visually plausible? |

**Deliverable:** A prompt gallery with at least six saved images and a CSV or markdown table containing your scores and metadata.

---

## Part C: Written Reflection

Answer in a markdown cell or separate document:

1. Which Stable Diffusion component corresponds to Foster's latent diffusion explanation?
2. How does latent diffusion differ from DALL-E 2's GLIDE-style decoder operating closer to image space?
3. What changed when you varied guidance scale or inference steps?
4. Which sample had high fidelity but weak alignment, and what prompt detail did it miss?
5. How would the lab change if you used an Imagen-style cascaded diffusion model instead?
6. What reproducibility metadata would another student need to recreate your results?

---

## Part D: Extended Exercises

Complete at least two exercises for **Stable Diffusion Pipeline Exploration**:

1. **Guidance sweep** - Run the same prompt and seed with guidance scale 3, 7.5, and 12. Compare alignment and artifacts.
2. **Step sweep** - Run 10, 30, and 60 inference steps. Compare speed and detail.
3. **Seed diversity** - Hold prompt and settings fixed while changing seeds. Identify stable versus unstable prompt elements.
4. **Scheduler comparison** - If available, swap schedulers and record how the denoising trajectory affects results.
5. **Architecture sketch** - Draw DALL-E 2 and Stable Diffusion side by side, labeling text encoder, prior, U-Net, autoencoder, and upsamplers.
6. **Failure log** - Document one failed or surprising output and assign the likely source: prompt ambiguity, text encoder, denoiser, VAE decoder, sampler, or safety filter.

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

---

## Troubleshooting

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| OOM on GPU | Model or image size too large | Use CPU, lower resolution, enable attention slicing, or use a smaller model |
| Very slow inference | Running on CPU or too many steps | Reduce steps for experiments; save high-step runs for final samples |
| Prompt ignored | Guidance too low or prompt ambiguous | Increase guidance moderately; rewrite prompt with clearer objects and relations |
| Oversaturated artifacts | Guidance too high | Lower guidance scale and compare seeds |
| Weak fine detail | VAE bottleneck, low steps, or upsampling limits | Increase steps, test another seed, and inspect whether the prompt demands small text or tiny objects |

---

## Common Lab Mistakes

| Mistake | What Goes Wrong | Fix |
|---------|-----------------|-----|
| Skipping device check | Unexpectedly slow inference | Print CUDA availability before loading the pipeline |
| No random seed | Prompt study cannot be reproduced | Use a fixed `torch.Generator` seed and log it |
| Comparing only beautiful samples | Alignment failures disappear from the report | Score every planned run, not only favorites |
| Changing many settings at once | No causal interpretation | Sweep one variable at a time |
| Forgetting model metadata | Results cannot be recreated | Log model id, scheduler, steps, seed, guidance, size, and prompt |

---

## Submission Checklist

- [ ] Stable Diffusion pipeline runs, or fallback pseudocode/component map is complete
- [ ] At least six generated samples or documented dry-run cases are saved
- [ ] Prompt log includes model id, scheduler, seed, steps, guidance, and image size
- [ ] Alignment and fidelity are scored separately
- [ ] DALL-E 2 versus Stable Diffusion architecture sketch is included
- [ ] Short write-up (300-500 words) interprets results using Foster Chapter 13 vocabulary

---

## Further Reading

- Foster, Ch. 13 - *Multimodal Models*
- [Chapter 13 README](./README.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Source-Grounded Lab Notes

This lab operationalizes three Foster Chapter 13 ideas:

| Source idea | Lab action |
|-------------|------------|
| Stable Diffusion performs denoising in latent space | Identify the VAE and U-Net roles before generating images |
| Prompt-conditioned diffusion is stochastic | Vary seed while holding prompt fixed |
| Evaluation must separate alignment and fidelity | Score prompt following and visual realism independently |

### Practice Checklist

- [ ] Run one baseline prompt with a fixed seed
- [ ] Change only the seed and compare diversity
- [ ] Change only guidance scale and compare prompt adherence
- [ ] Change only inference steps and compare speed/detail trade-off
- [ ] Save a component diagram linking VAE, U-Net, text encoder, and scheduler
- [ ] Write one failure analysis that localizes the likely component

### Architecture Reflection

Stable Diffusion differs from DALL-E 2 at the representation boundary. DALL-E 2 predicts a CLIP image embedding and then decodes through a GLIDE-style diffusion process. Stable Diffusion denoises an autoencoder latent directly, then decodes the latent to pixels. Your lab report should make that contrast visible.

---
**Previous:** [Section 13.8 - Comparison & Synthesis](./section-08-comparison-and-synthesis.md)
**Next:** [Chapter 13 README](./README.md)

---

## Assessment Practice

Use the shared [Assessment Appendix](../../ASSESSMENT_APPENDIX.md) for concept audits, worked examples, implementation checks, experiment logs, oral-exam prompts, and deliverable checklists.
