# Capstone Project: Generative AI System

> **Course:** 4 — Generative Deep Learning  
> **When:** After completing Chapters 01–14  
> **Estimated time:** 25–50 hours

---

## Project Overview

Demonstrate mastery of Course 4 by building a generative AI system with a clear model family, evaluation plan, and responsible deployment story. This can be a research notebook, an interactive prototype, or a production-style application.

---

## Requirements

### 1. Project Track

Choose one track:

- **Image generation:** VAE, GAN, diffusion, or Stable Diffusion pipeline
- **Text generation:** transformer language model, fine-tuning, RAG, or evaluation harness
- **Music generation:** Transformer or MuseGAN-style symbolic music model
- **World model:** generative model inside an RL or simulation loop
- **Multimodal app:** CLIP, image-text retrieval, text-to-image, or vision-language workflow

### 2. Technical Requirements

| Component | Requirement |
|-----------|-------------|
| Model family | Explicit architecture choice tied to Course 4 chapters |
| Dataset | Documented source, preprocessing, splits, and limitations |
| Baseline | Prompt-only, simpler model, or reference pipeline |
| Training/adaptation | Train, fine-tune, adapter-tune, or configure a reproducible pipeline |
| Evaluation | Quality, diversity, grounding/alignment, latency, and failure cases |
| Interface | Notebook, CLI, Gradio/Streamlit, or API prototype |
| Responsible AI | Misuse, bias, copyright, privacy, environmental cost, and mitigations |
| Maintenance | Model card, versioned config, and rollback/update plan |

### 3. Deliverables

```text
capstone/
├── README.md              # Product/research goal, setup, usage
├── report.md              # Architecture, experiments, evaluation, ethics
├── notebooks/             # Exploration and diagnostics
├── src/                   # Pipeline or app code
├── configs/               # Model, prompt, adapter, or scheduler configs
├── eval/                  # Test prompts, scoring scripts, human review notes
├── samples/               # Generated artifacts with seeds/configs
└── model_card.md
```

---

## Evaluation Rubric

| Criterion | Weight | Excellent |
|-----------|--------|-----------|
| Architecture fit | 20% | Model family matches goal and is justified against alternatives |
| Implementation | 20% | Reproducible system with clear configs and setup |
| Evaluation | 20% | Measures quality, diversity, safety, and real task usefulness |
| Modern practice | 15% | Uses RAG/PEFT/diffusers/eval tooling where appropriate |
| Responsible deployment | 15% | Risks and mitigations are concrete and testable |
| Documentation | 10% | Clear model card, limitations, and maintenance notes |

---

## Milestone: Generative AI Architect

Completing this capstone plus Course 4 earns the **Generative AI Architect** milestone. In this curriculum, that means strong generative foundations plus practical modern-system awareness; production frontier LLM specialization still requires ongoing paper/tooling updates.
