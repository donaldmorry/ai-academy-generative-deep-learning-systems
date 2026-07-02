# Section 8.1: Diffusion Models Introduction

> **Source inheritance:** Foster, Ch. 8 — "Introduction" / "Denoising Diffusion Models"  
> **Enhanced with:** DiffuseTV story, DDPM context, score-based lineage, and generative taxonomy placement  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

**Diffusion models** learn to generate data by reversing a gradual noising process. Start with a real image $x_0$; add tiny amounts of Gaussian noise over $T$ steps until $x_T$ is pure noise. Train a neural network to **denoise** — predict what was added or what $x_{t-1}$ should be. At generation time: sample $x_T \sim \mathcal{N}(0, I)$, iteratively denoise back to $x_0$.

Foster's **DiffuseTV** story: a shop of TVs showing progressively clearer pictures as you walk past — each set undoes one step of static. The viewer picks random static as the seed; unlimited "channels" of generated images.

> **Readable form:** Destroy images with noise in training; learn to un-destroy; at test time, un-destroy pure noise into new art.

---

## DiffuseTV Analogy Formalized

| Story element | Diffusion role |
|---------------|----------------|
| Training TV shows | Forward process $q(x_t \mid x_{t-1})$ |
| Walking deeper into shop | Increasing timestep $t$ |
| Final clear TV | Original data $x_0$ |
| Random static choice | Sample $x_T \sim \mathcal{N}(0,I)$ |
| Denoising circuitry | Learned $p_\theta(x_{t-1} \mid x_t)$ |

Unlike [VAEs](../chapter-03-variational-autoencoders/section-04-variational-autoencoders.md), the forward noising process is **fixed** — not learned. Unlike [GANs](../chapter-04-generative-adversarial-networks/section-01-gan-introduction.md), no adversary — just regression on noise. Unlike [EBMs](../chapter-07-energy-based-models/section-01-energy-based-models-introduction.md), sampling is a fixed Markov chain, not Langevin on an energy landscape.

---

## Historical Lineage

| Year | Milestone |
|------|-----------|
| 2015 | Sohl-Dickstein et al. — diffusion probabilistic models |
| 2019 | Song & Ermon — score matching / NCSN |
| 2020 | Ho et al. — **DDPM**, rivals GANs on images |
| 2021 | Nichol & Dhariwal — improved schedules, guidance |
| 2022 | Stable Diffusion — latent diffusion at scale |
| 2024+ | Video diffusion, consistency models |

DDPM connected diffusion to **denoising score matching** — unifying Foster's Chapter 07 and 08 narratives.

---

## Denoising Diffusion Probabilistic Models (DDPM)

Core training loop (preview — full math in [Sections 8.2–8.6](./section-02-forward-diffusion-process.md)):

1. Sample image $x_0$ from dataset
2. Sample timestep $t \in \{1, \ldots, T\}$
3. Sample noise $\epsilon \sim \mathcal{N}(0, I)$
4. Form noisy image $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$
5. Train network $\epsilon_\theta(x_t, t)$ to predict $\epsilon$ with MSE

Generation: start from $x_T$, repeatedly apply learned reverse step for $t = T, \ldots, 1$.

---

## Why Diffusion Works

**Stable training** — MSE on noise is simpler than GAN minimax.  
**Mode coverage** — likelihood-based objective; less collapse than GANs.  
**Quality** — with enough steps and U-Net capacity, SOTA on ImageNet, faces, text-to-image.  
**Trade-off** — **slow sampling** (hundreds–thousands of network evaluations per image). DDIM, latent diffusion, and distillation address this in production.

---

## Chapter 08 Roadmap

| Section | Topic |
|--------|-------|
| 8.1 | Introduction (this section) |
| 8.2 | Forward diffusion process |
| 8.3 | Reparameterization & schedules |
| 8.4 | Reverse diffusion process |
| 8.5 | U-Net denoising architecture |
| 8.6 | Training loop + EMA |
| 8.7 | Sampling from DDPM |
| 8.8 | Analysis & connections |

Dataset: Oxford 102 Flowers, 64×64 RGB. Notebook: `notebooks/08_diffusion/01_ddm/ddm.ipynb`.

---

## Placement in Generative Taxonomy

From [Chapter 01](../chapter-01-generative-modeling/section-07-generative-model-taxonomy.md):

| Family | Likelihood | Sampling | Image SOTA |
|--------|------------|----------|------------|
| Diffusion | VLB (approx) | Many steps | **Leading** |
| GAN | Implicit | 1 step | Strong |
| Flow | Exact | 1 step | Good |
| VAE | ELBO | 1 step | Moderate |
| AR | Exact | Slow | Niche |

Diffusion occupies **high quality + slow sampling** — the default for Stable Diffusion, DALL·E 3 pipeline components, Midjourney-era systems.

---

## Minimal Noise Preview (TensorFlow)

```python
import tensorflow as tf

x0 = tf.random.uniform((1, 64, 64, 3))  # image in [0,1]
epsilon = tf.random.normal(tf.shape(x0))
alpha_bar_t = 0.5  # example signal fraction at timestep t
xt = tf.sqrt(alpha_bar_t) * x0 + tf.sqrt(1.0 - alpha_bar_t) * epsilon
# xt is a 50% noisy blend — network will learn to recover epsilon from xt
```

Full schedule in [Section 8.3](./section-03-reparameterization-and-diffusion-schedules.md).

---

## What You Will Build End-to-End

By the end of Chapter 08 you will have:

1. Loaded and preprocessed the Oxford 102 Flowers dataset at 64×64 resolution
2. Implemented offset cosine noise schedules and one-shot corruption $x_t = \text{signal} \cdot x_0 + \text{noise} \cdot \epsilon$
3. Built a U-Net that accepts `(noisy_image, noise_variance)` and returns $\hat{\epsilon}$
4. Trained with L1 noise loss and EMA weight tracking
5. Sampled novel flower images from pure Gaussian noise through 1000 reverse steps

This pipeline is the unconditional core inside larger systems. Stable Diffusion adds a VAE latent space ([Chapter 03](../chapter-03-variational-autoencoders/section-04-variational-autoencoders.md)), CLIP text embeddings (Part III), and classifier-free guidance — but the denoising U-Net you train here is the same object, just operating on pixels instead of latents.

---

## Diffusion vs Iterative Refinement Analogy

Think of a sculptor starting from a block of marble (noise):

| Step | Sculptor | DDPM |
|------|----------|------|
| Start | Rough stone | $x_T \sim \mathcal{N}(0,I)$ |
| Middle | Coarse shape emerges | Mid-$t$ denoising removes large-scale static |
| End | Fine detail chiseled | Low-$t$ steps restore petal texture |
| Tool | Hand + chisel | Shared U-Net weights at every scale |

Each reverse step makes a **small** change — mirroring the forward process that added noise in tiny increments. Large jumps without training support (skipping timesteps naively) produce artifacts; DDIM exists precisely to skip safely.

---

## Engineering Considerations

| Concern | Practical note |
|---------|----------------|
| GPU memory | 64×64×3 U-Net, batch 64 — reduce batch if OOM |
| Training time | Hours on single GPU for visible flowers |
| Sampling time | Minutes per grid at T=1000 — use DDIM for demos |
| Reproducibility | Fix seeds for `epsilon` and `x_T` when comparing checkpoints |
| Checkpointing | Save EMA weights, not only training weights |

Foster's notebook is the ground truth for tensor shapes — verify your `DownBlock` output sizes match before long runs.

---

## Prerequisites Checklist

Before starting Chapter 08 labs, you should be comfortable with:

- Conv2D / UpSampling2D from [Chapter 02](../chapter-02-deep-learning/section-06-convolutional-layers.md)
- VAE encoder-decoder analogy from [Chapter 03](../chapter-03-variational-autoencoders/section-01-autoencoder-architecture.md)
- Gaussian reparameterization from [Section 3.5](../chapter-03-variational-autoencoders/section-05-vae-loss-and-reparameterization.md)
- Custom `train_step` from [GAN](../chapter-04-generative-adversarial-networks/section-03-training-the-dcgan.md) or [EBM](../chapter-07-energy-based-models/section-05-training-the-ebm.md) training

Missing any of these — skim the linked section first; diffusion combines all of them.

---

## Reading Order Within Chapter 08

Follow sections 8.1→8.8 sequentially — each section assumes the previous. Do not skip to U-Net code (8.5) before forward/reverse math (8.2–8.4) or schedules will seem arbitrary. Lab 08 should reuse Foster's `ddm.ipynb` with the same hyperparameters before ablating.

---

## Stable Diffusion Stack (Named Components)

| Component | Foster chapter section |
|-----------|---------------------|
| VAE encoder/decoder | [3.4 VAE](../chapter-03-variational-autoencoders/section-04-variational-autoencoders.md) |
| Denoising U-Net | [8.5 U-Net](./section-05-u-net-denoising-model.md) |
| Noise schedule | [8.3 Schedules](./section-03-reparameterization-and-diffusion-schedules.md) |
| Text conditioning | Part III (not Part II) |

Recognizing this stack helps when reading Rombach et al. after finishing Foster Ch. 8.

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Diffusion model** | Generative model via forward noising + learned denoising |
| **DDPM** | Denoising Diffusion Probabilistic Model (Ho et al.) |
| **Forward process** | Fixed Gaussian noising $q(x_t \mid x_{t-1})$ |
| **Reverse process** | Learned denoising $p_\theta(x_{t-1} \mid x_t)$ |
| **Noise prediction** | Network target $\epsilon_\theta(x_t, t) \approx \epsilon$ |

---

## Reflection Questions

1. How does DiffuseTV's "random static" map to the generative sampling procedure?
2. Why is the forward process fixed rather than learned like a VAE encoder?
3. What trade-off makes diffusion slower than GAN sampling?
4. How does DDPM relate to energy/score models from Chapter 07?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 8 — Introduction. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- Sohl-Dickstein, J. et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics.

---

**Previous:** [Chapter 08 Overview](./README.md)  
**Next:** [Section 8.2 — Forward Diffusion Process](./section-02-forward-diffusion-process.md)
