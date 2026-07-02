# Section 1.7: Generative Model Taxonomy

> **Source inheritance:** Foster, Ch. 1 - taxonomy of generative model families  
> **Enhanced with:** Explicit vs implicit models, tractability trade-offs, course roadmap map  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)    
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## A Map of the Territory

By now you know that a [generative model](../../GLOSSARY.md#generative-model) learns $P_\theta(x)$ and samples new data. But *how* it represents and trains that distribution varies enormously - from a single Bernoulli parameter to a billion-weight diffusion U-Net.

This section is your **road map** for the rest of the course. Every family in Part II occupies a distinct point in design space. Understanding the taxonomy now prevents the common beginner mistake of reaching for a GAN when a flow or autoregressive model is the better tool.

> **Readable form:** Generative models are like vehicles. Bicycles, sedans, and rockets all move you - but you would not commute in a Saturn V. The taxonomy tells you which engine fits which journey.

---

## Two Fundamental Axes

### Axis 1: Explicit vs Implicit Density

| Type | Defines $P_\theta(x)$ directly? | Can compute likelihood?
|------|-----------------------------------|------------------------|
| **Explicit** | Yes - formula or tractable approximation | Yes (or bounded, e.g., ELBO) |
| **Implicit** | No - only a sampling procedure | No exact likelihood |

- **Explicit:** VAEs, normalizing flows, autoregressive models (PixelCNN, GPT)
- **Implicit:** GANs (sample via generator; no $P(x)$ formula)

> **Readable form:** Explicit models publish the recipe *and* the nutrition label. Implicit models hand you a dish and say "trust me, it tastes like the training set."

### Axis 2: Tractable Likelihood

Even among explicit models, computing $P_\theta(x)$ may be:

| Tractability | Examples | Training
|-------------|----------|--------|
| **Exact** | Flows, autoregressive | Direct [MLE](../../GLOSSARY.md#maximum-likelihood-estimation) |
| **Approximate** | VAEs (ELBO) | Variational inference |
| **Intractable** | GANs, some EBMs | Adversarial or MCMC |

The intractability of $P(x) = \int P(x|z)P(z)\,dz$ for complex decoders is why VAEs exist - see [Section 1.6](./section-06-core-probability-theory.md).

---

## The Model Family Tree

```
                        Generative Models
                              |
            +-----------------+-----------------+
            |                                   |
      Explicit Density                    Implicit Density
            |                                   |
    +-------+-------+-------+            +------+------+
    |       |       |       |            |             |
   VAE    Flow   Autoreg.  Diffusion     GAN          EBM
```

Each branch maps to a chapter in this course.

---

## Family 1: Variational Autoencoders (VAEs)

**Idea:** Learn encoder $q_\phi(z|x)$ and decoder $p_\theta(x|z)$. Optimize the ELBO:

$$
\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}\bigl(q_\phi(z|x) \,\|\, p(z)\bigr)
$$
> **Readable form:** KL divergence measures expected log-ratio between approximate distribution Q and target P.

| Property | Value
|----------|-------|
| Likelihood | Approximate (lower bound) |
| Sampling | $z \sim P(z)$, then $x \sim p_\theta(x |
| z)$ | Latent space |
| Continuous, navigable | Strengths |
| Stable training, meaningful [latent variables](../../GLOSSARY.md#latent-variable) | Weaknesses |

**Chapter:** [03 - Variational Autoencoders](../chapter-03-variational-autoencoders/README.md)  
**Key paper:** Kingma & Welling (2013) - [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)

---

## Family 2: Generative Adversarial Networks (GANs)

**Idea:** Generator $G$ vs discriminator $D$ - minimax game:

$$
\min_G \max_D \; \mathbb{E}_{x \sim P_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim P(z)}[\log(1 - D(G(z)))]
$$
> **Readable form:** GAN trains generator to fool discriminator while discriminator maximizes ability to distinguish real from fake.

| Property | Value
|----------|-------|
| Likelihood | Not defined |
| Sampling | $z \to G(z)$ - one forward pass |
| Latent space | Continuous, semantically rich |
| Strengths | Sharp, realistic samples |
| Weaknesses | Training instability, mode collapse, no likelihood |

**Chapters:** [04 - GANs](../chapter-04-generative-adversarial-networks/README.md), [10 - Advanced GANs](../chapter-10-advanced-gans/README.md)  
**Key paper:** Goodfellow et al. (2014) - [https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)

---

## Family 3: Autoregressive Models

**Idea:** Factorize the joint via the chain rule:

$$
P(x) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$
> **Readable form:** Joint probability factorizes as product of each token (or pixel) given all previous tokens.

Each conditional is a neural network (RNN, CNN, or Transformer).

| Property | Value
|----------|-------|
| Likelihood | Exact, tractable |
| Sampling | Sequential - one token/pixel at a time |
| Strengths | Principled, strong for text and audio |
| Weaknesses | Slow generation (cannot parallelize sampling) |

**Examples:** PixelCNN, WaveNet, **GPT**, Llama  
**Chapters:** [05 - Autoregressive](../chapter-05-autoregressive-models/README.md), [09 - Transformers](../chapter-09-transformers/README.md)

---

## Family 4: Normalizing Flows

**Idea:** Build complex $P(x)$ by composing invertible transformations:

$$
P(x) = P(z) \left| \det \frac{\partial f^{-1}}{\partial x} \right|
$$
> **Readable form:** Flow model density equals base density times absolute Jacobian determinant of inverse transform.

| Property | Value
|----------|-------|
| Likelihood | Exact |
| Sampling | $z \sim P(z)$, apply inverse flow |
| Strengths | Exact density, efficient sampling |
| Weaknesses | Architectural constraints (invertibility) |

**Chapter:** [06 - Normalizing Flows](../chapter-06-normalizing-flow-models/README.md)  
**Key paper:** Dinh et al. (2016) RealNVP - [https://arxiv.org/abs/1605.08803](https://arxiv.org/abs/1605.08803)

---

## Family 5: Diffusion Models

**Idea:** Forward process adds noise; learn to reverse it:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\bigl(\sqrt{1-\beta_t}\, x_{t-1},\; \beta_t \mathbf{I}\bigr)
$$
> **Readable form:** Forward diffusion adds Gaussian noise with variance beta-t to previous timestep.

Training learns $p_\theta(x_{t-1} \mid x_t)$; sampling iteratively denoises.

| Property | Value
|----------|-------|
| Likelihood | Variational bound (not commonly used) |
| Sampling | Many steps (50-1000), but high quality |
| Strengths | State-of-the-art image quality, stable training |
| Weaknesses | Slow sampling; compute-heavy |

**Chapter:** [08 - Diffusion Models](../chapter-08-diffusion-models/README.md)  
**Key paper:** Ho et al. (2020) DDPM - [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)

---

## Family 6: Energy-Based Models (EBMs)

**Idea:** Define an energy function $E_\theta(x)$; probability via Boltzmann distribution:

$$
P_\theta(x) = \frac{e^{-E_\theta(x)}}{Z(\theta)}, \quad Z(\theta) = \int e^{-E_\theta(x)} dx
$$
> **Readable form:** Generative model assigns probability P-theta to data point x, parameterized by learned weights theta.

The partition function $Z(\theta)$ is intractable.

| Property | Value
|----------|-------|
| Likelihood | Intractable (partition function) |
| Sampling | MCMC (Langevin dynamics) |
| Strengths | Flexible, no generator architecture constraints |
| Weaknesses | Slow sampling, training difficulty |

**Chapter:** [07 - Energy-Based Models](../chapter-07-energy-based-models/README.md)

---

## Comparison at a Glance

| Family | Likelihood | Sample quality | Speed | Latent space | Course chapter
|--------|-----------|----------------|-------|-------------|---------------|
| VAE | Approximate | Medium (blurry) | Fast | Yes | 03 |
| GAN | None | High (sharp) | Fast | Yes | 04, 10 |
| Autoregressive | Exact | High | Slow | Optional | 05, 09 |
| Flow | Exact | Medium-High | Fast | Yes | 06 |
| Diffusion | Bound | Very high | Slow | Yes (latent) | 08 |
| EBM | Intractable | Variable | Very slow | No | 07 |

---

## Hybrid and Modern Architectures

Real systems mix families:

| System | Hybrid of
|--------|-----------|
| **Stable Diffusion** | VAE (latent space) + Diffusion + CLIP conditioning |
| **DALL·E 2** | Autoregressive prior + diffusion decoder |
| **VQ-GAN** | Vector quantization + GAN + autoregressive transformer |
| **GPT-4V** | Autoregressive LLM + vision encoder |

**Bold milestone:** No single family won. The 2024-2026 landscape combines the best of each - and you will understand all of them by course end.

---

## Choosing a Model Family

| Your goal | Start here
|-----------|-----------|
| Interpretable latent space, smooth interpolation | VAE |
| Photorealistic images, fast sampling | GAN (or distilled diffusion) |
| Text, code, music sequences | Autoregressive / Transformer |
| Exact likelihood, density estimation | Flow |
| Highest image quality, text conditioning | Diffusion |
| Research flexibility, energy landscapes | EBM |

---

## Connection to [Section 1.2](./section-02-generative-vs-discriminative.md)

Every generative family can support [discriminative models](../../GLOSSARY.md#discriminative-model) via Bayes or learned representations - but not vice versa. The taxonomy is about **how** you parameterize $P_\theta(x)$, not whether you can also classify.

---

## Key Vocabulary

| Term | Definition
|------|-----------|
| **Explicit density** | Model provides a formula for $P_\theta(x)$ |
| **Implicit model** | Model provides sampling only (no tractable $P_\theta(x)$) |
| **ELBO** | Evidence lower bound - VAE training objective |
| **Partition function $Z$** | Normalizer making $e^{-E(x)}$ a valid distribution |
| **Mode collapse** | GAN failure - generator produces limited variety |

---

## Reflection Questions

1. Why can GANs produce sharp images but VAEs often look blurry?
2. Which family would you use for a medical anomaly detector that needs $P(x)$? Why?
3. Draw your own taxonomy diagram with the six families and one example system each.

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 1. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
- Goodfellow, I. et al. (2014). Generative Adversarial Nets. [https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density Estimation using Real NVP. [https://arxiv.org/abs/1605.08803](https://arxiv.org/abs/1605.08803)
- Bond-Taylor, S. et al. (2021). Deep Generative Modelling: A Comparative Review of VAEs, GANs, Normalizing Flows, Energy-Based and Autoregressive Models. [https://arxiv.org/abs/2103.04999](https://arxiv.org/abs/2103.04999)

---

**Previous:** [Section 1.6 - Core Probability Theory](./section-06-core-probability-theory.md)  
**Next:** [Section 1.8 - Codebase Setup](./section-08-codebase-setup.md)
