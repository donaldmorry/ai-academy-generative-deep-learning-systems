# Section 1.1: What Is Generative Modeling?

> **Source inheritance:** Foster, Ch. 1 - "What Is Generative Modeling?"  
> **Enhanced with:** Modern generative AI context, ecosystem connections, creative-AI analogies    
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)    
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Most machine learning you have encountered so far answers a **discriminative** question: given an input, what label does it belong to? Spam or ham? Cat or dog? Will this customer churn?

[Generative modeling](../../GLOSSARY.md#generative-model) asks a different question entirely:

> **Given examples of data, can we learn to create new data that looks like it came from the same source?**

Instead of drawing a boundary between classes, a generative model learns the **distribution** of the data itself - the patterns, textures, rhythms, and structures that make a cat photo look like a cat photo, or a Bach fugue sound like Bach.

Think of it like this:

| Approach | Analogy | Question |

|----------|---------|----------|
| **Discriminative** | Art critic | "Is this a Van Gogh or a forgery?" |
| **Generative** | Forger who studied Van Gogh for years | "Paint me a new painting in Van Gogh's style." |

The forger does not just learn to spot fakes - they internalize brushstroke patterns, color palettes, and compositional habits, then **produce** new works. That is generative modeling in a nutshell.

---

## What Generative Models Actually Do

Formally, a generative model learns a probability distribution over data:

$$
P_\theta(x)
$$
> **Readable form:** Generative model assigns probability P-theta to data point x, parameterized by learned weights theta.

where $x$ might be an image, a sentence, a melody, or a sequence of coin flips. Once trained, the model can:

1. **Sample** - draw new examples $x' \sim P_\theta(x)$
2. **Evaluate likelihood** - score how probable a given example is under the model
3. **Complete or transform** - condition on partial input to fill in the rest

> **Readable form:** A generative model is a statistical imagination engine. It studies thousands of examples until it can dream up plausible new ones.

### Three Superpowers

**Density estimation.** The model assigns probabilities to data points. High-probability regions are "typical" data; low-probability regions are outliers or nonsense. This is how anomaly detection and data compression connect to generation.

**Sampling.** Draw random examples from the learned distribution. Stable Diffusion samples images; ChatGPT samples text one token at a time. Every "AI art" post you have seen is sampling in action.

**Representation.** Many generative models discover internal structures - [latent variables](../../GLOSSARY.md#latent-variable) - that capture meaning in compressed form. More on this in [Section 1.5](./section-05-representation-learning.md).

---

## Real-World Applications

Generative AI is not a laboratory curiosity. It powers production systems today:

| Domain | Example | What the model generates
|--------|---------|--------------------------|
| **Images** | DALL·E 3, Midjourney, Stable Diffusion | Photorealistic or stylized images from text prompts |
| **Text** | ChatGPT, Claude, Gemini | Essays, code, dialogue, summaries |
| **Music** | Suno, MuseNet, MusicLM | Melodies, full tracks, instrument stems |
| **Video** | Sora, Runway Gen-3 | Short clips from text or image prompts |
| **3D & design** | Point-E, Shap-E | Meshes, textures, product mockups |
| **Science** | AlphaFold (structure), molecular VAEs | Protein folds, drug candidates |
| **Simulation** | World models in robotics | Future frames for planning without real-world risk |

Each of these is a different flavor of the same recipe: learn $P(x)$ (or $P(x \mid \text{condition})$) from data, then sample.

---

## A Minimal TensorFlow Preview

Before we build intuition with Foster's coin-flip model in [Section 1.4](./section-04-our-first-generative-model.md), here is the shape of generative code in TensorFlow/Keras - learning a simple Gaussian and sampling from it:

```python
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

# Observed 1-D data (e.g., heights in cm)
data = tf.constant([162., 168., 171., 165., 174., 169., 170.])

# MLE for Gaussian: mean and std from data
mu = tf.reduce_mean(data)
sigma = tf.math.reduce_std(data)

# Generative model: P(x) = Normal(mu, sigma)
model = tfd.Normal(loc=mu, scale=sigma)

# Sample 5 new "heights"
new_samples = model.sample(5)
print("Generated:", new_samples.numpy())
# Evaluate likelihood of a specific value
log_prob = model.log_prob(170.0)
print("Log P(170):", log_prob.numpy())
```

This is trivial - but the **pattern** scales to billion-parameter transformers. Learn parameters $\theta$, define $P_\theta(x)$, sample.

> **Readable form:** Even this five-line script is a generative model. It learned "what heights look like" and can invent new ones. Stable Diffusion does the same thing in 512×512 pixel space instead of one number.

---

## Generative vs Discriminative: First Glimpse

You will study this distinction deeply in [Section 1.2](./section-02-generative-vs-discriminative.md). For now:

- **Discriminative:** models $P(y \mid x)$ - the boundary between classes
- **Generative:** models $P(x)$ or $P(x, y)$ - the data-generating process

A generative model *can* do classification (via Bayes' rule), but a discriminative model cannot paint you a picture. That asymmetry is why generative AI feels like magic - and why this course exists.

If you trained a cat/dog classifier in [Course 1, Chapter 03](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-03-classification-models/README.md), you built a discriminative model. This course teaches you to build the forger.

---

## Why Generative Modeling Is Hard

High-dimensional data - images with millions of pixels, sentences with thousands of possible tokens - lives on intricate **manifolds** in vast spaces. Most random pixel arrays are visual noise; only a tiny fraction look like photographs.

**Challenge 1: Dimensionality.** $P(x)$ for a 256×256 RGB image is a distribution over $\mathbb{R}^{196{,}608}$. You cannot store a lookup table.

**Challenge 2: Multimodality.** The distribution of "faces" is not one blob - it is countless modes (different people, angles, lighting). The model must capture all of them.

**Challenge 3: Evaluation.** There is no single accuracy score for "how good is this generated image?" We will revisit metrics throughout the course.

These challenges motivate every architecture in Part II - VAEs, GANs, flows, autoregressive models, diffusion. [Section 1.7](./section-07-generative-model-taxonomy.md) maps the landscape.

---

## Connection to Prior Courses

| Concept | Where you learned it | How it applies here
|---------|---------------------|---------------------|
| Probability basics | [Course 3, Chapter 03](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-03-probability-information-theory/README.md) | $P(x)$, Bayes, MLE are the language of generative models |
| Neural networks | [Course 1, Chapters 08-09](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-08-deep-learning/README.md) | Deep nets parameterize $P_\theta(x)$ |
| Unsupervised learning | [Course 1, Section 1.3](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-01-machine-learning/section-03-supervised-vs-unsupervised-learning.md) | Most generative training uses unlabeled data |
| Bayesian reasoning | [Course 2, Chapter 13](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-13-probabilistic-reasoning/README.md) | Priors, posteriors, and latent variables |

---

## Historical Milestones

| Year | **Milestone**
|------|---------------|
| 1980s | Boltzmann machines, autoassociative memories |
| 2013 | **Variational Autoencoders** (Kingma & Welling) - tractable latent-variable models |
| 2014 | **GANs** (Goodfellow et al.) - implicit generation via adversarial training |
| 2016 | PixelRNN/PixelCNN - tractable likelihood for images |
| 2020 | **DDPM** (Ho et al.) - diffusion models reignite image synthesis |
| 2022 | Stable Diffusion, ChatGPT - generative AI goes mainstream |
| 2024-2026 | Multimodal models, video generation, agentic systems |

You are entering the field at its most exciting moment. The fundamentals in this chapter - probability, likelihood, sampling - are the constants beneath the hype.

---

## Key Vocabulary

| Term | Definition
|------|-----------|
| **Generative model** | Learns $P(x)$ to create or score data |
| **Sampling** | Drawing new examples from a learned distribution |
| **Likelihood** | $P(x \mid \theta)$ - how probable data is under parameters $\theta$ |
| **Latent variable** | Hidden factor $z$ that generates observed $x$ |
| **Density estimation** | Learning the probability density function of data |

---

## Reflection Questions

1. Name three tasks where *generating* new data is more useful than *classifying* existing data.
2. Why is modeling $P(x)$ harder than modeling $P(y \mid x)$ for a 1024×1024 image?
3. How is a language model like ChatGPT an example of generative modeling?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). O'Reilly. Ch. 1. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
- Goodfellow, I. et al. (2014). Generative Adversarial Nets. [https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. Ch. 20 - Deep Generative Models. [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

---

**Next:** [Section 1.2 - Generative vs Discriminative](./section-02-generative-vs-discriminative.md)




---

## Assessment Practice

Use the shared [Assessment Appendix](../../ASSESSMENT_APPENDIX.md) for concept audits, worked examples, implementation checks, experiment logs, oral-exam prompts, and deliverable checklists.
