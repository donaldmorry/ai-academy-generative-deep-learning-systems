# Section 1.6: Core Probability Theory

> **Source inheritance:** Foster, Ch. 1 - probabilistic foundations for generative modeling  
> **Enhanced with:** Exact derivations, KL divergence preview, connections to Course 3  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)    
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Why Probability Is the Language of Generation

Every [generative model](../../GLOSSARY.md#generative-model) is a probability distribution in disguise. VAEs optimize an evidence lower bound. GANs implicitly sample from $P_\theta(x)$. Diffusion models define a sequence of conditional distributions. Autoregressive models factorize joint probabilities.

If you speak probability fluently, the rest of this course is architecture - engineering choices around the same mathematical skeleton.

You built foundations in [Course 3, Chapter 03](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-03-probability-information-theory/README.md) and [Course 2, Chapter 13](https://github.com/Collaborative-ai/ai-academy-artificial-intelligence-modern-approach/blob/main/chapters/chapter-13-probabilistic-reasoning/README.md). This section collects the exact formulas generative deep learning uses daily.

---

## Joint, Marginal, and Conditional Distributions

### Joint distribution

The joint probability $P(x, z)$ describes **everything** - both observed data $x$ and [latent variable](../../GLOSSARY.md#latent-variable) $z$ together:

$$
P(x, z) = P(x \mid z) \cdot P(z) \quad \text{(product rule)}
$$
> **Readable form:** Joint probability of observation and latent equals likelihood times prior on latent.

For a dataset with labels: $P(x, y) = P(x \mid y) \cdot P(y)$.

### Marginal distribution

**Marginalize** (sum or integrate) over variables you do not observe:

$$
P(x) = \sum_{z} P(x, z) = \sum_{z} P(x \mid z) P(z)
$$
> **Readable form:** Marginal P(x) sums joint probability over all discrete latent values z.

Continuous case:

$$
P(x) = \int P(x \mid z) \, P(z) \, dz
$$
> **Readable form:** Marginal P(x) integrates joint over continuous latent space.

This integral is the heart of latent-variable models - and notoriously intractable for complex $P(x \mid z)$.

> **Readable form:** The marginal $P(x)$ answers "how likely is this image, period?" You get it by averaging over every possible hidden explanation $z$.

### Conditional distribution

Condition on what you know:

$$
P(z \mid x) = \frac{P(x \mid z) \, P(z)}{P(x)}
$$
> **Readable form:** Posterior over latent z given x is proportional to likelihood times prior.

The posterior $P(z \mid x)$ - "given this image, what latent code generated it?" - is what the VAE encoder approximates.

---

## Bayes' Rule

$$
P(z \mid x) = \frac{P(x \mid z) \, P(z)}{P(x)}
$$
> **Readable form:** Posterior over latent z given x is proportional to likelihood times prior.

| Term | Name | Role
|------|------|------|
| $P(z)$ | **Prior** | Belief about $z$ before seeing data |
| $P(x \mid z)$ | **Likelihood** | How $z$ generates $x$ |
| $P(z \mid x)$ | **Posterior** | Updated belief about $z$ after observing $x$ |
| $P(x)$ | **Evidence** | Normalizer: $P(x) = \int P(x \mid z) P(z) \, dz$ |

**Classification via Bayes** (from [Section 1.2](./section-02-generative-vs-discriminative.md)):

$$
P(y \mid x) = \frac{P(x \mid y) \, P(y)}{P(x)}
$$
> **Readable form:** Bayes flips the direction of conditioning. Generative models naturally compute $P(x \mid y)$; Bayes turns that into $P(y \mid x)$ for classification.

---

## Maximum Likelihood Estimation (MLE)

Given data $\mathcal{D} = \{x^{(1)}, \ldots, x^{(N)}\}$, choose parameters $\theta$ that maximize the likelihood:

$$
\theta^* = \arg\max_\theta \, P(\mathcal{D} \mid \theta) = \arg\max_\theta \prod_{i=1}^{N} P(x^{(i)} \mid \theta)
$$
> **Readable form:** Best parameters maximize probability of observed data (or log-sum thereof).

Log-likelihood (equivalent, numerically stable):

$$
\theta^* = \arg\max_\theta \sum_{i=1}^{N} \log P(x^{(i)} \mid \theta)
$$
> **Readable form:** Best parameters maximize probability of observed data (or log-sum thereof).

For i.i.d. data, this decomposes into a sum - one term per example.

**Coin flip** ([Section 1.4](./section-04-our-first-generative-model.md)):

$$
\hat{\theta}_{\text{MLE}} = \frac{1}{N} \sum_{i=1}^{N} x_i
$$
> **Readable form:** MLE coin bias equals average of binary observations.

**Gaussian** with fixed variance $\sigma^2$:

$$
\hat{\mu}_{\text{MLE}} = \frac{1}{N} \sum_{i=1}^{N} x_i
$$
> **Readable form:** [Maximum likelihood estimation](../../GLOSSARY.md#maximum-likelihood-estimation) asks: "What parameters make the data I actually saw as probable as possible?" Training a neural generative model is gradient ascent on this same objective (or a tractable bound).

### MLE in TensorFlow

```python
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

data = tf.constant([2.1, 2.8, 3.0, 2.5, 3.2, 2.9])

mu = tf.Variable(0.0)
sigma = tf.Variable(1.0, constraint=lambda v: tf.nn.softplus(v) + 1e-3)

def neg_log_likelihood():
    dist = tfd.Normal(loc=mu, scale=sigma)
    return -tf.reduce_sum(dist.log_prob(data))

optimizer = tf.keras.optimizers.Adam(0.1)
for _ in range(500):
    with tf.GradientTape() as tape:
        loss = neg_log_likelihood()
    grads = tape.gradient(loss, [mu, sigma])
    optimizer.apply_gradients(zip(grads, [mu, sigma]))

print(f"MLE: μ={mu.numpy():.3f}, σ={sigma.numpy():.3f}")
```

---

## Maximum A Posteriori (MAP)

MLE can overfit with little data. **MAP** adds a prior:

$$
\theta^*_{\text{MAP}} = \arg\max_\theta \, P(\theta \mid \mathcal{D}) = \arg\max_\theta \, P(\mathcal{D} \mid \theta) \, P(\theta)
$$
> **Readable form:** MAP estimate maximizes posterior: likelihood times prior over parameters.

Log form:

$$
\theta^*_{\text{MAP}} = \arg\max_\theta \left[ \log P(\mathcal{D} \mid \theta) + \log P(\theta) \right]
$$
> **Readable form:** MAP estimate maximizes posterior: likelihood times prior over parameters.

The prior $P(\theta)$ acts as **regularization**. L2 weight decay in neural networks corresponds to a Gaussian prior on weights - a bridge to [Course 3, Chapter 07](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-07-regularization/README.md).

---

## KL Divergence (Preview)

The **Kullback-Leibler divergence** measures how one distribution differs from another:

$$
D_{\mathrm{KL}}\bigl( Q \,\|\, P \bigr) = \mathbb{E}_{x \sim Q}\left[ \log \frac{Q(x)}{P(x)} \right] = \sum_x Q(x) \log \frac{Q(x)}{P(x)}
$$
> **Readable form:** KL divergence measures expected log-ratio between approximate distribution Q and target P.

Continuous:

$$
D_{\mathrm{KL}}\bigl( Q \,\|\, P \bigr) = \int Q(x) \log \frac{Q(x)}{P(x)} \, dx
$$
> **Readable form:** KL divergence measures expected log-ratio between approximate distribution Q and target P.

**Properties:**

- $D_{\mathrm{KL}}(Q \| P) \geq 0$, equality iff $Q = P$ almost everywhere
- **Asymmetric:** $D_{\mathrm{KL}}(Q \| P) \neq D_{\mathrm{KL}}(P \| Q)$ in general

> **Readable form:** KL divergence is a "distance" between distributions - but only in one direction. It asks: "If I use model $P$ to encode data from $Q$, how many extra bits do I waste?"

### Why generative models care

The VAE objective is the **ELBO** (Evidence Lower BOund):

$$
\log P(x) \geq \mathbb{E}_{q(z|x)}[\log P(x|z)] - D_{\mathrm{KL}}\bigl( q(z|x) \,\|\, P(z) \bigr)
$$
> **Readable form:** KL divergence measures expected log-ratio between approximate distribution Q and target P.

Maximizing the ELBO = minimizing reconstruction error + minimizing KL to the prior. You will derive this in [Chapter 03](../chapter-03-variational-autoencoders/README.md).

```python
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

P = tfd.Normal(loc=0., scale=1.)   # target
Q = tfd.Normal(loc=1., scale=1.5)  # approximation

kl = tfd.kl_divergence(Q, P)  # KL(Q || P)
print(f"KL(Q || P) = {kl.numpy():.4f}")
```

---

## Factorization and Autoregressive Models

For high-dimensional $x = (x_1, \ldots, x_T)$, the chain rule factorizes the joint:

$$
P(x) = P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_1, x_2) \cdots P(x_T \mid x_1, \ldots, x_{T-1})
$$
> **Readable form:** Joint distribution equals chain of conditionals: first element, then each next given all prior.

Compactly:

$$
P(x) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$
> **Readable form:** Joint probability factorizes as product of each token (or pixel) given all previous tokens.

This is exactly how **GPT** and **PixelCNN** work - and why language models are generative. Each conditional is a neural network output.

---

## Summary Table

| Concept | Formula | Generative use
|---------|---------|----------------|
| Joint | $P(x, z) = P(x | z)P(z)$ |
| Define full generative story | Marginal | $P(x) = \int P(x |
| z)P(z)\,dz$ | The quantity we want to maximize | Posterior |
| $P(z | x) \propto P(x | z)P(z)$ |
| Encoder in VAE | MLE | $\arg\max_\theta \sum_i \log P(x^{(i)} |
| \theta)$ | Default training objective | KL |
| $D_{\mathrm{KL}}(Q \ | P) = \mathbb{E}_Q[\log Q - \log P]$ | VAE regularizer, distillation |

---

## Key Vocabulary

| Term | Definition
|------|-----------|
| **Prior $P(z)$** | Distribution over latent variables before observing data |
| **Likelihood $P(x \mid z)$** | How latents generate observations |
| **Posterior $P(z \mid x)$** | Latent distribution given observed data |
| **Evidence / marginal likelihood** | $P(x)$ - often intractable |
| **ELBO** | Tractable lower bound on $\log P(x)$ |

---

## Reflection Questions

1. Why is $P(x) = \int P(x|z)P(z)\,dz$ generally intractable for neural decoders?
2. Derive the MAP estimate for a coin flip with a $\text{Beta}(\alpha, \beta)$ prior. How does it differ from MLE?
3. Why is KL divergence asymmetric, and which direction does the VAE use?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 1. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. Ch. 3 - Probability; Ch. 5.5 - MLE; Ch. 6 - KL. [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Ch. 1-2. [https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
- Kullback, S., & Leibler, R. A. (1951). On Information and Sufficiency. *Annals of Mathematical Statistics*. [https://doi.org/10.1214/aoms/1177729694](https://doi.org/10.1214/aoms/1177729694)
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)

---

**Previous:** [Section 1.5 - Representation Learning](./section-05-representation-learning.md)  
**Next:** [Section 1.7 - Generative Model Taxonomy](./section-07-generative-model-taxonomy.md)



