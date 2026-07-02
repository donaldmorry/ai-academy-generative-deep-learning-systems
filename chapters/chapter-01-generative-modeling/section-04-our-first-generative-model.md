# Section 1.4: Our First Generative Model

> **Source inheritance:** Foster, Ch. 1 - coin-flip generative model walkthrough  
> **Enhanced with:** Step-by-step likelihood, sampling, TensorFlow implementation  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)    
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Why Start with a Coin Flip?

Stable Diffusion has a billion parameters. GPT-4 has trillions of tokens in its training set. Neither is a good *first* generative model.

Foster begins with a biased coin - the simplest possible [generative model](../../GLOSSARY.md#generative-model). One parameter $\theta$, binary outcomes, exact math. Every concept you need - **likelihood**, **sampling**, **maximum likelihood estimation** - fits on one page.

> **Readable form:** If you can generate coin flips from a learned model, you understand the loop that also generates symphonies and cat photos. The coin is just easier to debug.

---

## The Model

Flip a coin $n$ times. Each flip $x_i \in \{0, 1\}$ (tails = 0, heads = 1) is independent and identically distributed (i.i.d.) from a Bernoulli distribution:

$$
P(x_i = 1) = \theta, \quad P(x_i = 0) = 1 - \theta
$$
> **Readable form:** Probability of heads equals theta; probability of tails equals 1 minus theta.

Compactly:

$$
P(x_i \mid \theta) = \theta^{x_i} (1 - \theta)^{1 - x_i}
$$
> **Readable form:** Bernoulli probability equals theta to the power x_i times (1-theta) to the power (1-x_i).

For a sequence $\mathbf{x} = (x_1, \ldots, x_n)$:

$$
P(\mathbf{x} \mid \theta) = \prod_{i=1}^{n} \theta^{x_i} (1 - \theta)^{1 - x_i} = \theta^{\sum x_i} (1 - \theta)^{n - \sum x_i}
$$
> **Readable form:** Likelihood of full flip sequence equals theta to the power of total heads times (1-theta) to the power of tails.

This is the **likelihood** - how probable the observed data is, given parameter $\theta$.

---

## Maximum Likelihood Estimation

We want the $\theta$ that makes our observed flips most probable. Take the log (sums are friendlier than products):

$$
\log P(\mathbf{x} \mid \theta) = \left(\sum_{i=1}^{n} x_i\right) \log \theta + \left(n - \sum_{i=1}^{n} x_i\right) \log(1 - \theta)
$$
> **Readable form:** Log-likelihood equals (heads count) times log theta plus (tails count) times log(1-theta).

Differentiate, set to zero, solve:

$$
\hat{\theta}_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$
> **Readable form:** [Maximum likelihood estimation](../../GLOSSARY.md#maximum-likelihood-estimation) for a coin is just counting heads and dividing by total flips. The fancy notation will generalize; the intuition stays the same.

**Example:** Observed flips `[1, 0, 1, 1, 0, 1, 1, 1]` → 6 heads out of 8 → $\hat{\theta} = 0.75$.

---

## Sampling: Generation in Action

Once we have $\hat{\theta}$, **generation** means drawing new flips:

$$
x_{\text{new}} \sim \text{Bernoulli}(\hat{\theta})
$$
> **Readable form:** New flip is drawn from Bernoulli distribution with learned parameter theta-hat.

Sample 100 new sequences of length 8. Plot the distribution of head counts. Compare to the theoretical binomial distribution. If your model learned well, they should match.

This is the generative modeling contract:

```
Observe data → Estimate θ via MLE → Sample x_new ~ P(x | θ)
```

Every model in this course follows the same contract - with neural networks replacing the coin.

---

## Full Python Implementation

### NumPy version (Foster's pedagogical starting point)

```python
import numpy as np
import matplotlib.pyplot as plt

# Observed data: 1 = heads, 0 = tails
observed_flips = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1])

# Step 1: MLE - fraction of heads
theta_mle = observed_flips.mean()
print(f"MLE estimate θ = {theta_mle:.3f}")

# Step 2: Log-likelihood of observed data
n_heads = observed_flips.sum()
n = len(observed_flips)
log_likelihood = (
    n_heads * np.log(theta_mle) + (n - n_heads) * np.log(1 - theta_mle)
)
print(f"Log-likelihood: {log_likelihood:.3f}")

# Step 3: Sample new sequences
n_samples = 10_000
seq_length = 16
sampled_flips = np.random.binomial(1, theta_mle, size=(n_samples, seq_length))

# Step 4: Distribution of head counts in generated data
head_counts = sampled_flips.sum(axis=1)
plt.hist(head_counts, bins=range(seq_length + 2), density=True, alpha=0.7,
         label="Sampled")
# Theoretical binomial PMF
from math import comb
k = np.arange(seq_length + 1)
binom_pmf = [comb(seq_length, ki) * theta_mle**ki * (1-theta_mle)**(seq_length-ki)
             for ki in k]
plt.plot(k, binom_pmf, "r-o", label="Theoretical Binomial")
plt.xlabel("Number of heads in 16 flips")
plt.ylabel("Probability")
plt.legend()
plt.title(f"Generative Coin Model (θ={theta_mle:.2f})")
plt.savefig("coin_flip_distribution.png", dpi=120)
plt.show()
```

### TensorFlow / TensorFlow Probability version

```python
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

observed = tf.constant([1., 0., 1., 1., 0., 1., 1., 1., 0., 1.])

# Generative model: single Bernoulli with learnable θ
theta = tf.Variable(0.5, dtype=tf.float32)
coin = tfd.Bernoulli(probs=theta)

# Negative log-likelihood loss
def nll():
    return -tf.reduce_sum(coin.log_prob(observed))

# Optimize θ via gradient descent (same answer as counting)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
for step in range(200):
    with tf.GradientTape() as tape:
        loss = nll()
    grads = tape.gradient(loss, [theta])
    optimizer.apply_gradients(zip(grads, [theta]))

print(f"Learned θ: {theta.numpy():.3f}")  # → 0.7

# Sample 20 new flips
new_flips = coin.sample(20)
print("Generated flips:", new_flips.numpy().astype(int))
```

> **Readable form:** The TensorFlow version is how deep generative models train. Replace `Bernoulli` with a neural network outputting pixel distributions, and `observed` with a batch of images. The loop - define $P_\theta(x)$, minimize $-\log P_\theta(x)$, sample - is identical.

---

## The Generative Modeling Framework

Foster's framework, which every subsequent chapter follows:

| Step | Coin flip | Stable Diffusion
|------|-----------|------------------|
| **1. Choose model family** | Bernoulli | Latent diffusion U-Net |
| **2. Parameterize** | Single $\theta$ | Millions of weights $\theta$ |
| **3. Define likelihood** | $\prod_i P(x_i \mid \theta)$ | Variational bound on $\log P(x \mid \theta)$ |
| **4. Optimize** | MLE (count heads) | SGD on GPU clusters |
| **5. Sample** | `binomial` / `Bernoulli.sample` | Iterative denoising loop |

**Bold milestone:** You have now completed the full generative pipeline. Everything else is engineering scale-up.

---

## Likelihood as a Score

Likelihood is not just for training - it is for **scoring**. Which of these sequences is more consistent with $\theta = 0.75$?

- A: `[1, 1, 1, 1, 1, 1, 1, 1]` → $\log P \approx 8 \log(0.75) \approx -2.31$
- B: `[0, 0, 0, 0, 0, 0, 0, 0]` → $\log P \approx 8 \log(0.25) \approx -11.09$

Sequence A is far more likely under a 75% heads coin. In high dimensions, low likelihood flags anomalies - a bank transaction that "doesn't look like" normal spending.

---

## Connection to Probability Theory

This section uses i.i.d. assumptions and MLE directly. [Section 1.6](./section-06-core-probability-theory.md) formalizes joint, marginal, and conditional distributions. [Section 1.5](./section-05-representation-learning.md) shows what happens when $\theta$ is not a single number but a [latent variable](../../GLOSSARY.md#latent-variable) $z$ inside a neural network.

---

## Key Vocabulary

| Term | Definition
|------|-----------|
| **Bernoulli distribution** | Single binary outcome with parameter $\theta$ |
| **Likelihood** | $P(\text{data} \mid \theta)$ as a function of $\theta$ |
| **Log-likelihood** | $\log P(\text{data} \mid \theta)$ - additive across i.i.d. samples |
| **i.i.d.** | Independent and identically distributed |
| **Sampling** | Drawing $x \sim P_\theta(x)$ |

---

## Reflection Questions

1. What happens to $\hat{\theta}_{\text{MLE}}$ if you observe only one flip?
2. How would you model a *fair* coin without looking at data? (Hint: prior.)
3. Why do we maximize log-likelihood instead of likelihood?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 1 - coin-flip example. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Foster's codebase: [https://github.com/davidADSP/GDL_code](https://github.com/davidADSP/GDL_code)
- TensorFlow Probability - Distributions: [https://www.tensorflow.org/probability/api_docs/python/tfp/distributions](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. Ch. 5.5 - maximum likelihood. [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

---

**Previous:** [Section 1.3 - The Rise of Generative AI](./section-03-the-rise-of-generative-ai.md)  
**Next:** [Section 1.5 - Representation Learning](./section-05-representation-learning.md)
