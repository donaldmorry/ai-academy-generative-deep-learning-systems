# Section 1.2: Generative vs Discriminative

> **Source inheritance:** Foster, Ch. 1 - generative vs discriminative framework  
> **Enhanced with:** Decision boundaries, Bayes connection, practical trade-offs    
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)    
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Two Philosophies of Machine Learning

Imagine two chefs judging a cooking competition:

- **Chef A (discriminative)** tastes each dish and says "Italian" or "not Italian." They never learned to cook - they only learned the boundary between cuisines.
- **Chef B (generative)** studied Italian cooking for years, internalized recipes and techniques, and can both recognize *and cook* authentic Italian food.

Chef A is a [discriminative model](../../GLOSSARY.md#discriminative-model). Chef B is a [generative model](../../GLOSSARY.md#generative-model).

The mathematical distinction is clean:

| Type | Models | Question answered |

|------|--------|-------------------|
| **Discriminative** | $P(y \mid x)$ or $f(x) \to y$ | "Given this input, what is the label?" |
| **Generative** | $P(x)$ or $P(x \mid y)$ | "What does data look like? Can I create more?" |

You met discriminative models throughout [Course 1](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/README.md): logistic regression, SVMs, neural classifiers. This course is about becoming Chef B.

---

## The Discriminative View

A discriminative classifier learns the **decision boundary** between classes. It does not care what a "cat" looks like in full generality - only how to separate cats from dogs in feature space.

For binary classification with logistic regression:

$$
P(y=1 \mid x) = \sigma(w^\top x + b) = \frac{1}{1 + e^{-(w^\top x + b)}}
$$
> **Readable form:** Probability of positive class equals sigmoid of linear combination of input features.

The model outputs a probability over labels **conditioned on** the input. Training maximizes $\prod_i P(y_i \mid x_i)$ - or equivalently minimizes cross-entropy loss.

```python
import tensorflow as tf
from tensorflow import keras

# Discriminative: P(y | x) - labels given features
model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(784,)),
    keras.layers.Dense(10, activation="softmax"),  # P(class | pixel values)
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # -log P(y | x)
    metrics=["accuracy"],
)
```

> **Readable form:** This network is an art critic. It looks at pixels and says "digit 7." It has no idea how to draw a 7 - only how to recognize one.

---

## The Generative View

A generative model learns the **data distribution** itself:

$$
P_\theta(x) \quad \text{or} \quad P_\theta(x \mid z)
$$
> **Readable form:** Generative model assigns probability P-theta to data point x, parameterized by learned weights theta.

For labeled data, it may model the joint distribution:

$$
P(x, y) = P(x \mid y) \cdot P(y)
$$
> **Readable form:** Joint probability of features and label equals probability of features given label, times prior probability of label.

- $P(x \mid y)$: what examples of class $y$ look like
- $P(y)$: how common each class is (the prior)

**Sampling** a new labeled example:

1. Draw $y \sim P(y)$
2. Draw $x \sim P(x \mid y)$

That is literally how conditional image generators work: "generate a cat" means sample from $P(x \mid y=\text{cat})$.

> **Readable form:** The generative model learned what cats *are* - whiskers, ears, fur texture - not just where the cat/dog line is drawn.

---

## Bayes' Rule: The Bridge

Generative models can classify via [Bayes' rule](./section-06-core-probability-theory.md):

$$
P(y \mid x) = \frac{P(x \mid y) \cdot P(y)}{P(x)}
$$
> **Readable form:** Class posterior equals class-conditional density times class prior, divided by evidence P(x).

where the evidence $P(x) = \sum_y P(x \mid y) P(y)$.

This is **Naive Bayes** in [Course 1, Chapter 03](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-03-classification-models/README.md) - a generative classifier. You model each class's word distribution, then invert with Bayes.

| Approach | Pros | Cons
|----------|------|------|
| **Discriminative** | Often higher accuracy with less data; simpler training | Cannot generate; no $P(x)$ |
| **Generative** | Can sample, impute, detect anomalies; handles missing data | Harder to train; more parameters; sometimes lower classification accuracy |

Ng & Jordan (2002) famously showed generative models can outperform discriminative ones in **low-data regimes** because they make stronger assumptions that act as regularization.

---

## Visual Intuition: Decision Boundaries vs Distributions

Picture two clusters of 2-D points - red and blue.

- **Discriminative model:** draws a curve separating red from blue. Points far from both clusters? Irrelevant, as long as they land on the correct side.
- **Generative model:** fits a Gaussian (or neural density) to red points and another to blue points. The boundary emerges from where $P(x \mid \text{red}) P(\text{red}) = P(x \mid \text{blue}) P(\text{blue})$.

```python
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

# Two Gaussian generative class components
red = tfd.MultivariateNormalDiag(
    loc=[2.0, 2.0], scale_diag=[0.8, 0.8]
)
blue = tfd.MultivariateNormalDiag(
    loc=[5.0, 5.0], scale_diag=[0.8, 0.8]
)
prior_red, prior_blue = 0.5, 0.5

def classify_point(x):
  x = tf.constant(x, dtype=tf.float32)
  log_post_red = red.log_prob(x) + np.log(prior_red)
  log_post_blue = blue.log_prob(x) + np.log(prior_blue)
  return "red" if log_post_red > log_post_blue else "blue"

print(classify_point([3.5, 3.5]))  # boundary region
```

> **Readable form:** The generative classifier asks "which blob of probability does this point belong to?" instead of "which side of the fence am I on?"

---

## When to Use Which

### Prefer discriminative when:

- You only need classification or regression
- Labeled data is plentiful
- Interpretability of decision boundaries matters
- Latency and simplicity are critical (production classifiers)

### Prefer generative when:

- You need to **create** new data (art, text, music, molecules)
- You need **anomaly detection** (low $P(x)$ = unusual)
- Data is **missing or incomplete** (impute via conditional sampling)
- You need **uncertainty over inputs**, not just labels
- Semi-supervised or self-supervised learning (learn $P(x)$, fine-tune on labels)

**Example:** A bank fraud detector might use a generative model of "normal transactions" and flag anything with low likelihood - without needing labeled fraud examples for every attack variant.

---

## Deep Learning: Same Split, Bigger Scale

| Architecture | Type | Typical use
|--------------|------|-------------|
| ResNet, EfficientNet | Discriminative | Image classification |
| BERT (fine-tuned classifier head) | Discriminative | Sentiment, NER |
| GPT, Llama | Generative | Text completion, chat |
| Stable Diffusion | Generative | Image synthesis |
| VAE, GAN, Flow | Generative | Images, audio, latent editing |

Modern LLMs are generative at their core: they model $P(x)$ over token sequences (or equivalently $P(x_t \mid x_{<t})$). When you add a classification head to BERT, you are bolting a discriminative task onto a generative pretraining objective.

See [Section 1.3](./section-03-the-rise-of-generative-ai.md) for how generative models moved from niche to mainstream.

---

## The Asymmetry That Matters

Here is the key insight Foster emphasizes:

> **Generative models are strictly more powerful in principle** - they model the full data distribution. Discriminative models model only the slice needed for decisions.

But "more powerful" does not mean "always better." Modeling all of $P(x)$ for ImageNet-scale images is astronomically harder than modeling $P(y \mid x)$. That tension - expressiveness vs tractability - drives every chapter in this course.

---

## Key Vocabulary

| Term | Definition
|------|-----------|
| **$P(y \mid x)$** | Conditional probability of label given input |
| **$P(x)$** | Marginal probability (density) of data |
| **Joint distribution** | $P(x, y)$ - captures both features and labels |
| **Decision boundary** | Surface where $P(y=1 \mid x) = P(y=0 \mid x)$ |
| **Evidence** | $P(x) = \sum_y P(x \mid y) P(y)$ - normalizer in Bayes |

---

## Reflection Questions

1. Is a spam filter trained with Naive Bayes generative or discriminative? Why?
2. Can a GAN perform image classification without modification? What would you need?
3. Why might a generative model detect novel fraud better than a discriminative one?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 1. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Ng, A. Y., & Jordan, M. I. (2002). On Discriminative vs. Generative Classifiers. [https://papers.nips.cc/paper/2020-on-discriminative-vs-generative-classifiers-a-comparison-of-logistic-regression-and-naive-bayes](https://papers.nips.cc/paper/2020-on-discriminative-vs-generative-classifiers-a-comparison-of-logistic-regression-and-naive-bayes)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. Ch. 5 - ML Basics; Ch. 8 - ML for Deep Learning. [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Ch. 1.4 - generative vs discriminative. [https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)

---

**Previous:** [Section 1.1 - What Is Generative Modeling?](./section-01-what-is-generative-modeling.md)  
**Next:** [Section 1.3 - The Rise of Generative AI](./section-03-the-rise-of-generative-ai.md)
