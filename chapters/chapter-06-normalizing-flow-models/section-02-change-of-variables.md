# Section 6.2: Change of Variables

> **Source inheritance:** Foster, Ch. 6 — "Change of Variables"  
> **Enhanced with:** Jacobian determinant, log-det-Jacobian, composition of flows, NLL training  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

Normalizing flows compute **exact** data likelihoods by pushing densities through invertible maps. If $z = f(x)$ and $p_Z$ is a known base distribution, the change-of-variables formula tells you $p_X(x)$ — no variational bound, no discriminator. Training minimizes **negative log-likelihood** over the dataset. Every architectural choice in RealNVP exists to make $\det \partial z / \partial x$ cheap.

---

## The Formula

For bijective $f: \mathbb{R}^d \to \mathbb{R}^d$:

$$
p_X(x) = p_Z(f(x)) \left| \det \frac{\partial f(x)}{\partial x} \right|
$$
> **Readable form:** This derivative measures local sensitivity: a larger magnitude means the output changes more for a small input change.

Log-domain (stable for optimization):

$$
\log p_X(x) = \log p_Z(z) + \log \left| \det \frac{\partial z}{\partial x} \right|
$$
> **Readable form:** Log density at x equals log base density at z plus log Jacobian determinant.

With $p_Z = \mathcal{N}(0, I)$:

$$
\log p_Z(z) = -\frac{1}{2}\|z\|^2 - \frac{d}{2}\log(2\pi)
$$
> **Readable form:** Standard-normal log density is highest near zero and falls with squared latent distance and dimensionality.

---

## Negative Log-Likelihood Loss

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log p_X(x_i)
$$
> **Readable form:** This loss is the scalar training signal: it is larger when predictions violate the target and smaller when they match.

Foster's RealNVP `train_step` minimizes this — `log_loss` returns mean negative sum of $\log p_Z(z) + \log\det J$.

```python
def log_loss(self, x):
    z, logdet = self(x)  # forward f(x)
    log_likelihood = self.distribution.log_prob(z) + logdet
    return -tf.reduce_mean(log_likelihood)
```

---

## Composition of Flows

Stack $f_1, \ldots, f_K$ with $z = f_K \circ \cdots \circ f_1(x)$:

$$
\log \left| \det \frac{\partial z}{\partial x} \right| = \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial x_{k-1}} \right|
$$
> **Readable form:** Total log-det is sum of per-layer log-dets.

Inverse generation: $x = f_1^{-1} \circ \cdots \circ f_K^{-1}(z)$ with $z \sim \mathcal{N}(0,I)$.

---

## RealNVP Coupling Jacobian

Affine coupling on split $x = [x_a, x_b]$:

$$
z_b = x_b \odot \exp(s(x_a)) + t(x_a), \quad z_a = x_a
$$
> **Readable form:** An affine coupling layer keeps part of the input fixed and uses it to scale and shift the remaining dimensions.

Jacobian is lower triangular — determinant is $\prod_j \exp(s_j)$, so:

$$
\log |\det J| = \sum_j s_j(x_a)
$$
> **Readable form:** The total combines the indexed terms, so each relevant example, state, feature, or dimension contributes once.

No $d \times d$ matrix determinant needed ([Section 6.3](./section-03-realnvp-architecture.md)).

---

## Comparison to Other Objectives

| Model | Objective |
|-------|-----------|
| VAE | $\mathbb{E}[\log p(x|z)] - D_{\mathrm{KL}}$ (bound) |
| GAN | $\min_G \max_D$ game |
| **Flow** | $-\log p_X(x)$ **exact** |
| PixelCNN | $-\sum \log p(x_i|x_{<i})$ |

Flows share tractable likelihood with autoregressive models but parallelize forward passes.

---

## Numerical Stability

- Work in **log space** — multiply determinants as sums of logs
- Clip extreme $s(x)$ in coupling layers if needed
- `layers.Normalization` on two moons data before training (Foster Example 6-1)

---

## Connection to Probability Course

Change of variables generalizes univariate $u = g(x)$:

$$
p_U(u) = p_X(g^{-1}(u)) \left| \frac{dx}{du} \right|
$$
> **Readable form:** The expression assigns probability to the event or value using the stated model assumptions.

Flows are the multivariate neural-parameterized version ([Course 3, Chapter 03](https://github.com/Collaborative-ai/ai-academy-deep-learning-foundations/blob/main/chapters/chapter-03-probability-information-theory/README.md)).

---

## Inverse Coupling Equations

Given forward $z_b = x_b \odot \exp(s(x_a)) + t(x_a)$ with $z_a = x_a$:

$$
x_a = z_a, \quad x_b = (z_b - t(z_a)) \odot \exp(-s(z_a))
$$
> **Readable form:** Affine coupling is analytically invertible — no iterative root-finding.

---

## Common Pitfalls

| Pitfall | Symptom | Remedy |
|---------|---------|--------|
| Forgetting log-det | Wrong MLE | Add `logdet` to loss |
| Non-invertible layer | Undefined $p(x)$ | Coupling / 1×1 conv only |
| Dimension change | Invalid Jacobian | Preserve $d$ throughout |
| Underflow in det | NaN | Log-space arithmetic |

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **Change of variables** | Density transform via Jacobian determinant |
| **Log-det-Jacobian** | $\log |\det \partial z/\partial x|$ |
| **NLL** | Negative log-likelihood training loss |
| **Composition** | Stacked flows with additive log-dets |
| **Base log-prob** | $\log p_Z(z)$ for Gaussian prior |

---

## Reflection Questions

1. Write the log-domain change-of-variables equation.
2. Why do RealNVP Jacobians factor as $\exp(\sum s)$?
3. How does flow MLE differ from VAE ELBO?
4. Why must flows preserve dimensionality?

---


## Foster Notebook Reference

Re-run the chapter notebook in [GDL_code](https://github.com/davidADSP/GDL_code) and compare your tensor shapes, loss curves, and saved sample grids to Foster's figures. Document one hyperparameter you changed and how outputs shifted — this habit transfers directly to Part III architectures (Transformers, Stable Diffusion, MuseGAN).

| Checkpoint | Action |
|------------|--------|
| After `model.summary()` | Verify spatial dims match hand calculation |
| Mid-training | Save sample grid or diagnostic plot |
| After training | Compare to Foster figure captions in the PDF |

---

## Extension Reading

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.) — full chapter walkthrough
- Goodfellow, Bengio & Courville (2016). *Deep Learning* — generative models part
- Original papers cited in Foster's chapter references



## Lab Integration Notes

When completing the chapter lab, tie this section's implementation checklist to your deliverable:

1. **Reproduce** Foster's primary figure for this topic (save PNG).
2. **Ablate** one hyperparameter (learning rate, latent dim, noise schedule, critic steps, etc.).
3. **Compare** to one other Part II model family on the same metric (likelihood, sample sharpness, or training stability).
4. **Write** three sentences explaining failure modes you observed and the fix you applied.

These steps mirror how generative modeling is evaluated in research and production — not only final image quality, but reproducibility and diagnosed trade-offs.

## Tensor Shape Debugging Template

```python
import tensorflow as tf

def trace_shapes(model, sample_input):
    x = sample_input
    for layer in model.layers:
        try:
            x = layer(x)
            print(f"{layer.name:30s} {x.shape}")
        except Exception as e:
            print(f"{layer.name:30s} ERROR: {e}")
            break
```

Use after every architecture change before committing to long training runs.


## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 6. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Dinh, L. et al. (2017). Density Estimation using Real NVP.
- Foster's codebase: `notebooks/06_normflow/01_realnvp/realnvp.ipynb`

---

**Previous:** [Section 6.1 — Normalizing Flows Introduction](./section-01-normalizing-flows-introduction.md)  
**Next:** [Section 6.3 — RealNVP Architecture](./section-03-realnvp-architecture.md)


