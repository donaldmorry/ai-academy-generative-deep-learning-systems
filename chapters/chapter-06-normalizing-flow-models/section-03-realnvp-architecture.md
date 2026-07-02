# Section 6.3: RealNVP Architecture

> **Source inheritance:** Foster, Ch. 6 — "RealNVP"  
> **Enhanced with:** Coupling layers, affine transforms, triangular Jacobians, and stacked masking  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

[Normalizing flows](../../GLOSSARY.md#normalizing-flow) need invertible transforms with **tractable Jacobian determinants**. Real-valued Non-Volume Preserving (**RealNVP**, Dinh et al., 2017) achieves both via **coupling layers**: split the input vector, pass one half through a neural net to produce scale $s$ and translation $t$, then affine-transform the other half. The Jacobian is lower-triangular — its determinant is just the product of diagonal exponentials.

Stack coupling layers with **alternating masks** so every dimension gets updated. The result: a bijective map from complex data (two moons, later images in GLOW) to Gaussian noise — exact likelihood and one-shot sampling.

> **Readable form:** Half the coordinates pass through unchanged; the other half gets scaled and shifted based on the first half. Flip the mask next layer.

---

## Coupling Layer: Forward Transform

For input $x \in \mathbb{R}^D$, choose split dimension $d < D$. Mask so the coupling network sees only $x_{1:d}$:

$$
z_{1:d} = x_{1:d}
$$

$$
z_{d+1:D} = x_{d+1:D} \odot \exp\bigl(s(x_{1:d})\bigr) + t(x_{1:d})
$$
> **Readable form:** First d components copy through; remaining components get element-wise scale and shift from a net reading the first half.

Here $s$ and $t$ are outputs of separate neural networks (Dense stacks for 2D; Conv2D for images). The $\exp(s)$ ensures positive scale factors. Foster uses **tanh** on $s$ and **linear** on $t$ for the two-moons example.

```python
from tensorflow.keras import layers, models, regularizers

def Coupling(input_dim=2):
  inp = layers.Input(shape=(input_dim,))

  def scaling_stream(x):
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(0.01))(x)
    return layers.Dense(input_dim, activation="tanh",
                        kernel_regularizer=regularizers.l2(0.01))(x)

  def translation_stream(x):
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(0.01))(x)
    return layers.Dense(input_dim, activation="linear",
                        kernel_regularizer=regularizers.l2(0.01))(x)

  s_out = scaling_stream(inp)
  t_out = translation_stream(inp)
  return models.Model(inp, [s_out, t_out])
```

---

## Masking and the Lower-Triangular Jacobian

Only $x_{1:d}$ feeds the coupling nets; $s$ and $t$ are masked so only positions $d+1:D$ receive updates. The Jacobian $\partial z / \partial x$ has structure:

$$
J = \begin{pmatrix} I_d & 0 \\ * & \text{diag}(\exp(s)) \end{pmatrix}
$$
> **Readable form:** Top-left is identity; bottom-right is diagonal of exponentials; determinant is product of those exponentials.

$$
\log \left| \det J \right| = \sum_{j} s_j(x_{1:d})
$$
> **Readable form:** The total combines the indexed terms, so each relevant example, state, feature, or dimension contributes once.

No $O(n^3)$ determinant — just sum the scale outputs. This was the computational breakthrough that made high-dimensional flows practical.

---

## Inverse Transform (Generation)

Invert by algebra (given $z$, recover $x$):

$$
x_{1:d} = z_{1:d}
$$

$$
x_{d+1:D} = \bigl(z_{d+1:D} - t(x_{1:d})\bigr) \odot \exp\bigl(-s(x_{1:d})\bigr)
$$
> **Readable form:** Undo affine transform on the second half using the same s and t from the first half.

**Sampling:** draw $z \sim \mathcal{N}(0, I)$, apply inverse coupling layers in reverse order → $x$ in data space. One forward pass through the inverse stack — fast generation, unlike PixelCNN.

---

## Stacking with Alternating Masks

One coupling layer leaves $x_{1:d}$ **unchanged**. Fix: stack layers, **flip** which dimensions are masked each time. Layer 1 updates dims $2$; layer 2 updates dim $1$; repeat.

Properties preserved under composition:

$$
\det(A \cdot B) = \det(A)\det(B)
$$
> **Readable form:** The determinant of composed linear transforms factors into the product of their determinants.

$$
(f_B \circ f_A)^{-1} = f_A^{-1} \circ f_B^{-1}
$$
> **Readable form:** To invert a composition, undo the later transform first and the earlier transform second.

```python
import tensorflow as tf

class CouplingLayer(tf.keras.layers.Layer):
  def __init__(self, coupling_net, mask):
    super().__init__()
    self.coupling_net = coupling_net
    self.mask = tf.constant(mask, dtype=tf.float32)

  def call(self, x, inverse=False):
    masked_x = x * self.mask
  # s, t from coupling_net(masked_x); apply affine to (1-mask) portion
  # return z, log_det_jacobian
```

Foster's full RealNVP chains multiple `CouplingLayer` instances with masks `[1,0]` and `[0,1]` alternating for $D=2$.

---

## RealNVP vs VAE Encoder-Decoder

| | VAE | RealNVP |
|---|-----|---------|
| Encoder | Approximate $q(z|x)$ | Exact $f(x)$ |
| Decoder | $g(z)$ | Exact inverse $f^{-1}(z)$ |
| Latent | Stochastic | Deterministic bijection |
| Likelihood | ELBO | Exact via change of variables |

From [Section 6.1](./section-01-normalizing-flows-introduction.md): Jacob's F.L.O.W. machine — forward digitizes paintings to Gaussian codes; ringing the bell samples noise and decodes new art. RealNVP is that machine with coupling layers as the gears.

---

## Extension to Images (Preview)

For 2D moons, Dense coupling nets suffice. **GLOW** ([Section 6.7](./section-07-glow.md)) replaces Dense with Conv2D coupling on checkerboard/channel masks for $32 \times 32$ RGB. The coupling math is identical — only the masking pattern and network architecture change.

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Same mask every layer | Half of dims never transform | Alternate masks |
| Unbounded $s$ | Numerical overflow in $\exp(s)$ | tanh on scaling stream |
| Forgetting log-det in loss | Wrong density | Add $\sum s_j$ to log-likelihood |
| No L2 regularization | Wild scales on moons | Foster uses `l2(0.01)` per Dense |

---

## Connection to Prior Sections

| Concept | Link |
|---------|------|
| Change of variables | [Section 6.2](./section-02-change-of-variables.md) |
| Base Gaussian $p(z)$ | [Section 6.1](./section-01-normalizing-flows-introduction.md) |
| Two moons data | [Section 6.4](./section-04-two-moons-dataset.md) |
| Training NLL | [Section 6.5](./section-05-training-realnvp.md) |

---

## Worked Micro-Example (D=2)

Let $x = (x_1, x_2)$, mask $[1,0]$ on first coupling layer:

- Input to coupling net: $(x_1, 0)$
- Outputs: $s = (0, s_2)$, $t = (0, t_2)$ after masking
- Update: $z_1 = x_1$, $z_2 = x_2 \cdot e^{s_2} + t_2$
- $\log|J| = s_2$

Second layer mask $[0,1]$ updates $z_1$ using $z_2$. After two layers, both coordinates transformed — Jacobian product is $e^{s_2^{(1)}} \cdot e^{s_1^{(2)}}$.

---

## Image-Ready Coupling (Mental Model)

Replace Dense with Conv2D; replace half-vector mask with **checkerboard**:

```
x x o x    x = updated this layer
o x o x    o = passed through unchanged
x o x o
```

Alternate checkerboard patterns between layers — GLOW's spatial analogue of flipping $[1,0]$ / $[0,1]$ on moons.

---

## Key Vocabulary

| Term | Definition |
|------|------------|
| **RealNVP** | Flow model using affine coupling layers |
| **Coupling layer** | Split-transform recombine with tractable Jacobian |
| **Scale $s$ / translation $t$** | Affine parameters from a sub-network |
| **Masking** | Zeroing dimensions so only a subset is transformed |
| **Lower-triangular Jacobian** | Structure making $\det J$ a simple product |

---

## Reflection Questions

1. Why must the first $d$ components of $x$ pass through unchanged in one coupling layer?
2. How does alternating masks ensure all dimensions are eventually updated?
3. Why is $\log |\det J|$ equal to $\sum_j s_j$ for this architecture?
4. What breaks if you use a standard ReLU MLP without invertible structure?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 6 — RealNVP. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Dinh, L. et al. (2017). Density Estimation using Real NVP. [https://arxiv.org/abs/1605.08803](https://arxiv.org/abs/1605.08803)
- Foster's notebook: `notebooks/06_normflow/01_realnvp/realnvp.ipynb`

---

**Previous:** [Section 6.2 — Change of Variables](./section-02-change-of-variables.md)  
**Next:** [Section 6.4 — Two Moons Dataset](./section-04-two-moons-dataset.md)
