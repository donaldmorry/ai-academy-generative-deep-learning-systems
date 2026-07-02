# Section 12.2: Reinforcement Learning Basics

> **Source inheritance:** Foster, *Generative Deep Learning* (2nd ed.), World Models
> **Enhanced with:** original explanations, implementation workflow, diagnostics, failure modes, and assessment prompts
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## The Core Idea

**Reinforcement Learning Basics** is part of the course's advanced generative AI arc. The source chapter supplies the motivating architecture and examples; this section turns them into a reusable engineering pattern for **world models for reinforcement learning**.

The practical objective is to **learn a compact latent simulator so an agent can plan or train inside imagined rollouts**. Keep that objective visible while reading the architecture details. Most mistakes in generative modeling come from optimizing a proxy while forgetting what the generated artifact is supposed to do for a user, evaluator, or downstream system.

This section is written as original course material rather than extracted textbook prose. Use the source book for the author's full narrative and this file for implementation-ready structure, diagnostics, and active-learning prompts.

---

## Learning Targets

By the end of this section, you should be able to:

1. Explain the role of **Reinforcement Learning Basics** in world models for reinforcement learning.
2. Identify the training objective and the sampling procedure.
3. Sketch the data representation used by the model.
4. Name at least three diagnostics that reveal whether the method is working.
5. Describe one failure mode and one mitigation.
6. Connect the section to an earlier course concept.

---

## Source-Aligned Concept Map

| Component | What it means here | What to check |
|-----------|--------------------|---------------|
| Data representation | observation, action, reward, and done sequences collected from an environment policy | Shapes, value ranges, masks, and vocabulary |
| Model family | world models for reinforcement learning | Whether inductive bias matches the artifact |
| Objective | learn a compact latent simulator so an agent can plan or train inside imagined rollouts | Loss terms, sampling assumptions, and evaluation proxy |
| Sampling path | Convert learned parameters into a generated artifact | Random seed, temperature/noise, constraints, and postprocessing |
| Diagnostics | reconstruction quality, rollout error, reward prediction, controller return, and sim-to-real gap | Quantitative curves plus qualitative inspection |

The concept map is deliberately compact. When implementing the section, expand each row into a checklist in your experiment log.

---

## Mathematical Anchors

$$
z_t = E_\phi(x_t)
$$
> **Readable form:** The encoder compresses a high-dimensional observation into a latent state.

$$
h_t = R_\theta(h_{t-1}, z_t, a_{t-1})
$$
> **Readable form:** The recurrent model updates memory using the previous latent state and action.

$$
a_t = \pi_\psi(z_t, h_t)
$$
> **Readable form:** The controller chooses actions from the imagined state rather than raw pixels.

These equations are not decorative. Before running a notebook, point to the line of code that implements each mathematical object. If you cannot find that line, the implementation is still a black box.

---

## Implementation Workflow

1. **Pin the representation.** Decide exactly how examples become tensors or token IDs.
2. **Build the smallest model first.** Use tiny dimensions and a tiny dataset until one batch overfits.
3. **Verify the loss.** Check shapes, masking, reductions, and whether the sign matches the objective.
4. **Sample early.** Low-quality samples are fine; invalid samples expose bugs.
5. **Add capacity gradually.** Increase layers, channels, heads, latent size, or rollout horizon only after the baseline behaves.
6. **Log artifacts.** Save generated examples, metric curves, config, seed, and code version.
7. **Write a failure note.** Every run should end with one hypothesis for the next run.

---

## Implementation Sketch

```python
def imagine_rollout(z0, h0, controller, dynamics, horizon):
    z, h = z0, h0
    trajectory = []
    for _ in range(horizon):
        action = controller(z, h)
        z, h, reward = dynamics(z, h, action)
        trajectory.append((z, action, reward))
    return trajectory
```

Treat this as a scaffold. The important parts are explicit shapes, named objectives, and a return value that can be logged. Production code should add validation, checkpointing, mixed precision only when stable, and deterministic evaluation runs.

---

## Data and Preprocessing Checklist

| Check | Why it matters |
|-------|----------------|
| Train/validation split is fixed | Prevents silent leakage across experiments |
| Input range is documented | Generators and decoders often assume a specific scale |
| Token or label vocabulary is saved | Sampling must use the same mapping as training |
| Masks are tested on toy examples | Incorrect masking can leak future information |
| Batch shape is printed once | Most generative bugs are shape bugs wearing a nicer hat |
| Postprocessing is reversible where possible | You need to inspect whether errors come from model or decoder |

---

## Diagnostics

Use at least one quantitative and one qualitative diagnostic.

| Diagnostic | Healthy signal | Warning signal |
|------------|----------------|----------------|
| Training curve | Smooth trend with explainable noise | Flatline, spikes, or sudden NaNs |
| Validation proxy | Improves before overfitting | Improves while samples get worse |
| Sample grid or trace | Diversity with recognizable structure | Repetition, collapse, or invalid outputs |
| Ablation | Removing a component hurts as expected | Component appears irrelevant |
| Seed sweep | Similar conclusions across seeds | One lucky seed carries the claim |

---

## Failure Modes

| Failure mode | Typical cause | Practical response |
|--------------|---------------|--------------------|
| Invalid samples | Representation or constraint mismatch | Add validity checks before quality metrics |
| Mode collapse or repetition | Objective rewards a narrow high-confidence region | Add diversity diagnostics and rebalance training |
| Overconfident artifacts | Sampling temperature/noise too low | Sweep sampling parameters against validation examples |
| Slow iteration | Model too large before the pipeline is proven | Shrink the experiment until debugging is cheap |
| Misleading metric | Proxy ignores user-visible quality | Pair metric curves with blind qualitative review |

---

## Worked Mini-Lab

Complete this short lab before moving to the full chapter notebook:

1. Create a synthetic dataset with 32 examples.
2. Run the preprocessing pipeline and print one decoded example.
3. Train for enough steps to overfit a single batch.
4. Generate or roll out five samples with fixed seeds.
5. Record one metric and one visual or textual artifact.
6. Change one hyperparameter and write a two-sentence comparison.

The goal is not performance. The goal is to prove the training loop, sampling loop, and evaluation loop all agree on the same representation.

---

## Design Trade-Offs

| Decision | Benefit | Cost |
|----------|---------|------|
| Larger model | More expressive distribution | More compute and harder debugging |
| Stronger regularization | Better stability | Possible loss of fine detail |
| Richer conditioning | More controllable outputs | More data and interface complexity |
| Longer context or horizon | Better global structure | Higher memory and latency |
| Heavier evaluation | More trustworthy claims | Slower iteration |

---

## Cross-Course Connections

- **Course 1:** operational discipline, validation splits, and deployment thinking.
- **Course 2:** agents, uncertainty, decision-making, and safety framing.
- **Course 3:** optimization, representation learning, approximate inference, and generative modeling theory.
- **Course 4:** this section composes with earlier VAEs, GANs, flows, diffusion models, and transformers.

---

## Reflection Questions

1. What distribution, policy, or simulator is being learned in this section?
2. Which tensor or token representation carries the most important inductive bias?
3. What would a one-batch overfit test prove?
4. Which diagnostic would you trust least by itself?
5. What is the most likely failure mode when scaling this method?
6. How would you explain the section to someone who has only completed Course 1?

---

## References

- Ha, D., & Schmidhuber, J. (2018). World Models. [https://arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)
- Foster, D. (2023). *Generative Deep Learning* (2nd ed.), Chapter 12.
- Foster's codebase: [https://github.com/davidADSP/GDL_code](https://github.com/davidADSP/GDL_code)

---

**Previous:** [Section 12.1: World Models Introduction](./section-01-world-models-introduction.md)  
**Next:** [Section 12.3: World Model Architecture](./section-03-world-model-architecture.md)




---

## Assessment Practice

Use the shared [Assessment Appendix](../../ASSESSMENT_APPENDIX.md) for concept audits, worked examples, implementation checks, experiment logs, oral-exam prompts, and deliverable checklists.
