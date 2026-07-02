# Shared Assessment Appendix

This appendix holds the reusable assessment prompts that apply across the AI learning courses. Individual sections should keep topic-specific explanations, examples, references, and checks in the section body, then link here for repeated mastery practice.

---

## Concept Audit

Answer these without looking back at the section:

1. What problem does the concept solve?
2. What representation does it require?
3. What computation changes the input into the output?
4. What assumption would break first in a messy real system?
5. Which earlier chapter provides the strongest prerequisite?
6. Which later chapter uses the idea at larger scale?

---

## Worked Example Protocol

Build a small example with numbers, symbols, tokens, states, or observations that fit on one page.

| Step | Requirement |
|------|-------------|
| Setup | Define every variable and unit before computing |
| Baseline | Solve the naive or simpler version first |
| Main method | Apply the section method one step at a time |
| Check | Verify dimensions, probabilities, utilities, or invariants |
| Perturb | Change one input and predict how the output should move |
| Explain | Translate the result into agent, model, or system behavior |

---

## Implementation Protocol

If the section includes an algorithm, implement the smallest version with tests before using a library abstraction.

```python
def sanity_check(result, expected, tolerance=1e-6):
    if isinstance(result, (int, float)):
        return abs(result - expected) <= tolerance
    return result == expected
```

Use the check on one ordinary case, one boundary case, and one intentionally invalid case.

---

## Diagnostic Questions

1. What metric would improve if the idea works?
2. What plot, trace, proof step, or sample would reveal a bug?
3. What would a skeptical reviewer ask about the result?
4. What is the simplest ablation that should make performance worse?
5. What should be logged so the experiment is reproducible?

---

## Transfer Prompt

Apply the section idea to a new domain. Specify the domain, rewrite the variables, and state what stays invariant across domains.

| Domain change | What to revisit |
|---------------|-----------------|
| More data | Validation strategy and compute cost |
| Noisier input | Uncertainty model or robustness check |
| More agents/users | Incentives, coordination, and safety constraints |
| Higher stakes | Monitoring, rollback, and human review |
| Different modality | Representation and preprocessing pipeline |

---

## Rubric

| Level | Evidence of mastery |
|-------|---------------------|
| 1 | Can define the term |
| 2 | Can work a toy example |
| 3 | Can implement and test the toy example |
| 4 | Can diagnose failure modes |
| 5 | Can decide when not to use the method |

---

## Portfolio Note

Write a half-page memo with four paragraphs: concept, implementation, failure mode, and connection. Keep it concrete enough that a future you could rebuild the example without reopening the textbook.

---

## Final Checks

- I can state the core idea in one sentence.
- I can identify the input and output types.
- I can name one computational bottleneck.
- I can name one evaluation risk.
- I can connect the section to a project I might build.

---

## Review Project

Turn the section into a small review project. The project should be narrow enough to finish in one sitting but concrete enough to expose misunderstandings.

| Artifact | What it should contain |
|----------|------------------------|
| Problem statement | One paragraph naming inputs, outputs, and success criteria |
| Diagram | Boxes for data, representation, computation, output, and evaluation |
| Toy case | A fully specified example with expected answer |
| Baseline | The simplest competing method or explanation |
| Main result | A calculation, trace, plot, proof, or generated sample |
| Failure case | An input where the method struggles or should refuse |
| Reflection | Two sentences about what changed in your understanding |

---

## Experiment Log Template

| Field | Notes |
|-------|-------|
| Date and chapter | Keep the learning path auditable |
| Source concept | Name the exact section idea |
| Dataset or toy input | State size, units, and assumptions |
| Baseline result | Record the simplest comparison |
| Method result | Record the section method outcome |
| Difference | Explain why the method changed the result |
| Runtime or complexity | Name the growth driver |
| Failure observed | Include at least one concrete failure |
| Next experiment | Make the next action specific |
| Confidence | Low, medium, or high, with a reason |

---

## Error Analysis Menu

| Error class | Question to ask |
|-------------|-----------------|
| Data error | Was the input malformed, biased, or out of distribution? |
| Representation error | Did the chosen variables omit a necessary factor? |
| Objective error | Did the metric reward the wrong behavior? |
| Search error | Did the algorithm miss a better candidate? |
| Optimization error | Did training stop at a poor solution? |
| Inference error | Did approximation introduce visible bias? |
| Evaluation error | Did the test set fail to represent use? |
| Communication error | Would a user misunderstand the output? |

---

## Oral Exam Bank

1. Explain the section using a concrete example from your own domain.
2. State the key assumption and defend why it is acceptable in the toy case.
3. Give one counterexample where the method would fail.
4. Compare the method with a simpler baseline.
5. Describe how you would test a production implementation.
6. Name the most important symbol, variable, or data structure.
7. Explain the computational bottleneck.
8. Connect the section to an earlier prerequisite.
9. Connect the section to a later advanced topic.
10. Summarize the idea for a non-technical stakeholder.

---

## Deliverables Checklist

- One-paragraph explanation.
- One hand-worked example.
- One code or pseudocode sketch.
- One diagnostic or test.
- One failure case.
- One cross-course connection.
- One open question to revisit.
