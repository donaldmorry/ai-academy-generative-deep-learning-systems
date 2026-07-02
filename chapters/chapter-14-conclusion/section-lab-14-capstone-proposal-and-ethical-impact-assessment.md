# Lab 14: Capstone Proposal & Ethical Impact Assessment

> **Prerequisites:** Sections 14.1-14.8
> **Estimated time:** 8-10 hours
> **Tools:** Python 3.10+, Jupyter, scikit-learn, optional TensorFlow 2.x / Foster GDL_code
> **Glossary:** [GLOSSARY.md](../../GLOSSARY.md) | **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Lab Objectives

1. Draft a capstone proposal: modality, architecture choice, dataset, and evaluation plan.
2. Map your chosen architecture to specific course chapters and Foster chapters.
3. Complete an ethical impact assessment: identify 5 risks and mitigation strategies.
4. Build a comparison matrix of generative method families for your use case.
5. *Deliverable:* 2-page capstone proposal with ethics appendix and architecture justification.

### Math You'll Use

$$
P_\theta(x) \rightarrow \text{sample } x'
$$
> **Readable form:** All generative families learn distributions to create new data.


$$
\text{Cost} \propto N_{\text{params}} \times N_{\text{tokens}}
$$
> **Readable form:** Training cost scales with model size and data.


---

Turn the objectives into a concrete artifact:

| Proposal section | Required content |
|------------------|------------------|
| Problem | User need, modality, input/output examples |
| Architecture | Chosen generative family and rejected alternatives |
| Data | Dataset source, license, preprocessing, privacy risks |
| Evaluation | Quality metrics, human review plan, failure cases |
| Ethics | Misuse, bias, consent, disclosure, rollback plan |

---

## Part A: Core Implementation

Build the architecture comparison table that anchors your capstone decision. Use at least three model families from Chapters 03-13 and one modern LLM-system option from Section 14.9.

| Candidate | Best fit | Data need | Compute need | Main risk | Decision |
|-----------|----------|-----------|--------------|-----------|----------|
| VAE | Structured latent exploration | Medium | Low-medium | Blurry samples | |
| GAN | Sharp image generation | High | Medium-high | Instability, mode collapse | |
| Diffusion | High-fidelity images/audio | High | High | Slow sampling, misuse | |
| Transformer | Text, code, sequences | High | High | hallucination, context limits | |
| RAG/LoRA | Domain LLM system | Low-medium | Low-medium | retrieval or adapter drift | |

**Task:** Fill the table for your capstone. Then write a one-paragraph decision memo that names the chosen architecture, the rejected runner-up, and the experiment that would change your mind.

---

## Part B: Evaluation & Diagnostics

```python
import matplotlib.pyplot as plt

# Plot training history
if 'history' in dir():
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val')
    plt.legend(); plt.title('Loss')
    plt.savefig(f'lab_{mod_num}_loss.png', dpi=150)
```

**Deliverable:** At least two figures - training curves and generated samples or reconstructions.

---

## Part C: Written Reflection

Answer in a markdown cell or separate document:

1. What was the hardest part of **capstone proposal & ethical impact assessment** to implement?
2. Which section topic proved most useful during the lab?
3. How would sample quality change with 10× more training data?
4. What generative chapter (03-13) shares the most DNA with this lab?

---

## Part D: Extended Exercises

Complete at least two exercises for **Capstone Proposal & Ethical Impact Assessment**:

1. **Hyperparameter sweep** - Vary learning rate or batch size; plot validation loss.
2. **Ablation** - Remove one component and compare outputs.
3. **Cross-chapter comparison** - Contrast with a model from an earlier chapter.
4. **Failure log** - Document one failed run and the fix applied.

```python
import tensorflow as tf
tf.random.set_seed(42)
print('GPU devices:', tf.config.list_physical_devices('GPU'))
```

---

## Part E: Modern LLM Mini-Lab

Use [Section 14.9](./section-09-modern-llm-engineering.md) to add one concrete LLM-system experiment to the capstone proposal. Choose **RAG**, **LoRA**, or both depending on your hardware.

### Option 1: Tiny RAG Evaluator

Create a small retrieval corpus from three course section files and test whether retrieval improves grounded answers.

```python
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

lesson_paths = [
    Path("../09-transformers/01-transformer-introduction.md"),
    Path("./02-current-state-llms.md"),
    Path("./09-modern-llm-engineering.md"),
]

docs = [path.read_text(encoding="utf-8") for path in lesson_paths]
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X = vectorizer.fit_transform(docs)

def retrieve(query, k=2):
    q = vectorizer.transform([query])
    scores = cosine_similarity(q, X)[0]
    ranked = scores.argsort()[::-1][:k]
    return [(lesson_paths[i].name, float(scores[i])) for i in ranked]

print(retrieve("When should I use retrieval instead of fine-tuning?"))
```

Write a five-row evaluation table:

| Query | Expected source | Retrieved source | Grounded? | Failure note |
|-------|-----------------|------------------|-----------|--------------|
| RAG vs fine-tuning | Section 14.9 | | | |
| Transformer attention | Chapter 09 | | | |
| Model deployment risk | Section 14.9 | | | |
| Capstone ethics | Lab 14 | | | |
| Out-of-scope question | None | | | |

### Option 2: LoRA Decision Memo

If you have GPU access and the `peft` stack installed, run a small adapter fine-tune on a toy instruction dataset. If not, write the engineering memo without training.

```python
lora_decision = {
    "base_model": "small open-weight instruct model or course-approved API model",
    "task": "domain-specific answer style for the capstone",
    "why_not_prompt_only": "prompting fails on repeated style or domain terms",
    "why_not_full_finetune": "adapter is cheaper and easier to roll back",
    "evaluation": ["held-out prompts", "grounding checks", "toxicity/safety review"],
}

for key, value in lora_decision.items():
    print(f"{key}: {value}")
```

Deliver one paragraph comparing:

| Choice | Use when | Main risk |
|--------|----------|-----------|
| Prompting | Behavior can be specified in context | Fragile on long or varied tasks |
| RAG | Missing knowledge is the problem | Retrieval can surface wrong context |
| LoRA | Repeated behavior or format must be learned | Adapter can overfit or drift |
| Full fine-tune | You own data, budget, and deployment | Cost, rollback, and safety burden |

Add the chosen path to the capstone proposal as a concrete architecture decision, not a buzzword.

---

## Troubleshooting

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| OOM on GPU | Batch too large | Halve batch size |
| Loss flat | LR wrong | Try 1e-4, 1e-3, 1e-2 |
| Poor samples | Under-training | Train longer; review Foster tips |

---

## Common Lab Mistakes

| Mistake | What Goes Wrong | Fix |
|---------|-----------------|-----|
| Skipping GPU check | Hours of CPU training | `tf.config.list_physical_devices('GPU')` |
| No random seed | Irreproducible results | Set `tf.random.set_seed(42)` and `np.random.seed(42)` |
| Wrong tensor shapes | Immediate crash | Print shapes after every layer |
| Not saving checkpoints | Lost work on crash | Use `ModelCheckpoint` callback |

---

## Submission Checklist

- [ ] Code runs end-to-end without errors
- [ ] Figures saved with clear labels
- [ ] Short write-up (200-400 words) interpreting results
- [ ] Connection to at least two section topics from this chapter

---

## Further Reading

- Foster, Ch. 14 - *Conclusion*
- [Chapter 14 README](./README.md)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Extended Notes

Re-read Chapter 14 after completing the lab. Focus on assumptions that make **capstone proposal & ethical impact assessment** valid.

### Practice checklist

- [ ] Run the Foster notebook for **capstone proposal & ethical impact assessment** end-to-end
- [ ] Save at least one diagnostic figure
- [ ] Write a 3-sentence summary in your own words
- [ ] Link this section to one later chapter

### Bridge to generative AI

Modern systems (diffusion models, LLMs, VAEs) reuse the ideas in **capstone proposal & ethical impact assessment**.
The vocabulary changes; the mathematics rhymes.

### Historical context

Foster situates **capstone proposal & ethical impact assessment** within the evolution from VAEs and GANs to transformers.

---

## Extended Notes
**Previous:** [Section 14.8 - Capstone Planning & Final Thoughts](./section-08-capstone-planning-and-final-thoughts.md)
**Next:** [Chapter 14 README](./README.md)

---

## Assessment Practice

Use the shared [Assessment Appendix](../../ASSESSMENT_APPENDIX.md) for concept audits, worked examples, implementation checks, experiment logs, oral-exam prompts, and deliverable checklists.
