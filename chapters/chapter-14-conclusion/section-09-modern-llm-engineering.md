# Section 14.9: Modern LLM Engineering

> **Source inheritance:** Post-Foster engineering supplement for 2024-2026 production practice  
> **Prerequisites:** [Section 14.8](./section-08-capstone-planning-and-final-thoughts.md), [Chapter 09 Transformers](../chapter-09-transformers/README.md), [Chapter 13 Multimodal Models](../chapter-13-multimodal-models/README.md)  
> **Vocabulary:** [RAG](../../GLOSSARY.md#retrieval-augmented-generation-rag) | [LoRA](../../GLOSSARY.md#lora) | [QLoRA](../../GLOSSARY.md#qlora) | [DPO](../../GLOSSARY.md#dpo) | [KV cache](../../GLOSSARY.md#kv-cache)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Why This Supplement Exists

Foster's second edition gives you the foundations: VAEs, GANs, flows, EBMs, diffusion, transformers, Stable Diffusion, CLIP, and multimodal generation. That is enough to understand the model families. It is not enough by itself to ship a 2026 LLM application.

Modern generative AI engineering is less about training a frontier model from scratch and more about composing pretrained models with data, retrieval, adapters, evaluation, safety gates, and inference constraints. The practical question changes from "Can I implement the architecture?" to "Can I adapt, ground, evaluate, serve, and monitor the model without breaking user trust or the budget?"

This section is the bridge from Course 4 foundations to production LLM systems.

---

## Learning Targets

By the end of this section, you should be able to:

1. Explain why [RAG](../../GLOSSARY.md#retrieval-augmented-generation-rag) is often preferred before fine-tuning.
2. Choose between prompting, RAG, [LoRA](../../GLOSSARY.md#lora), [QLoRA](../../GLOSSARY.md#qlora), and full fine-tuning.
3. Describe the role of RLHF, [DPO](../../GLOSSARY.md#dpo), and preference data in alignment.
4. Name core inference optimizations: [KV cache](../../GLOSSARY.md#kv-cache), quantization, batching, speculative decoding, and efficient attention.
5. Design an evaluation harness for answer quality, grounding, latency, and safety.
6. Connect agentic tool use back to Course 2 rational-agent concepts.

---

## The Production Stack

| Layer | Typical choice | Failure to watch |
|-------|----------------|------------------|
| Base model | Hosted API or open-weight checkpoint | Mismatch between model capability and task risk |
| Adaptation | Prompting, RAG, LoRA/QLoRA, full fine-tune | Overfitting, stale facts, hidden cost |
| Data | Documents, examples, preference pairs, logs | Leakage, poor provenance, privacy exposure |
| Serving | Quantized inference, batching, KV cache | Latency spikes, memory limits, quality regressions |
| Evaluation | Golden sets, human review, LLM judge with checks | Metric gaming, ungrounded claims |
| Safety | Policy filters, refusal tests, audit logs | Silent unsafe behavior under edge prompts |
| Monitoring | Drift, feedback, incident review | No way to detect degradation |

The stack is intentionally layered. Do not fine-tune before you know whether retrieval or a better prompt would solve the problem.

---

## Decision Ladder

Start with the least invasive method that can meet the requirement.

| Need | First method | Escalate when |
|------|--------------|---------------|
| Better instructions | Prompt template | Failures are consistent despite clear prompts |
| Private or changing facts | RAG | Retrieval is insufficient or the model must learn style |
| Domain style or format | LoRA | Adapter quality saturates or domain shift is deep |
| Low-memory adaptation | QLoRA | Quantized training hurts critical outputs |
| Behavior preference | DPO / RLHF | You have reliable preference data |
| New capability | Full fine-tune or new model | Base model cannot perform the task |

The ladder prevents expensive heroics. A model that answers from retrieved policy documents is easier to update than a model that memorized a policy last quarter.

---

## RAG: Ground the Model Before You Change It

Retrieval-augmented generation adds a search step before generation:

1. Split documents into chunks.
2. Embed chunks into vectors.
3. Retrieve top-k chunks for a user query.
4. Compose a prompt with the retrieved context.
5. Generate an answer with citations or source references.
6. Evaluate both retrieval quality and answer quality.

$$
p(y \mid q, D) \approx p_\theta(y \mid q, r_k(q, D))
$$
> **Readable form:** answer probability is approximated by conditioning the model on the query plus the retrieved chunks from the document collection.

RAG is strongest when facts change, sources matter, or the domain corpus is too large to fit into the context window.

---

## RAG Failure Modes

| Failure mode | Symptom | Fix |
|--------------|---------|-----|
| Bad chunking | Relevant answer split across chunks | Add semantic chunking or overlap |
| Embedding mismatch | Search returns fluent but irrelevant text | Test embeddings on labeled query-document pairs |
| Lost-in-context | Correct chunk retrieved but ignored | Rerank, compress, or quote only the useful span |
| Hallucinated citation | Answer cites source that does not support claim | Require claim-source alignment checks |
| Stale index | System answers from old docs | Add index freshness and document version tests |

Evaluate retrieval separately. A perfect generator cannot recover evidence that retrieval never supplied.

---

## Parameter-Efficient Fine-Tuning

Full fine-tuning updates all weights. [LoRA](../../GLOSSARY.md#lora) freezes the base weights and learns low-rank update matrices:

$$
W' = W + \Delta W = W + \frac{\alpha}{r} BA
$$
> **Readable form:** keep the original weight matrix fixed and learn a small low-rank correction scaled by alpha over rank.

This works because many downstream changes can be represented by low-rank adjustments. You store small adapters instead of a full model copy.

[QLoRA](../../GLOSSARY.md#qlora) adds quantization: the frozen base model is loaded in low precision while the adapter trains in higher precision where needed. It makes single-GPU domain adaptation more realistic, but quantization can expose edge-case quality loss.

---

## PEFT Sketch

```python
# pip install transformers peft accelerate bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

base = "open-weight-or-internal-checkpoint"
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
```

The important check is the trainable-parameter fraction. If most weights are trainable, you are no longer doing parameter-efficient adaptation.

---

## Preference Optimization

Instruction tuning teaches the model the task format. Preference optimization teaches it which acceptable answer is better.

RLHF trains a reward model from human preferences and then optimizes the policy against that reward. [DPO](../../GLOSSARY.md#dpo) removes the explicit reward-model training loop and optimizes directly from chosen/rejected pairs.

$$
\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \left(\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right)\right]
$$
> **Readable form:** increase the policy's relative preference for the winning answer over the losing answer while staying anchored to a reference model.

Preference data quality dominates algorithm choice. No alignment method can reliably learn values that annotators did not express consistently.

---

## Inference Optimization

Training is not the only bottleneck. For deployed LLMs, inference often dominates cost.

| Technique | What it saves | Tradeoff |
|-----------|---------------|----------|
| KV cache | Recomputing previous attention keys/values | More memory per active sequence |
| Quantization | Weight and activation memory | Possible quality drop |
| Continuous batching | Idle GPU time | More scheduling complexity |
| Speculative decoding | Serial token latency | Needs draft model or draft head |
| FlashAttention-style kernels | Attention memory traffic | Hardware/kernel compatibility |
| Paged attention | Fragmentation in serving many users | More specialized serving stack |

For a product, measure time-to-first-token, tokens/sec, peak memory, and failure rate under concurrent load.

---

## KV Cache Intuition

Autoregressive generation appends one token at a time. Without caching, the model recomputes attention keys and values for the whole prefix at every step. With a [KV cache](../../GLOSSARY.md#kv-cache), it reuses prior keys and values:

$$
K_{1:t} = [K_{1:t-1}; K_t], \quad V_{1:t} = [V_{1:t-1}; V_t]
$$
> **Readable form:** append the new token's key and value to the stored prefix cache instead of recomputing the entire prefix.

This speeds decoding but creates memory pressure. Long context windows can become a serving problem even when the model "supports" them.

---

## Long Context Is Not Just a Bigger Window

Long-context systems use techniques such as RoPE scaling, ALiBi-like biases, retrieval, summarization, memory compression, and hierarchical routing. The key engineering question is whether the model can use distant information, not merely accept many tokens.

Test long context with adversarial placement:

1. Put the answer near the beginning, middle, and end.
2. Add distractor facts that look plausible.
3. Ask for cited evidence.
4. Measure whether accuracy changes by position.
5. Track latency and cost as context grows.

RAG plus shorter focused context often beats a huge context window full of irrelevant material.

---

## Mixture of Experts

[Mixture of Experts](../../GLOSSARY.md#mixture-of-experts-moe) routes each token through a subset of expert feed-forward networks:

$$
h' = \sum_{i \in \operatorname{TopK}(g(h))} g_i(h) E_i(h)
$$
> **Readable form:** a router chooses a few experts for the token, weights their outputs, and skips the rest.

MoE increases parameter count without activating every parameter for every token. Serving is harder: expert routing, load balancing, memory placement, and batching become central engineering concerns.

---

## Agents Are Course 2 Again

LLM agents add tools, memory, planning loops, and action selection. The vocabulary is new; the shape is AIMA:

| Agent concept | LLM system version |
|---------------|--------------------|
| Percept | User message, retrieved context, tool result |
| State | Conversation, scratchpad, memory, environment record |
| Action | Tool call, code execution, retrieval query, final answer |
| Performance measure | Task success, safety, cost, latency, user rating |
| Policy | Prompt, planner, controller, model weights |

Use tools when the task needs external truth, computation, or side effects. Use the model alone when language reasoning is sufficient and the risk is low.

---

## Evaluation Harness

Production LLM evaluation needs more than "looks good."

| Dimension | Test |
|-----------|------|
| Task quality | Golden prompts with expected properties |
| Grounding | Claim-source entailment checks |
| Retrieval | Recall@k and mean reciprocal rank |
| Safety | Red-team prompts and refusal boundary tests |
| Latency | p50/p95 time-to-first-token and total time |
| Cost | Tokens per successful task |
| Robustness | Prompt injection, noisy input, out-of-domain queries |
| Regression | Fixed seed or fixed-output comparison where possible |

LLM-as-judge can help triage, but calibrate it against human labels and include examples where the judge is expected to fail.

---

## Capstone Upgrade

If your Course 4 capstone uses an LLM, add this production slice:

1. Baseline prompt-only result.
2. RAG or adapter experiment with an ablation.
3. Evaluation set with at least 30 realistic cases.
4. Latency and cost measurements.
5. Safety and privacy review.
6. Monitoring plan: what will you log, alert on, and roll back?
7. Model card addendum: known limitations and maintenance owner.

This converts a demo into an engineering artifact.

---

## Common Mistakes

| Mistake | Why it hurts | Better move |
|---------|--------------|-------------|
| Fine-tuning for fresh facts | Model facts go stale | RAG with document versioning |
| Evaluating only happy paths | Real users hit edge cases | Build adversarial and ambiguous prompts |
| Ignoring retrieval metrics | Generator gets blamed for search failure | Measure retrieval before generation |
| Using a giant context by default | Slow and expensive | Retrieve, compress, or summarize |
| Treating LLM judges as truth | Judges share model biases | Calibrate with human-labeled subsets |
| Skipping rollback | Bad model stays live | Version prompts, adapters, indexes, and safety configs |

---

## Self-Check

1. When would you choose RAG over LoRA?
2. What does LoRA update, and what stays frozen?
3. Why can QLoRA reduce hardware requirements?
4. What is the difference between RLHF and DPO?
5. Why does KV caching increase memory use?
6. What should you measure before claiming a model is production-ready?
7. How does an LLM agent map onto the rational-agent framework from Course 2?

---

## References

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" - [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" - [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
- Rafailov et al., "Direct Preference Optimization" - [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
- Lewis et al., "Retrieval-Augmented Generation" - [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
- Dao et al., "FlashAttention" - [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)

---

**Previous:** [Section 14.8 - Capstone Planning & Final Thoughts](./section-08-capstone-planning-and-final-thoughts.md)
