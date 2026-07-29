# Research & Model Selection — Replacing GPT-4 Mini for Resume Intelligence

## 1. Task Definition

Before picking a model, define what "replacing GPT-4 Mini" actually requires for this platform:

| Task | Characteristics |
|---|---|
| Resume parsing / extraction | Long-ish input (1-2 pages of text), structured JSON output, needs strict format-following, low creativity |
| Resume generation | Medium-length generation, needs fluency and instruction-following |
| Cover letter generation | Similar to above, more "creative" tone control |
| Recruitment workflow reasoning (matching, ranking) | Needs reasonable general reasoning, not frontier-level |

None of these require frontier-level reasoning (no multi-step math, no advanced coding). They require **strong instruction-following, reliable structured-output generation, and decent general English fluency** — which small (2B–8B) open models handle well after fine-tuning. This is the central argument for *not* needing a 70B+ model.

## 2. Chosen Model: **Qwen2.5-3B-Instruct**

### Why this model
- **Instruction-tuned out of the box** — Qwen2.5-Instruct models are RLHF/SFT-aligned for chat and structured tasks, so they follow "extract into this JSON schema" style prompts well even before any fine-tuning of ours.
- **Strong benchmark performance for its size.** Qwen2.5 3B outperforms most other sub-4B open models (Phi-3.5-mini, Gemma-2-2B) on instruction-following and structured-generation benchmarks, and is competitive with some 7B models from a generation earlier.
- **Excellent JSON / structured output behavior.** Qwen2.5 was explicitly trained with more structured-data (tables, JSON, code) in its pretraining mix than earlier open models, which directly matters for a parsing task.
- **128K context window** — resumes are short, but this headroom lets us do multi-resume batch prompts, few-shot examples in-context, or RAG-augmented job-matching without truncation.
- **Practical fit with existing infra** — we already have `qwen2.5:3b` pulled via Ollama, meaning zero extra setup cost for local inference/dev, and Ollama + Qwen2.5 GGUF quantizations are well supported.
- **Small enough to fine-tune and serve cheaply**, large enough to have genuine language competence (unlike sub-1B models which tend to hallucinate structure).

### Parameter size
3.09B parameters (dense, not MoE).

### Hardware requirements
| Mode | Requirement |
|---|---|
| Inference, fp16 | ~6-7 GB VRAM |
| Inference, 4-bit (GGUF/Q4) | ~2-2.5 GB VRAM or even CPU RAM (what Ollama uses by default) |
| QLoRA fine-tuning (4-bit base + LoRA adapters) | ~8-10 GB VRAM (fits a single T4/RTX 3060/Colab free-tier GPU) |
| Full fine-tuning | ~40+ GB VRAM — not recommended, not needed |
| CPU-only inference | Feasible for low QPS internal tools (slow, ~2-5 tok/s), not for production API load |

This matters commercially: GPT-4 Mini is a metered per-token API cost. A 3B model that we own can be self-hosted on a single low/mid-tier GPU instance (e.g., AWS `g4dn.xlarge` or GCP `T4` instance), turning a variable API cost into a fixed, much lower infra cost at any real volume.

### Context window
32,768 tokens native (some Qwen2.5 variants extend to 128K via YaRN). Comfortably fits full multi-page resumes, few-shot prompt exemplars, and RAG-retrieved job description context in a single call.

### License
Apache 2.0 (for the 0.5B/1.5B/3B/7B/14B/32B sizes — note the 72B model uses a separate, more restrictive Qwen license). Apache 2.0 is fully permissive: commercial use, modification, and redistribution of fine-tuned weights are all allowed without royalty or copyleft obligation. This is important for a startup building a product on top of it — no legal ambiguity, unlike some "open" models with non-commercial or revenue-cap clauses (e.g., some Llama community license restrictions for very large-scale deployments).

### Pros
- Best-in-class small-model instruction-following and structured output today among sub-4B open models.
- Truly permissive license (Apache 2.0) — no usage-tier restrictions.
- Strong multilingual support (resumes come in mixed languages/names) — Qwen's pretraining corpus has heavy multilingual coverage, better than Llama 3.2 3B or Phi at non-English names/text fragments.
- Mature tooling: first-class support in Hugging Face Transformers, PEFT, vLLM, Ollama, llama.cpp.
- Cheap to fine-tune (QLoRA fits on free-tier Colab T4).
- Already validated locally via Ollama — reduces integration risk.

### Cons
- 3B models still hallucinate on ambiguous/adversarial input more than GPT-4 Mini; needs a strict JSON-schema validation + retry layer around it (covered in Part 4).
- Weaker long-form reasoning than GPT-4 Mini for complex recruiter-workflow logic (e.g., nuanced candidate ranking with soft criteria) — may still need a larger model or GPT-4 Mini as a fallback for the highest-stakes reasoning tasks.
- Smaller community fine-tuning precedent than Llama/Mistral — fewer existing LoRA recipes/blog posts to lean on if something breaks.
- Base multi-turn conversational polish is slightly behind Llama 3.1/3.2 Instruct for open-ended chat (less relevant here since our tasks are mostly single-turn extraction/generation).

## 3. Why Not the Other Popular Options

| Model | Why not chosen (for this task) |
|---|---|
| **Llama 3.2 3B Instruct** | Very close competitor, genuinely viable alternative. Slightly behind Qwen2.5-3B on structured-output/JSON benchmarks in most public evals, and Meta's license carries a use-restriction clause for companies with >700M MAU (irrelevant here, but philosophically Apache 2.0 is simpler) and prior "acceptable use" restrictions history. Would be my #2 choice. |
| **Phi-3.5-mini (3.8B)** | Strong reasoning-per-parameter (Microsoft optimized heavily for benchmark reasoning), but historically weaker at multilingual text and less robust at long-context structured extraction; also MIT license is fine, but community fine-tuning tooling/support is thinner than Qwen/Llama. |
| **Gemma 2 2B** | Smaller footprint is attractive, but noticeably weaker instruction-following/JSON reliability at this size in practice, and only an 8K context window — too small if we want few-shot prompting with multiple resume exemplars in-context. Google's Gemma license also has some redistribution terms to review carefully. |
| **Mistral 7B Instruct** | Excellent general model, but 2x+ the parameter count for marginal quality gain on a narrow extraction task — more VRAM, slower inference, higher hosting cost, without a clear accuracy win over a fine-tuned Qwen2.5-3B for this specific narrow domain. Better suited if we later need one model to also handle harder recruiter-reasoning tasks. |
| **Llama 3.1 8B** | Same story as Mistral 7B — better raw capability, but oversized (and more expensive to serve at scale) for a task that fine-tuning can close the gap on with a much smaller, cheaper model. |
| **GPT-4 Mini itself (kept as-is)** | Rejected as the long-term answer because: (a) unbounded per-token cost at scale, (b) no ability to fine-tune on our proprietary resume-parsing patterns / company style, (c) data governance — sending candidate PII to a third-party API is a real compliance concern for an HR/recruitment platform, (d) vendor lock-in risk. |

## 4. Decision Summary

**Primary recommendation: Qwen2.5-3B-Instruct**, fine-tuned with QLoRA on a resume-parsing/generation instruction dataset, served via vLLM or Ollama behind a FastAPI layer, with GPT-4 Mini (or a larger open model) kept only as an optional fallback for the small subset of complex recruiter-reasoning tasks that a 3B model genuinely cannot handle reliably.

**Fallback recommendation: Llama 3.2 3B Instruct**, if Qwen2.5 fine-tuning results underperform expectations in evaluation (Part 4 gives the exact methodology to make that call objectively rather than by gut feel).
