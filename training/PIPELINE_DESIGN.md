# Part 3 — Fine-Tuning Pipeline Design (Option B)

No GPU was available in this environment, so this is a full pipeline design rather than an
executed run. `train_lora.py` and `config.yaml` in this folder are real, ready-to-execute
artifacts — the design below explains the reasoning behind every choice and what I'd expect
to observe, so this can be picked up and run as-is on Colab or a cloud GPU.

## 1. Model Choice
**Qwen2.5-3B-Instruct**, per the Part 1 research doc. Small enough for QLoRA on a single
free-tier Colab T4 (16GB VRAM, but 4-bit-quantized 3B model + LoRA adapters need well under
that — realistically 8-10GB), already instruction-tuned so we're doing *adaptation* not
teaching it English/JSON from scratch.

## 2. Dataset Preparation
- Source: the 50-example `resume_parsing_dataset.jsonl` (Part 2), scaled per the plan in
  `DATASET_CARD.md` before a production run (target: 2,000-5,000 examples for a real deployment).
- Each `{instruction, input, output}` record is rendered into the Qwen chat template
  (system/user/assistant turns) with the assistant turn being the target JSON string —
  see `format_example()` in `train_lora.py`.
- 80/20 train/val split, stratified by category where possible so validation loss actually
  reflects performance across edge cases, not just the majority "standard resume" category.
- Loss is computed only on the assistant (completion) tokens, not the prompt — standard SFT
  masking, handled automatically by `SFTTrainer`.

## 3. Training Strategy: QLoRA
- **Why QLoRA over full fine-tuning:** full fine-tuning a 3B model needs ~40GB+ VRAM (params +
  gradients + optimizer states in fp32/fp16) — not accessible on free/cheap tiers. QLoRA
  quantizes the frozen base model to 4-bit (NF4) and trains small low-rank adapter matrices
  in bf16 on top, cutting VRAM needs by ~4-6x with minimal quality loss for narrow-domain tasks.
- **Why QLoRA over full LoRA (no quantization):** at 3B params, plain LoRA (fp16 base) is
  already feasible on a T4, so QLoRA isn't strictly necessary at this size — but designing for
  QLoRA from the start means the same pipeline scales cleanly if we later move to a 7B/8B base
  model for harder tasks (cover-letter generation, recruiter reasoning) without a redesign.
- **LoRA target modules:** all attention projections (`q,k,v,o_proj`) + all MLP projections
  (`gate,up,down_proj`). Attention-only LoRA is cheaper but MLP layers carry a lot of the
  "formatting/output-structure" behavior — including them matters specifically for a
  JSON-structured-output task.
- **Rank (r=16), alpha (32):** conservative middle ground. With only 50-2,000 examples, a
  higher rank (64+) risks overfitting the adapter to memorized examples rather than learning
  the general schema-following pattern; r=16 is enough capacity to shift output formatting
  and field-selection behavior.
- **Epochs (3) + small effective batch size (16):** small-dataset regime — enough passes to
  converge, but eval loss must be watched every 10 steps to catch overfitting early
  (early stopping / picking the best checkpoint by eval loss, not just the last one).

## 4. Hardware Requirements

| Stage | Hardware | Est. time (50-example set) | Est. time (2,000-example set) |
|---|---|---|---|
| QLoRA fine-tune | 1x T4 (16GB) — Colab free tier | ~5-10 min | ~2-3 hours |
| QLoRA fine-tune | 1x A10/A100 (24-40GB) | ~2-3 min | ~30-45 min |
| Inference (4-bit) | 1x T4 or CPU (Ollama) | N/A | N/A |
| Merged-weights serving (vLLM, fp16) | 1x T4/L4 (16-24GB) | N/A | N/A |

## 5. Evaluation Plan
Full detail in `../evaluation/`. Summary: automated JSON-schema validation, field-level
precision/recall against ground truth, hallucination detection (flagging any output value not
traceable to the input text), and human spot-checks on the held-out validation split before
and after fine-tuning — comparing the base Qwen2.5-3B-Instruct's zero-shot parsing accuracy to
the fine-tuned adapter's accuracy on the same held-out examples.

## 6. Estimated Costs
| Item | Cost |
|---|---|
| Colab free tier (T4) — sufficient for 50-2,000 example QLoRA runs | $0 |
| Colab Pro (faster T4/A100 access, longer sessions) | ~$10/month |
| Cloud GPU on-demand (e.g. AWS `g5.xlarge`, ~$1/hr) for a 2,000-example run (~1hr) | ~$1-3 per run |
| W&B experiment tracking (free tier) | $0 |
| Total to reach a production-quality fine-tuned checkpoint, including iteration (5-10 runs) | Well under $50 |

This is the core economic argument from Part 1: replacing GPT-4 Mini's per-token cost with a
self-hosted 3B model has a fine-tuning cost measured in dollars, not thousands of dollars, and
an inference cost that becomes a fixed (small) infra bill instead of a variable API bill.

## 7. Potential Risks and Mitigation

| Risk | Mitigation |
|---|---|
| Overfitting on a small (50-example) dataset — model memorizes rather than generalizes | Watch eval loss per-step, keep val split honest (20%), scale dataset before any production deployment, use early stopping on eval loss plateau |
| Catastrophic forgetting of general instruction-following ability | Keep LoRA rank modest, keep base model frozen (only adapters trained), periodically eval on a few *non*-resume prompts to confirm general chat ability is intact |
| Hallucinated fields in output (invented email/phone/dates) | Dataset explicitly trains `null`/`[]` for missing fields (Part 2); Part 4 evaluation includes an automated hallucination check (no output value should be absent from the input text, except for standard normalization e.g. date reformatting) |
| Schema drift — model outputs valid-looking JSON but wrong keys/types | Always validate against a strict JSON Schema post-generation; reject and retry (or fall back to a stricter re-prompt) on failure — never trust raw model output directly in production |
| PII/compliance risk in training data | Only train on scrubbed/synthetic or consented data (Part 2 scaling plan); never send resumes to third-party APIs for synthetic augmentation without a data processing agreement |
| Adapter/base-model version mismatch in production | Pin exact base model revision + adapter version together in `config.py`/deployment manifest; never silently auto-update the base model |
| Regression after retraining on new data | Maintain a fixed regression test set (Part 4) that every new checkpoint must pass before promotion |
