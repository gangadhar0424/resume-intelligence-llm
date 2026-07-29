# Resume Intelligence LLM — Domain-Specific Model Prototype

Prototype for replacing GPT-4 Mini with a self-hosted, fine-tunable open-source LLM
(**Qwen2.5-3B-Instruct**) for resume parsing, built for the Hidani Tech AI/LLM Engineer
Intern assessment.

## Project Structure
```
resume-intelligence-llm/
├── research/
│   └── model_selection.md          # Part 1 — model research & selection reasoning
├── dataset/
│   ├── generate_dataset.py         # Part 2 — generates the instruction-tuning dataset
│   ├── resume_parsing_dataset.jsonl  # 50 examples, {instruction, input, output}
│   └── DATASET_CARD.md             # schema explanation + scaling-to-thousands plan
├── training/
│   ├── train_lora.py               # Part 3 — QLoRA fine-tuning script (runnable on GPU)
│   ├── config.yaml                 # training hyperparameters
│   ├── requirements.txt
│   └── PIPELINE_DESIGN.md          # Option B: full pipeline design, cost, risk analysis
├── evaluation/
│   ├── evaluate.py                 # Part 4 — automated evaluation harness
│   ├── EVALUATION_METHODOLOGY.md   # metrics, hallucination detection, regression testing
│   └── demo_predictions_selftest.jsonl  # self-test artifact proving the harness works
├── api/
│   ├── main.py                     # Part 5 — FastAPI inference service
│   ├── config.py                   # centralized configuration
│   ├── schemas.py                  # Pydantic request/response models
│   └── requirements.txt
└── README.md                       # this file
```

## Quickstart

### 1. Run the API (uses your local Ollama `qwen2.5:3b`)
```bash
# Terminal 1
ollama serve
ollama pull qwen2.5:3b   # skip if already pulled

# Terminal 2
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Test it:
```bash
curl -X POST http://localhost:8000/parse-resume \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Jane Doe, jane.doe@email.com, Software Engineer at Acme Corp, 2020-Present. Skills: Python, AWS."}'
```
Health check: `curl http://localhost:8000/health`

### 2. Regenerate / inspect the dataset
```bash
cd dataset
python3 generate_dataset.py
```

### 3. Run the evaluation harness
```bash
cd evaluation
python3 evaluate.py --predictions demo_predictions_selftest.jsonl \
                     --ground_truth ../dataset/resume_parsing_dataset.jsonl
```

### 4. Fine-tune (requires a CUDA GPU — e.g. Google Colab)
```bash
cd training
pip install -r requirements.txt
python3 train_lora.py --config config.yaml
```
See `training/PIPELINE_DESIGN.md` for the full design rationale (this was authored as an
Option B design since no GPU was available in the prototyping environment; the script itself
is real and ready to execute).

## Deliverables Mapping (per assessment rubric)

| Part | Marks | Location |
|---|---|---|
| 1. Research & Model Selection | 20 | `research/model_selection.md` |
| 2. Dataset Creation | 25 | `dataset/` |
| 3. Fine-Tuning / Training Strategy | 25 | `training/` |
| 4. Evaluation Methodology | 20 | `evaluation/` |
| 5. Engineering & Documentation | 10 | `api/`, this README |

## Design Summary
- **Model**: Qwen2.5-3B-Instruct — Apache 2.0, strong structured-output behavior at small
  size, already available locally via Ollama.
- **Fine-tuning**: QLoRA (4-bit base + LoRA adapters), designed for a single consumer/free-tier
  GPU, full config and reasoning in `training/`.
- **Dataset**: 50 hand-curated instruction examples across 21 distinct resume edge-case
  categories (missing fields, OCR noise, table layouts, non-standard headers, etc.) — quality
  over quantity, with a concrete plan to scale to thousands.
- **Evaluation**: automated JSON-validity, schema-validity, field-level F1, and heuristic
  hallucination detection, with a CI-ready pass/fail quality gate.
- **Serving**: FastAPI wrapping Ollama, with input validation, timeouts, and clean error
  responses for every model-backend failure mode.
