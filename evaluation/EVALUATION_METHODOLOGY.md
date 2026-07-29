# Part 4 — Evaluation Framework for the Resume Intelligence Model

`evaluate.py` automates layers 1-4 below. Run it with:
```
python evaluate.py --predictions predictions.jsonl --ground_truth ../dataset/resume_parsing_dataset.jsonl
```
It was self-tested against a "perfect predictions = ground truth" file (`demo_predictions_selftest.jsonl`)
to confirm the scoring logic itself is correct (100% on every metric except a 10% hallucination
*flag* rate — explained below, this is an expected, informative false-positive, not a bug).

## 1. Metrics

| Metric | What it measures | How |
|---|---|---|
| **JSON validity rate** | Does the model output parseable JSON at all | `try_parse_json()` — attempts direct parse, then strips markdown fences / extracts the first `{...}` block as a fallback (small models often wrap JSON in prose or code fences) |
| **Schema validity rate** | Correct top-level keys and types (lists vs scalars) | `check_schema()` — set-difference against the required key list, type-checks list fields |
| **Field-level Precision/Recall/F1** | Per-field accuracy vs. ground truth | Scalar fields (name/email/phone/location/summary): exact match after normalization (lowercase, strip). List fields (skills/education/experience/certifications/projects/links): set-based F1 over flattened items, so a partially-correct experience list still gets partial credit instead of all-or-nothing |
| **Hallucination flag rate** | % of examples where a scalar field value doesn't appear in the source text | `hallucination_check()` — substring match, case-insensitive |
| **Overall average field score** | Single number to track over time / compare checkpoints | Mean of all per-field scores across all examples |

## 2. Measuring Parsing Accuracy
Field-level F1 (above) is the core number, but a single aggregate hides which fields are weak.
The script reports **per-field averages** (`avg_field_scores`) specifically so we can see, e.g.,
"education extraction is at 0.95 F1 but experience duration parsing is only 0.7" — actionable in
a way a single blended accuracy number isn't. In production I'd track this per-field breakdown
over time (dashboard) and set per-field alert thresholds, since some fields matter more than
others operationally (a wrong `email` breaks candidate outreach; a slightly imperfect `summary`
paraphrase doesn't).

## 3. Detecting Hallucinations
Three layers, from cheapest/automatable to most reliable/expensive:

1. **Substring heuristic** (implemented in `evaluate.py`): flag any scalar field value not found
   verbatim in the source text. Cheap, runs on every example, but has false positives (e.g. the
   model correctly normalizes `"Ahm3dabad"` -> `"Ahmedabad"` for an OCR-noisy input — a *correct*
   extraction that still trips the naive substring check, which is exactly what happened in the
   self-test's 10% flag rate). This is why it's a **flag for review**, not an auto-fail.
2. **NLI/entailment-based check** (recommended next step, not yet implemented): for list fields
   like `experience[].description`, use a lightweight NLI model (e.g. a small cross-encoder) to
   check whether the source text *entails* each generated description, catching subtler
   hallucinations the substring check misses (paraphrased-but-false claims).
3. **Human-in-the-loop spot audits**: a random 5-10% sample of production outputs (or all
   flagged-by-heuristic outputs) routed to human review, both to catch what automation misses and
   to generate new training data (see Part 2 scaling plan, "adversarial/edge-case mining").

## 4. Validating JSON Correctness
Two levels, both automated in `evaluate.py`:
- **Syntactic**: can it be `json.loads()`'d (after stripping common wrapper artifacts)?
- **Semantic/schema**: does it have exactly the 11 required top-level keys, correct types
  (list vs scalar), no extra hallucinated keys? For a stricter production system, this would be
  upgraded to a full **JSON Schema** (`jsonschema` library) with type constraints on nested
  objects too (e.g. `education[].year` should be a string matching a year-ish pattern), which is
  the natural next step beyond this prototype's hand-rolled `check_schema()`.
- In production, invalid JSON should trigger an automatic **retry with a corrective re-prompt**
  ("Your previous output was not valid JSON: <error>. Return only valid JSON.") before ever
  surfacing a raw failure to the user — this is a cheap, high-value reliability layer regardless
  of model quality.

## 5. Comparing Two Model Versions (e.g. base vs. fine-tuned, or v1 vs. v2 adapter)
- Run both models over the **same held-out evaluation set** through `evaluate.py`, producing two
  summary JSON reports.
- Compare `overall_avg_field_score`, `json_validity_rate`, and `hallucination_flag_rate` head to
  head. A fine-tune is only a "win" if it improves the aggregate *without* regressing any
  individual field noticeably (a model that gets great at `skills` but starts hallucinating
  `email` is not an improvement).
- For close calls, run a small **paired human preference eval**: same input, two model outputs
  side-by-side (blind), human picks which is more correct/useful — catches quality differences
  the automated metric doesn't (e.g. more natural `summary` phrasing).
- Track results over time in a simple table/W&B dashboard so decisions aren't made on a single
  run's noise, especially given the eval set is currently small (50 examples).

## 6. Regression Testing After Retraining
- Maintain a **fixed, version-controlled regression set** — never edited casually, only
  deliberately expanded — covering at minimum one example per edge-case category from
  `DATASET_CARD.md`. This *is* `resume_parsing_dataset.jsonl`'s held-out split in this prototype;
  in production it would be a separate, larger, frozen set.
- Every new checkpoint runs through `evaluate.py` against this fixed set before being promoted.
  The script's built-in **quality gate** (`json_validity_rate >= 0.95` and
  `overall_avg_field_score >= 0.85`, exit code 1 on failure) is written specifically so this can
  be wired into a CI step — GitHub Actions can run `evaluate.py` on every new adapter push and
  block merge/deploy on failure.
- Additionally diff **per-category** scores (not just the aggregate) between old and new
  checkpoint, since an aggregate improvement can hide a regression in one specific edge-case
  category (e.g. new model got better at standard resumes but worse at OCR-noisy ones).

## 7. Automation Summary (bonus)
- `evaluate.py` is fully automated: JSON validity, schema validity, field-level F1, and
  heuristic hallucination detection all run without human input, and the script exits non-zero
  on quality-gate failure so it's CI-ready.
- Self-tested end-to-end against ground truth (see `demo_predictions_selftest.jsonl` /
  output above) to confirm the scoring logic is itself correct before trusting it on real
  model output.
- Not automated in this prototype (flagged as next steps above): NLI-based hallucination
  detection, full JSON-Schema nested validation, and human preference eval — all reasonable
  v2 additions once there's a real model checkpoint and real production traffic to learn from.
