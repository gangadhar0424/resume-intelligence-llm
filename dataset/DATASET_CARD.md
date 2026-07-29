# Dataset Card — Resume Parsing Instruction Dataset

## Contents
- `generate_dataset.py` — source of truth; generates `resume_parsing_dataset.jsonl`
- `resume_parsing_dataset.jsonl` — 50 instruction-tuning examples, one JSON object per line
- `category_breakdown.json` — how the 50 examples are distributed across edge-case categories

## Schema

Every example has the same three-key contract:

```json
{
  "instruction": "<fixed task instruction, identical across all examples>",
  "input": "<raw resume text>",
  "output": { "name": ..., "email": ..., "phone": ..., "location": ..., "summary": ...,
              "skills": [...], "education": [...], "experience": [...],
              "certifications": [...], "projects": [...], "links": [...] }
}
```

Design choices and why they matter:
- **Fixed instruction string across all examples.** During fine-tuning we want the model to bind
  strongly to *one* schema/contract, not learn to reinterpret a new instruction each time. In
  production, the instruction is fixed by our own prompt template anyway.
- **`null` for missing scalar fields, `[]` for missing list fields — never guessed values.**
  This is the single most important property of the dataset. If the model ever learns to
  "fill in a plausible email" when none exists, that's a hallucination risk in a system whose
  output may be used to auto-contact candidates or auto-populate ATS records.
- **Nested objects for `education`/`experience`/`projects`** instead of flat strings, so the
  output is directly usable by downstream code without a second parsing pass.

## Coverage (why these 50, not 50 random resumes)

Quality > quantity was the brief, so instead of scraping/generating 50 "normal" resumes, the set
is deliberately stratified across the failure modes that actually break naive resume parsers in
production:

| Category | Count | Real-world failure it guards against |
|---|---|---|
| Standard chronological | 5 | Baseline correctness |
| Functional / unstructured career history | 3 | No clean "Experience" section to key off of |
| Academic CV (publications, no industry exp) | 3 | Wrong assumption that everyone has "companies" |
| Missing email | 2 | Model inventing a plausible-looking email |
| Missing phone | 2 | Same, for phone |
| Missing/vague dates | 3 | Model inventing dates instead of returning null |
| Fresher / no work experience | 3 | Model injecting a fake "entry-level" job |
| Career gap | 2 | Model silently dropping or mislabeling the gap |
| Multiple degrees | 2 | Model only capturing the first/last degree |
| Non-standard section headers | 3 | Model relying on literal keyword match ("Experience:") |
| OCR-noisy / garbled text (digit/letter substitution) | 3 | Robustness to scanned-resume artifacts |
| Table/column layout flattened to plain text | 3 | Robustness to PDF-to-text extraction artifacts |
| Bullet-fragment style, no full sentences | 2 | Model over-relying on sentence structure |
| Multiple phone numbers/emails | 2 | Correct disambiguation (prefer personal/mobile) |
| Links present (LinkedIn/GitHub/portfolio) | 3 | Correct population of `links` |
| Certifications only, no formal degree | 2 | Model not inventing a fake degree |
| Freelance/concurrent overlapping roles | 2 | Model handling non-linear timelines |
| Objective statement instead of summary | 2 | Correct mapping of alternate section to `summary` |
| Minimal one-line resume | 1 | Graceful degradation on sparse input |
| Irrelevant personal info (age, marital status) | 1 | Model correctly *ignoring* non-schema fields (GDPR/EEOC-sensitive — should never leak into structured fields) |
| Abbreviated dates + salary-expectation trap | 1 | Model not inventing a `salary` field that isn't in the schema |

21 distinct edge-case categories across 50 examples — every example earns its place rather than
padding a "happy path" majority.

## How to Scale This to Thousands of Examples

A 50-example set is enough to *demonstrate* schema-following via LoRA, but production fine-tuning
would want 2,000–10,000+ examples. Scaling plan:

1. **Real resumes + PII scrubbing (primary source).**
   Pull a large, permissively-licensed resume corpus (e.g. Kaggle's public resume datasets, or
   our own platform's historical resumes, hopefully we have this already since we're a
   recruitment firm). Run a PII-scrubbing pass (regex + NER) to replace real names/emails/phones
   with synthetic but structurally-realistic ones — this preserves layout/language diversity
   while removing compliance risk, since we cannot train on candidates' real PII without consent.

2. **LLM-assisted synthetic generation + programmatic ground truth.**
   Rather than "ask GPT-4 to write a resume and a JSON," which risks GPT-4 hallucinating its own
   ground truth, invert the process: **generate the structured JSON first** (randomly sample
   name, skills-from-a-taxonomy, education, work history with realistic date logic) and then
   **prompt a strong model to render that JSON into natural resume prose** in a randomly chosen
   layout/style/language. This guarantees the `output` is always 100% correct by construction —
   the LLM only does the (easier, lower-hallucination-risk) job of prose generation, not of
   inventing facts we then trust as ground truth.

3. **Layout/format augmentation.**
   Take existing (text, JSON) pairs and mechanically re-render the same JSON in 5-10 different
   surface forms: bullet-only, table-flattened, OCR-noise-injected (swap `s`->`5`, `o`->`0` at a
   controlled rate), different section-header vocabularies, different date formats
   (`Jan 2020`/`01/2020`/`2020-01`/`Jan'20`). This multiplies each hand-verified example ~8x
   without needing new ground truth each time — same output JSON, held constant.

4. **Adversarial/edge-case mining from production.**
   Once a v1 model is live behind human review (recruiters correcting AI-parsed fields), every
   correction is a free, high-value training example — real distribution, real failure modes,
   already labeled by a human. This is the highest-ROI long-term source and should be wired up
   from day one (log low-confidence or human-corrected extractions to a review queue -> dataset).

5. **Stratified sampling to keep balance.**
   As volume grows, explicitly track category counts (like `category_breakdown.json` here) and
   actively re-balance — it's easy for a scraped corpus to become 90% "standard chronological
   software engineer resumes" and quietly starve the edge cases that actually matter for
   robustness.

6. **Deduplication and near-duplicate filtering.**
   Use embedding similarity (e.g. MiniLM embeddings + cosine threshold) to catch near-duplicate
   resumes before they inflate the dataset without adding real diversity.

7. **Automated quality gate before anything enters the training set.**
   Every candidate example passes: (a) JSON schema validation, (b) a rule check that no `output`
   field contains a string not substringable to the `input` (catches hallucinated facts),
   (c) a held-out human spot-check on a random 5% sample per batch.
