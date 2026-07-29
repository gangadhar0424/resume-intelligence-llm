"""
evaluate.py — Automated evaluation harness for the Resume Intelligence model.

Runs three layers of checks against a set of {instruction, input, output(=ground truth)}
examples and a model's predictions:

  1. JSON validity      — did the model produce parseable JSON at all?
  2. Schema correctness — does it have exactly the right keys/types?
  3. Field-level accuracy — precision/recall/F1 per field vs. ground truth
  4. Hallucination check — does every extracted string actually appear (or closely match
     a normalized form of something) in the source resume text?

This script is model-agnostic: it takes a `predict_fn(instruction, input_text) -> str`
callable, so the same harness evaluates the base model, the fine-tuned model, or two
checkpoints against each other for regression testing.

Usage:
    python evaluate.py --predictions predictions.jsonl --ground_truth ../dataset/resume_parsing_dataset.jsonl
"""
import argparse
import json
import re
import sys
from collections import defaultdict

REQUIRED_KEYS = {
    "name", "email", "phone", "location", "summary", "skills",
    "education", "experience", "certifications", "projects", "links",
}
LIST_KEYS = {"skills", "education", "experience", "certifications", "projects", "links"}


def try_parse_json(raw: str):
    """Attempts to parse model output as JSON, stripping common wrapper artifacts
    (markdown code fences, leading/trailing prose) that small models sometimes emit."""
    raw = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace_match:
            raw = brace_match.group(0)
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as e:
        return None, str(e)


def check_schema(parsed: dict) -> list:
    """Returns a list of schema violations (empty list = fully valid)."""
    errors = []
    if not isinstance(parsed, dict):
        return ["top-level output is not a JSON object"]
    missing = REQUIRED_KEYS - set(parsed.keys())
    extra = set(parsed.keys()) - REQUIRED_KEYS
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected extra keys: {sorted(extra)}")
    for k in LIST_KEYS & set(parsed.keys()):
        if not isinstance(parsed[k], list):
            errors.append(f"'{k}' should be a list, got {type(parsed[k]).__name__}")
    return errors


def normalize(v):
    if v is None:
        return ""
    return str(v).strip().lower()


def field_level_score(pred: dict, gold: dict) -> dict:
    """
    Field-level precision/recall for scalar fields (exact-ish match) and list fields
    (set-based overlap, since order shouldn't matter for e.g. skills lists).
    """
    scores = {}
    scalar_fields = ["name", "email", "phone", "location", "summary"]
    for f in scalar_fields:
        gold_v = normalize(gold.get(f))
        pred_v = normalize(pred.get(f)) if isinstance(pred, dict) else ""
        if gold_v == "" and pred_v == "":
            scores[f] = 1.0  # both correctly null
        elif gold_v == "":
            scores[f] = 0.0 if pred_v else 1.0  # gold null, pred hallucinated something
        else:
            scores[f] = 1.0 if gold_v == pred_v else 0.0

    for f in LIST_KEYS:
        gold_list = gold.get(f) or []
        pred_list = pred.get(f) if isinstance(pred, dict) else []
        pred_list = pred_list or []

        def flatten(item):
            if isinstance(item, dict):
                return normalize(json.dumps(item, sort_keys=True))
            return normalize(item)

        gold_set = {flatten(i) for i in gold_list}
        pred_set = {flatten(i) for i in pred_list}

        if not gold_set and not pred_set:
            scores[f] = 1.0
            continue
        tp = len(gold_set & pred_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(gold_set) if gold_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        scores[f] = f1

    return scores


def hallucination_check(pred: dict, source_text: str) -> list:
    """
    Flags any non-null scalar value in the prediction that doesn't appear (case-insensitive,
    loose substring) anywhere in the source resume text. This is a heuristic, not a proof of
    non-hallucination -- legitimate normalization (e.g. reformatting a date) can trigger a
    false positive, which is why this feeds a human-review queue rather than an auto-reject.
    """
    flags = []
    if not isinstance(pred, dict):
        return ["prediction not a dict, cannot check"]
    source_lower = source_text.lower()
    for f in ["name", "email", "phone", "location"]:
        v = pred.get(f)
        if v and str(v).lower() not in source_lower:
            flags.append(f"'{f}' value '{v}' not found verbatim in source text")
    return flags


def evaluate(predictions_path: str, ground_truth_path: str):
    with open(ground_truth_path, encoding="utf-8") as f:
        gold_records = [json.loads(l) for l in f]
    with open(predictions_path, encoding="utf-8") as f:
        pred_records = [json.loads(l) for l in f]

    assert len(gold_records) == len(pred_records), "predictions and ground truth must be aligned 1:1"

    n = len(gold_records)
    json_valid_count = 0
    schema_valid_count = 0
    field_scores_agg = defaultdict(list)
    hallucination_count = 0
    per_example_results = []

    for gold, pred_record in zip(gold_records, pred_records):
        raw_output = pred_record.get("raw_output", "")
        parsed, parse_err = try_parse_json(raw_output)

        result = {"input_preview": gold["input"][:60] + "..."}

        if parsed is None:
            result["json_valid"] = False
            result["parse_error"] = parse_err
            per_example_results.append(result)
            continue

        json_valid_count += 1
        result["json_valid"] = True

        schema_errors = check_schema(parsed)
        result["schema_errors"] = schema_errors
        if not schema_errors:
            schema_valid_count += 1

        f_scores = field_level_score(parsed, gold["output"])
        result["field_scores"] = f_scores
        for k, v in f_scores.items():
            field_scores_agg[k].append(v)

        halluc_flags = hallucination_check(parsed, gold["input"])
        result["hallucination_flags"] = halluc_flags
        if halluc_flags:
            hallucination_count += 1

        per_example_results.append(result)

    summary = {
        "total_examples": n,
        "json_validity_rate": round(json_valid_count / n, 3),
        "schema_validity_rate": round(schema_valid_count / n, 3),
        "hallucination_flag_rate": round(hallucination_count / n, 3),
        "avg_field_scores": {k: round(sum(v) / len(v), 3) for k, v in field_scores_agg.items()},
        "overall_avg_field_score": round(
            sum(sum(v) for v in field_scores_agg.values()) /
            sum(len(v) for v in field_scores_agg.values()), 3
        ) if field_scores_agg else 0.0,
    }

    return summary, per_example_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                         help="JSONL with one {'raw_output': '<model's raw text output>'} per line, "
                              "aligned 1:1 with ground_truth rows")
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    summary, details = evaluate(args.predictions, args.ground_truth)

    print(json.dumps(summary, indent=2))
    if args.verbose:
        for d in details:
            print(json.dumps(d, indent=2))

    # Simple regression gate: exit non-zero if quality drops below thresholds,
    # so this can be wired into CI (see README).
    if summary["json_validity_rate"] < 0.95 or summary["overall_avg_field_score"] < 0.85:
        print("\nFAILED quality gate (json_validity_rate < 0.95 or overall_avg_field_score < 0.85)",
              file=sys.stderr)
        sys.exit(1)
    print("\nPASSED quality gate")
