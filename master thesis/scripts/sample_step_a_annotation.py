"""Generate a blank annotation template for Step A quality evaluation.

Randomly samples 10% of the Step A outputs (~16 queries from 164) using
the fixed random seed, and writes a JSONL template for manual annotation.

Usage:
    PYTHONPATH=. python scripts/sample_step_a_annotation.py

Output:
    results/step_a_annotation_template.jsonl — blank template to fill in
    results/step_a_annotation.jsonl          — your completed annotations (rename when done)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Fixed seed — same as thesis random_seed
RANDOM_SEED = 4000243
SAMPLE_FRACTION = 0.10

INPUT_PATH = Path("results/step_a_outputs.jsonl")
OUTPUT_PATH = Path("results/step_a_annotation_template.jsonl")


def main() -> None:
    # Load all Step A outputs
    records = [json.loads(line) for line in INPUT_PATH.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(records)} Step A outputs")

    # Sample 10%
    n_sample = max(1, round(len(records) * SAMPLE_FRACTION))
    random.seed(RANDOM_SEED)
    sampled = random.sample(records, n_sample)
    sampled.sort(key=lambda r: r["query_id"])  # sort for readability

    print(f"Sampled {n_sample} queries (seed={RANDOM_SEED})")

    # Write blank annotation template
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for r in sampled:
            annotation = {
                "query_id": r["query_id"],
                "query": r["query"],
                "RelationalRequirements": r["RelationalRequirements"],
                "TextualProperties": r["TextualProperties"],
                # --- Fill in these fields manually ---
                "relational_correct": None,   # true / false
                "textual_correct": None,       # true / false
                "relational_error_type": None, # null | "hallucinated" | "wrong_relation" | "missing" | "wrong_value"
                "textual_error_type": None,    # null | "missing_key_property" | "too_coarse" | "too_granular"
                "notes": ""                    # any free-text observation
            }
            f.write(json.dumps(annotation) + "\n")

    print(f"Annotation template written to: {OUTPUT_PATH}")
    print()
    print("Instructions:")
    print("  1. Open results/step_a_annotation_template.jsonl")
    print("  2. For each record, fill in:")
    print("       relational_correct: true if RelationalRequirements are correct, false otherwise")
    print("       textual_correct:    true if TextualProperties capture key requirements, false otherwise")
    print("       relational_error_type: null if correct, else one of:")
    print("         'hallucinated'   — brand/category invented, not in query")
    print("         'wrong_relation' — e.g. used has_brand when should be textual")
    print("         'missing'        — a clear relational constraint was missed")
    print("         'wrong_value'    — relation type correct but value wrong")
    print("       textual_error_type: null if correct, else one of:")
    print("         'missing_key_property' — important requirement not captured")
    print("         'too_coarse'           — properties too vague to be useful")
    print("         'too_granular'         — over-split into too many tiny fragments")
    print("  3. Save as results/step_a_annotation.jsonl when complete")
    print()
    print(f"Sampled query IDs: {[r['query_id'] for r in sampled]}")


if __name__ == "__main__":
    main()