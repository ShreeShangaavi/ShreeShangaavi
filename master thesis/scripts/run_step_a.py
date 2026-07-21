"""Run Step A (constraint decomposition) over the full test-0.1 split.

Reads the 164 official STARK-Amazon test-0.1 queries, runs the Qwen3
constraint decomposition prompt on each, and writes results to a JSONL file.

Usage:
    PYTHONPATH=. python scripts/run_step_a.py
    PYTHONPATH=. python scripts/run_step_a.py --config configs/base.yaml
    PYTHONPATH=. python scripts/run_step_a.py --dry-run  # first 5 queries only

Output:
    results/step_a_outputs.jsonl — one JSON object per line:
    {
        "query_id": 173,
        "query": "Can you recommend a Thill brand fishing bobber?...",
        "answer_ids": [2532],
        "RelationalRequirements": [{"relation": "has_brand", "value": "Thill"}],
        "TextualProperties": ["fishing bobber"],
        "error": null
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from stark_qa import load_qa

from src.step_a_decompose import decompose_query
from src.utils.api_client import make_client
from src.utils.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step A over test-0.1 split")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to base config YAML (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--output",
        default="results/step_a_outputs.jsonl",
        help="Output JSONL file path (default: results/step_a_outputs.jsonl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on first 5 queries only to verify setup",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait between API calls (default: 0.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config and build API client
    cfg = load_config(args.config)
    client = make_client(cfg)

    # Load QA dataset and get test-0.1 split
    logger.info("Loading STARK-Amazon QA dataset...")
    qa_dataset = load_qa("amazon")
    idx_split = qa_dataset.get_idx_split()
    test_01_ids = [int(i) for i in idx_split[cfg.evaluation.rerank_split]]
    logger.info(f"Loaded {len(test_01_ids)} queries from '{cfg.evaluation.rerank_split}' split")

    # Dry run: first 5 queries only
    if args.dry_run:
        test_01_ids = test_01_ids[:5]
        logger.info("DRY RUN: processing first 5 queries only")

    # Prepare output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run Step A
    n_total = len(test_01_ids)
    n_success = 0
    n_error = 0

    logger.info(f"Starting Step A decomposition for {n_total} queries...")
    logger.info(f"Output: {output_path}")

    with open(output_path, "w") as out_f:
        for i, idx in enumerate(test_01_ids):
            query, query_id, answer_ids, _ = qa_dataset[idx]
            query_id = int(query_id)
            answer_ids = [int(a) for a in answer_ids]

            # Progress log every 10 queries
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{n_total} queries processed "
                           f"({n_success} success, {n_error} errors)")

            # Run decomposition
            error = None
            relational = []
            textual = []

            try:
                result = decompose_query(query, client, cfg)
                relational = result["RelationalRequirements"]
                textual = result["TextualProperties"]
                n_success += 1
            except Exception as e:
                error = str(e)
                n_error += 1
                logger.warning(f"query_id={query_id} failed: {error}")

            # Write result
            record = {
                "query_id": query_id,
                "query": query,
                "answer_ids": answer_ids,
                "RelationalRequirements": relational,
                "TextualProperties": textual,
                "error": error,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()  # flush after each write so partial results are saved

            # Optional delay between calls
            if args.delay > 0 and i < n_total - 1:
                time.sleep(args.delay)

    # Final summary
    logger.info("=" * 60)
    logger.info(f"Step A complete.")
    logger.info(f"  Total queries : {n_total}")
    logger.info(f"  Success       : {n_success}")
    logger.info(f"  Errors        : {n_error}")
    logger.info(f"  Output saved  : {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()