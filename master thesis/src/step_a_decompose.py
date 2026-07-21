"""Step A: Constraint Decomposition via LLM.

Takes a raw STARK-Amazon query and returns a structured JSON with:
  - RelationalRequirements: graph-traversable constraints
  - TextualProperties: free-text constraints for dense/BM25 retrieval

Model: Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 (non-thinking mode)
API:   vLLM server at http://localhost:8002/v1
"""

from __future__ import annotations
import json
import logging

from src.utils.api_client import chat_completion, make_client
from src.utils.config_loader import PipelineConfig, load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a query parser for the STARK-Amazon product knowledge graph.

Your task is to decompose a natural-language product search query into two parts:
1. RelationalRequirements — constraints that map to explicit graph edges in the product KG
2. TextualProperties — all remaining product requirements expressed as short phrases

## Knowledge graph schema
The product graph has these relation types only:
- has_brand      → the product's manufacturer or brand name (e.g. "Sony", "Thill", "Reebok")
- has_category   → the product's category (e.g. "fishing bobbers", "headphones")
- has_color      → the product's color (e.g. "black", "red")
- also_bought    → products frequently bought together with a named product
- also_viewed    → products frequently viewed together with a named product

## Output schema
Return ONLY a JSON object with this exact structure:
{
  "RelationalRequirements": [
    {"relation": "<relation_type>", "value": "<entity_string>"}
  ],
  "TextualProperties": ["<phrase1>", "<phrase2>", ...]
}

## Rules
1. Only extract a RelationalRequirement if the query explicitly names a brand, category,
   color, or product. Never invent or infer one that is not clearly stated.
2. A brand mentioned as a compatibility target (e.g. "fits Colt 1911", "compatible with Sony")
   is NOT a has_brand constraint — it belongs in TextualProperties.
3. Vague references like "reputable brand", "well-known manufacturer", or "popular brand"
   are NOT extractable — put them in TextualProperties.
4. If there are no relational constraints, return an empty list for RelationalRequirements.
5. Output only the JSON object — no markdown fences, no explanation, no preamble.

## Examples

Query: "Can you recommend a Thill brand fishing bobber? I specifically prefer this brand."
Output:
{
  "RelationalRequirements": [
    {"relation": "has_brand", "value": "Thill"}
  ],
  "TextualProperties": ["fishing bobber"]
}

Query: "Can you suggest a Peyton Manning youth football jersey that includes authentic NFL and Reebok logos?"
Output:
{
  "RelationalRequirements": [
    {"relation": "has_brand", "value": "NFL"},
    {"relation": "has_brand", "value": "Reebok"}
  ],
  "TextualProperties": ["Peyton Manning youth football jersey", "authentic logos"]
}

Query: "Is there a durable, waterproof trail map available for hiking and biking that can withstand rainy conditions?"
Output:
{
  "RelationalRequirements": []
  "TextualProperties": ["durable", "waterproof", "trail map", "hiking and biking", "withstand rainy conditions"]
}

Query: "What's the best scope mount for a Colt 1911 and Colt Government 45 that still allows for iron sights usage and optics mounting? Also, I need it to be compatible with non-ambidextrous safety."
Output:
{
  "RelationalRequirements": []
  "TextualProperties": ["scope mount", "compatible with Colt 1911 and Colt Government 45", "allows iron sights usage", "optics mounting", "compatible with non-ambidextrous safety"]
}"""

USER_TEMPLATE = "Query: {query}\nOutput:"

# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def decompose_query(
    query: str,
    client,
    cfg: PipelineConfig,
) -> dict:
    """Decompose a single query into relational and textual constraints.

    Returns a dict with keys RelationalRequirements and TextualProperties.
    Raises ValueError if the model output cannot be parsed as valid JSON.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(query=query)},
    ]

    raw = chat_completion(
        client=client,
        model=cfg.step_a.model_name,
        messages=messages,
        temperature=cfg.step_a.temperature,
        max_tokens=cfg.step_a.max_tokens,
    )

    # Strip accidental markdown fences if model adds them despite instructions
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Step A: model output is not valid JSON.\n"
            f"Raw output: {raw}\n"
            f"Error: {e}"
        )

    # Validate expected keys
    if "RelationalRequirements" not in result or "TextualProperties" not in result:
        raise ValueError(
            f"Step A: missing required keys in output.\n"
            f"Got keys: {list(result.keys())}"
        )

    return result


# ---------------------------------------------------------------------------
# Quick test (run directly to verify prompt)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)

    cfg = load_config("configs/base.yaml")
    client = make_client(cfg)

    test_queries = [
        "Can you recommend a Thill brand fishing bobber? I specifically prefer this brand.",
        "Is there a durable, waterproof trail map available for hiking and biking that can withstand rainy conditions?",
        "What's the best scope mount for a Colt 1911 and Colt Government 45 that still allows for iron sights usage and optics mounting?",
        "Can you suggest a Peyton Manning youth football jersey that includes authentic NFL and Reebok logos?",
        "Is there a fully functional and certified Verizon LG VX4700 No Contract Feature Cell Phone available?",
    ]

    for i, query in enumerate(test_queries):
        print(f"\n{'='*60}")
        print(f"[{i}] {query}")
        print("-" * 60)
        try:
            result = decompose_query(query, client, cfg)
            print(json.dumps(result, indent=2))
        except ValueError as e:
            print(f"ERROR: {e}")