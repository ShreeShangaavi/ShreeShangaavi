"""Step C: LLM Verification.

Scores each of the top cfg.step_c.k_verify candidates in [0, 1] against the
decomposed constraints and a candidate card (graph facts + BM25 snippets).
Returns candidates re-ranked by score.

Call cfg.assert_step_c_model_set() at the start to fail fast if model_name
is still null (i.e. Model Comparison Study not yet completed).
"""
# TODO: implement
